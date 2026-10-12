"""Numerical gates for the split, XLA updates, data partitioning, and resume."""

from dataclasses import replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from rocora.checkpoint import save_checkpoint, load_checkpoint
from rocora.config import ModelConfig, OptimizerConfig, ExperimentConfig, load_config
from rocora.curvature import coded_probe, core_sample, residual_diagonal
from rocora.data import TokenBatches
from rocora.model import init_params, logits, loss_sum, decay_mask
from rocora.optimizers import init_state, update, commit_curvature
from rocora.prepare import TokenWriter, is_validation_document


def test_sophia_is_exact_core_only_control():
    size=1024
    params={"weights":jnp.ones((size,),jnp.float32)}
    base=OptimizerConfig(name="sophia")
    res=replace(base,name="rocora",residual_weight=0)
    a,b=init_state(params,base),init_state(params,res)
    sample={"weights":jnp.arange(size,dtype=jnp.float32)/100}
    def advance(state,cfg):
        state=commit_curvature(state,sample,None,cfg)
        return update(state,sample,.001,cfg,decay_mask(params))
    core_step=jax.jit(lambda state:advance(state,base))
    residual_step=jax.jit(lambda state:advance(state,res))
    for _ in range(5):
        a,b=core_step(a),residual_step(b)
        np.testing.assert_array_equal(a.params["weights"],b.params["weights"])
        np.testing.assert_array_equal(a.optimizer.core["weights"],b.optimizer.core["weights"])


def test_fixed_snapshot_residual_matches_dense_hessian_split():
    cfg=ModelConfig(width=2,layers=1,heads=1,sequence_length=2,vocab_size=4,valid_vocab_size=4,
                    attention="reference",loss_chunk=2,remat=False)
    params=init_params(cfg,8)
    batch={"tokens":jnp.asarray([[0,1]]),"targets":jnp.asarray([[1,2]]),"mask":jnp.ones((1,2))}
    flat,unpack=ravel_pytree(params)
    f=lambda z:logits(unpack(z),batch["tokens"],cfg,curvature=True).reshape(-1)
    J=jax.jacrev(f)(flat).reshape(2,4,-1)
    probabilities=jax.nn.softmax(f(flat).reshape(2,4),axis=-1)
    lambdas=jax.vmap(lambda p:jnp.diag(p)-jnp.outer(p,p))(probabilities)/2
    S=jnp.einsum("tvi,tvw,twj->ij",J,lambdas,J)
    H=jax.hessian(lambda z:loss_sum(unpack(z),batch,cfg,curvature=True)/2)(flat)
    R=H-S
    key=jax.random.PRNGKey(41)
    expected=jnp.zeros_like(flat)
    for row in range(4):
        v,_=ravel_pytree(coded_probe(params,key,row,4))
        expected+=v*(R@v)/4
    actual,_=ravel_pytree(residual_diagonal(params,batch,cfg,key,4)[0])
    np.testing.assert_allclose(actual,expected,rtol=4e-3,atol=4e-4)
    assert float(jnp.linalg.norm(R))>1e-5


def test_core_sample_is_nonnegative_and_correctly_normalized():
    cfg=ModelConfig(width=2,layers=1,heads=1,sequence_length=2,vocab_size=4,valid_vocab_size=4,
                    attention="reference",loss_chunk=2,remat=False)
    params=init_params(cfg,3)
    batch={"tokens":jnp.asarray([[0,1]]),"targets":jnp.asarray([[1,2]]),"mask":jnp.ones((1,2))}
    key=jax.random.PRNGKey(12)
    sample,count=core_sample(params,batch,cfg,key)
    grads=jax.grad(lambda p:loss_sum(p,batch,cfg,sample_key=key)/2)(params)
    assert float(count)==2
    for actual,g in zip(jax.tree.leaves(sample),jax.tree.leaves(grads)):
        assert np.all(np.asarray(actual)>=0)
        np.testing.assert_allclose(actual,2*g*g,rtol=1e-6)


def test_disjoint_hosts_and_exact_final_token_budget(tmp_path):
    path=tmp_path/"tokens.npy"
    np.save(path,np.arange(101,dtype=np.uint16))
    loaders=[TokenBatches(path,4,8,4,i,seed=5,token_budget=37) for i in range(4)]
    first=[loader.batch(0) for loader in loaders]
    starts=np.concatenate([b["tokens"][:,0] for b in first])
    assert len(np.unique(starts))==8
    assert sum(b["mask"].sum() for b in first)==32
    last=[loader.batch(1) for loader in loaders]
    assert sum(b["mask"].sum() for b in last)==5
    np.testing.assert_array_equal(first[0]["tokens"],TokenBatches(path,4,8,4,0,seed=5,token_budget=37).batch(0)["tokens"])


def test_checkpoint_preserves_all_channels_and_rejects_wrong_seed(tmp_path):
    cfg=ExperimentConfig(model=ModelConfig(attention="reference"))
    params={"w":jnp.ones((2,3))}
    state=init_state(params,cfg.optimizer)
    sample={"w":jnp.full((2,3),.2)}
    state=commit_curvature(state,sample,sample,cfg.optimizer)
    path=tmp_path/"checkpoint.npz"
    save_checkpoint(path,state,cfg,12.5)
    restored,seconds=load_checkpoint(path,init_state(params,cfg.optimizer),cfg)
    assert seconds==12.5
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match="differs"):
        load_checkpoint(path,state,replace(cfg,seed=42))


def test_stage_counts_and_sophia_rocora_settings():
    root=Path(__file__).resolve().parents[1]
    for path,expected in (("pilot60m.json",60_894_720),("scale125m.json",124_373_760)):
        cfg=load_config(root/"configs"/path,"rocora")
        assert cfg.model.parameter_count==expected
        sophia=load_config(root/"configs"/path,"sophia")
        for field in ("learning_rate","beta1","beta2","rho","eps","weight_decay","core_interval"):
            assert getattr(cfg.optimizer,field)==getattr(sophia.optimizer,field)
    pod=json.loads((root/"configs/pod.json").read_text())
    assert pod["seeds"]["pilot60m"]==[1337]
    assert pod["seeds"]["scale125m"]==[1337,1338,1339]


def test_token_writer_and_document_split(tmp_path):
    writer=TokenWriter(tmp_path/"train.npy")
    writer.append([1,2,3])
    writer.append([4,5,6],limit=5)
    writer.finish()
    np.testing.assert_array_equal(np.load(tmp_path/"train.npy"),[1,2,3,4,5])
    assert is_validation_document("example")==is_validation_document("example")


def test_exclude_document_shared_with_validation(tmp_path):
    path=tmp_path/"tokens.npy"
    np.save(path,np.asarray([2,3,50256,5,6,7,8,9,10,11,12],np.uint16))
    loader=TokenBatches(path,2,2,shuffle=False,skip_initial_document=True)
    assert loader.skipped_tokens==3
    np.testing.assert_array_equal(loader.batch(0)["tokens"],[[5,6],[7,8]])


def test_report_rejects_partial_budgets_and_confounded_controls(tmp_path):
    from rocora.report import summarize
    base=ExperimentConfig(model=ModelConfig(attention="reference"))
    paths=[]
    for i,arm in enumerate(("adamw","sophia","rocora")):
        cfg=replace(base,optimizer=replace(base.optimizer,name=arm))
        rows=[{"type":"meta","seed":1337,"optimizer":arm,"config":cfg.to_dict(),"code_sha256":"same",
               "devices":1,"processes":1,"jax":"test","device_kind":"CPU"},
              {"type":"eval","step":1,"tokens":10,"loss":5.,"train_seconds":1.},
              {"type":"eval","step":2,"tokens":cfg.train_tokens,"loss":4.-i/10,"train_seconds":10.+i},
              {"type":"complete","budget_complete":True}]
        path=tmp_path/f'{arm}.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        paths.append(path)
    result=summarize(paths)
    assert result["paired"][0]["core_minus_residual_loss_equal_tokens"]>0
    rows[0]["config"]["optimizer"]["rho"]=.5
    paths[-1].write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError,match="confounded"):
        summarize(paths)
    rows[-1]["budget_complete"]=False
    paths[-1].write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError,match="not completed"):
        summarize(paths)


def test_published_adamw_and_sophia_update_rules():
    params={"w":jnp.ones((3,),jnp.float32)}
    grad={"w":jnp.asarray([-2.,0.,.01],jnp.float32)}
    adam=OptimizerConfig(name="adamw",beta1=.9,beta2=.95,eps=1e-8)
    result=jax.jit(lambda state:update(state,grad,.01,adam,{"w":True}))(init_state(params,adam))
    expected=np.ones(3)*(1-.01*adam.weight_decay)-.01*np.asarray(grad["w"])/(np.abs(np.asarray(grad["w"]))+adam.eps)
    np.testing.assert_allclose(result.params["w"],expected,rtol=2e-7,atol=2e-7)
    sophia=OptimizerConfig(name="sophia")
    state=init_state(params,sophia)
    state=state._replace(optimizer=state.optimizer._replace(moment={"w":jnp.full((3,),.0001)},core={"w":jnp.full((3,),.5)}))
    moment=sophia.beta1*.0001+(1-sophia.beta1)*np.asarray(grad["w"])
    expected=np.ones(3)*(1-.01*sophia.weight_decay)-.01*np.clip(moment/(sophia.rho*.5+sophia.eps),-1,1)
    result=jax.jit(lambda state:update(state,grad,.01,sophia,{"w":True}))(state)
    np.testing.assert_allclose(result.params["w"],expected,rtol=2e-7,atol=2e-7)
    rocora=replace(sophia,name="rocora")
    state=init_state(params,rocora)
    state=state._replace(optimizer=state.optimizer._replace(
        moment={"w":jnp.full((3,),.0001)},core={"w":jnp.full((3,),.5)},
        residual={"w":jnp.asarray([-1.,.01,-.5])}))
    scaled_moment=rocora.beta1*.0001+(1-rocora.beta1)*np.asarray(grad["w"])*.7
    denominator=np.maximum(rocora.rho*(.5+np.asarray(state.optimizer.residual["w"])),0)+rocora.eps
    expected=np.ones(3)*(1-.01*rocora.weight_decay)-.01*np.clip(scaled_moment/denominator,-1,1)
    result=jax.jit(lambda state:update(state,grad,.01,rocora,{"w":True},gradient_scale=.7))(state)
    np.testing.assert_allclose(result.params["w"],expected,rtol=2e-7,atol=2e-7)


def test_resume_rejects_a_changed_implementation(tmp_path):
    cfg=ExperimentConfig()
    state=init_state({"w":jnp.ones((2,))},cfg.optimizer)
    path=tmp_path/"checkpoint.npz"
    save_checkpoint(path,state,cfg,1.,code_sha256="original")
    with pytest.raises(ValueError,match="different"):
        load_checkpoint(path,state,cfg,code_sha256="changed")


def test_learning_rate_budget_above_int32_limit():
    from rocora.config import learning_rate
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/"configs/scale125m.json","adamw")
    schedule=jax.jit(lambda step:learning_rate(step,cfg))
    first=float(schedule(jnp.int32(0)))
    last=float(schedule(jnp.int32(cfg.steps-1)))
    assert np.isfinite(first) and first>0
    np.testing.assert_allclose(last,cfg.optimizer.learning_rate*cfg.min_lr_fraction,rtol=1e-6)


def test_worker_lock_and_disconnect_reap_child(tmp_path):
    import os
    import shlex
    import subprocess
    import sys

    root=Path(__file__).resolve().parents[1]
    child_code="import os,time; print(os.getpid(),flush=True); time.sleep(30)"
    command="exec "+shlex.join([sys.executable,"-u","-c",child_code])
    args=[sys.executable,str(root/"scripts/worker.py"),"--lock",str(tmp_path/"pod.lock"),
          "--job-id","test",command]
    first=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert first.stdout.readline().strip()=="ROCORA_WORKER_READY test"
        busy=subprocess.run(args,input="START\n",capture_output=True,text=True,timeout=5)
        assert busy.returncode==75
        assert "Worker busy" in busy.stdout
        first.stdin.write("START\n")
        first.stdin.flush()
        child_pid=int(first.stdout.readline().strip())
        first.stdin.close()
        assert first.wait(timeout=10)==130
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid,0)
        # The kernel releases the flock after cleanup; no stale lock deletion.
        released=subprocess.run(args,input="",capture_output=True,text=True,timeout=5)
        assert released.returncode==130
        assert "ROCORA_WORKER_READY" in released.stdout
        completed=subprocess.Popen(args[:-1]+["true"],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,text=True)
        assert completed.stdout.readline().strip()=="ROCORA_WORKER_READY test"
        completed.stdin.write("START\n")
        completed.stdin.flush()
        assert completed.wait(timeout=5)==0
        completed.stdin.close()
    finally:
        if first.poll() is None:
            first.stdin.close()
            first.wait(timeout=10)
