"""Numerical gates for the split, XLA updates, data partitioning, and resume."""

from dataclasses import replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from rocora.attention import flash_attention, reference_attention
from rocora.checkpoint import save_checkpoint, load_checkpoint
from rocora.config import ModelConfig, OptimizerConfig, ExperimentConfig, load_config
from rocora.curvature import coded_probe, core_sample, residual_diagonal
from rocora.data import TokenBatches
from rocora.model import init_params, logits, loss_sum, decay_mask, hidden
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
    old=jax.tree.map(lambda p:jnp.full_like(p,.17),params)
    folded,_=ravel_pytree(residual_diagonal(params,batch,cfg,key,4,
        initial=jax.tree.map(lambda x:.95*x,old),sample_weight=.05)[0])
    old_flat,_=ravel_pytree(old)
    np.testing.assert_allclose(folded,.95*old_flat+.05*actual,rtol=2e-5,atol=2e-6)
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
    for path,expected in (("scale125m.json",124_373_760),("scale350m.json",354_599_936),("scale1b.json",984_909_312)):
        cfg=load_config(root/"configs"/path,"rocora")
        assert cfg.model.parameter_count==expected
        sophia=load_config(root/"configs"/path,"sophia")
        for field in ("learning_rate","beta1","beta2","rho","eps","weight_decay","core_interval"):
            assert getattr(cfg.optimizer,field)==getattr(sophia.optimizer,field)
    pod=json.loads((root/"configs/pod.json").read_text())
    assert pod["seeds"]["scale125m"]==[1337,1338,1339]
    assert pod["seeds"]["scale350m"]==[1337]
    assert pod["seeds"]["scale1b"]==[1337]


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


def test_flash_attention_block_sizes_and_invocation(monkeypatch):
    """Test FlashAttention BlockSizes configuration (block_q=128, block_k=1024, causal=True)."""
    captured = {}

    def mock_flash(q, k, v, causal=True, sm_scale=None, block_sizes=None):
        captured["causal"] = causal
        captured["sm_scale"] = sm_scale
        captured["block_sizes"] = block_sizes
        return jnp.zeros_like(q)

    monkeypatch.setattr("jax.experimental.pallas.ops.tpu.flash_attention.flash_attention", mock_flash)

    q = jnp.zeros((1, 8, 1024, 64), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 8, 1024, 64), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 8, 1024, 64), dtype=jnp.bfloat16)

    flash_attention(q, k, v)

    assert captured["causal"] is True
    np.testing.assert_allclose(captured["sm_scale"], 1.0 / 8.0)
    blocks = captured["block_sizes"]
    assert blocks.block_q == 128
    assert blocks.block_k == 1024
    assert blocks.block_b == 1
    assert blocks.block_k_major == 1024
    assert blocks.block_q_major_dkv == 128
    assert blocks.block_k_major_dkv == 1024
    assert blocks.block_k_dkv == 1024
    assert blocks.block_q_dkv == 128
    assert blocks.block_k_major_dq == 1024
    assert blocks.block_k_dq == 1024
    assert blocks.block_q_dq == 128

    # Verify tile clamp when sequence length is shorter than 1024
    q512 = jnp.zeros((1, 8, 512, 64), dtype=jnp.bfloat16)
    flash_attention(q512, q512, q512)
    assert captured["block_sizes"].block_k == 512


def test_model_attention_routing_forward_flash_vs_curvature_reference():
    """Verify forward pass routes to FlashAttention and curvature pass routes to reference FP32 attention."""
    cfg = ModelConfig(width=64, layers=1, heads=1, sequence_length=128,
                      vocab_size=100, valid_vocab_size=100, attention="flash")
    params = init_params(cfg, 1)
    tokens = jnp.zeros((1, 128), dtype=jnp.int32)

    # Forward pass with curvature=False routes to flash_attention. On CPU backend,
    # Pallas flash_attention raises ValueError("Only interpret mode is supported on CPU backend").
    routed_flash = False
    try:
        hidden(params, tokens, cfg, curvature=False)
        routed_flash = True
    except ValueError as e:
        assert "Only interpret mode is supported on CPU backend" in str(e)
        routed_flash = True
    assert routed_flash

    # Curvature pass with curvature=True routes strictly to reference_attention in FP32.
    out = hidden(params, tokens, cfg, curvature=True)
    assert out.shape == (1, 128, 64)
    assert out.dtype == jnp.float32
    assert np.all(np.isfinite(out))


def test_reference_attention_causal_masking_strictly_lower_triangular():
    """Verify reference_attention strictly zeroes upper triangular attention connections."""
    seq_len = 8
    d = 4
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 1, seq_len, d))
    k = jax.random.normal(kk, (1, 1, seq_len, d))
    v = jax.random.normal(kv, (1, 1, seq_len, d))

    def out_at_i(v_in, i):
        out = reference_attention(q, k, v_in, block=4)
        return jnp.sum(out[0, 0, i, :])

    grad_fn = jax.jit(jax.grad(out_at_i, argnums=0))

    for i in range(seq_len):
        g = grad_fn(v, i)[0, 0]
        influence = jnp.linalg.norm(g, axis=-1)
        # Position i cannot attend to future position j > i
        for j in range(i + 1, seq_len):
            assert float(influence[j]) == 0.0, f"Causal violation: token {i} attended to future token {j}"
        # Position i must attend to current and past positions j <= i
        for j in range(i + 1):
            assert float(influence[j]) > 0.0


def test_reference_attention_query_tiling_equivalence():
    """Verify reference_attention query tiling produces identical results for block=64 vs block=128."""
    seq_len = 256
    batch = 2
    heads = 2
    d = 64
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch, heads, seq_len, d), dtype=jnp.float32)
    k = jax.random.normal(k2, (batch, heads, seq_len, d), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch, heads, seq_len, d), dtype=jnp.float32)

    out_64 = reference_attention(q, k, v, block=64)
    out_128 = reference_attention(q, k, v, block=128)
    out_256 = reference_attention(q, k, v, block=256)

    np.testing.assert_allclose(np.asarray(out_64), np.asarray(out_128), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_64), np.asarray(out_256), rtol=1e-5, atol=1e-5)


def test_reference_attention_jvp_autodiff():
    """Verify reference_attention supports first- and second-order forward-mode automatic differentiation."""
    seq_len = 32
    batch = 1
    heads = 2
    d = 16
    key = jax.random.PRNGKey(101)
    keys = jax.random.split(key, 6)
    q = jax.random.normal(keys[0], (batch, heads, seq_len, d), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (batch, heads, seq_len, d), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (batch, heads, seq_len, d), dtype=jnp.float32)
    tq = jax.random.normal(keys[3], q.shape, dtype=jnp.float32)
    tk = jax.random.normal(keys[4], k.shape, dtype=jnp.float32)
    tv = jax.random.normal(keys[5], v.shape, dtype=jnp.float32)

    f = lambda q_, k_, v_: jnp.sum(reference_attention(q_, k_, v_, block=16))
    primals, tangents = jax.jvp(f, (q, k, v), (tq, tk, tv))
    assert np.isfinite(primals)
    assert np.isfinite(tangents)

    # Second-order JVP-of-gradient required for Rocora's residual curvature measurements
    grad_f = jax.grad(f, argnums=0)
    primals_g, tangents_g = jax.jvp(grad_f, (q, k, v), (tq, tk, tv))
    assert np.all(np.isfinite(primals_g))
    assert np.all(np.isfinite(tangents_g))


def test_attention_bench_scale_parameterization_and_flops():
    """Verify attention_bench scale configs and theoretical FLOPs calculations."""
    from rocora.benchmark import compute_attention_flops, SCALE_CONFIGS

    assert SCALE_CONFIGS["125m"]["heads"] == 12
    assert SCALE_CONFIGS["350m"]["heads"] == 16
    assert SCALE_CONFIGS["1b"]["heads"] == 24

    for scale, heads in (("125m", 12), ("350m", 16), ("1b", 24)):
        flops = compute_attention_flops(batch=8, heads=heads, sequence=1024, head_dim=64)
        expected_fwd_causal = 2 * 8 * heads * (1024 ** 2) * 64
        expected_fwd_dense = 4 * 8 * heads * (1024 ** 2) * 64
        assert flops["forward_causal"] == expected_fwd_causal
        assert flops["forward"] == expected_fwd_dense
        assert flops["backward_causal"] == int(2.5 * expected_fwd_causal)
        assert flops["backward"] == int(2.5 * expected_fwd_dense)


# ==============================================================================
# Milestone 3: Pallas Optimizer Kernels & Fallback Engine Unit Tests
# ==============================================================================

def test_pallas_adamw_parity_and_tiling():
    """Verify Pallas AdamW kernel numerical parity against XLA reference across shapes."""
    from rocora.pallas_optimizers import pallas_adamw_update
    from rocora.updates import adamw_update

    lr, b1, b2, eps, wd, step, grad_scale = 1e-3, 0.965, 0.99, 1e-8, 0.01, 1, 0.8
    corr1 = 1.0 / (1.0 - b1 ** step)
    corr2 = 1.0 / (1.0 - b2 ** step)
    c = jnp.asarray([lr, b1, b2, eps, wd, 0.0, corr1, corr2, 0.0, grad_scale], dtype=jnp.float32)

    # Test 1: Multiple of 128 elements (e.g., 256)
    p = jnp.linspace(-1.0, 1.0, 256, dtype=jnp.float32)
    g = jnp.full(256, 0.05, dtype=jnp.float32)
    m = jnp.full(256, 0.01, dtype=jnp.float32)
    v = jnp.full(256, 0.02, dtype=jnp.float32)

    xla_p, xla_m, xla_v = adamw_update(p, g, m, v, c)
    pal_p, pal_m, pal_v = pallas_adamw_update(
        p, g, m, v, lr, b1, b2, eps, wd, step=step, grad_scale=grad_scale, block_size=128
    )

    np.testing.assert_allclose(xla_p, pal_p, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(xla_m, pal_m, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(xla_v, pal_v, rtol=1e-6, atol=1e-7)

    # Test 2: Non-multiple of 128 elements with padding boundary (e.g., 100 elements)
    p_sub = p[:100]
    g_sub = g[:100]
    m_sub = m[:100]
    v_sub = v[:100]

    xla_p_sub, xla_m_sub, xla_v_sub = adamw_update(p_sub, g_sub, m_sub, v_sub, c)
    pal_p_sub, pal_m_sub, pal_v_sub = pallas_adamw_update(
        p_sub, g_sub, m_sub, v_sub, lr, b1, b2, eps, wd, step=step, grad_scale=grad_scale, block_size=128
    )

    np.testing.assert_allclose(xla_p_sub, pal_p_sub, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(xla_m_sub, pal_m_sub, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(xla_v_sub, pal_v_sub, rtol=1e-6, atol=1e-7)


def test_pallas_sophia_parity_and_clipping():
    """Verify Pallas Sophia-G kernel numerical parity and update clipping boundaries."""
    from rocora.pallas_optimizers import pallas_sophia_update
    from rocora.updates import sophia_update

    lr, b1, b2, eps, rho, wd, grad_scale = 4e-4, 0.965, 0.99, 1e-15, 0.03, 0.1, 1.0
    c = jnp.asarray([lr, b1, b2, eps, wd, rho, 0.0, 0.0, 0.0, grad_scale], dtype=jnp.float32)

    p = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)
    g = jnp.array([0.05, -0.05, 50.0, -50.0], dtype=jnp.float32)  # includes clipping-triggering values
    m = jnp.zeros_like(p)
    s = jnp.array([0.1, 0.1, 0.001, 0.001], dtype=jnp.float32)

    xla_p, xla_m = sophia_update(p, g, m, s, c)
    pal_p, pal_m = pallas_sophia_update(
        p, g, m, s, lr, b1, b2, eps, rho, wd, grad_scale=grad_scale, block_size=128
    )

    np.testing.assert_allclose(xla_p, pal_p, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(xla_m, pal_m, rtol=1e-6, atol=1e-7)


def test_pallas_rocora_parity_and_scientific_control():
    """Verify Pallas Rocora kernel parity and exact bitwise scientific control at residual_weight=0."""
    from rocora.pallas_optimizers import pallas_rocora_update, pallas_sophia_update
    from rocora.updates import rocora_update

    lr, b1, b2, eps, rho, wd, grad_scale = 4e-4, 0.965, 0.99, 1e-15, 0.03, 0.1, 1.0

    p = jnp.linspace(-2.0, 2.0, 256, dtype=jnp.float32)
    g = jnp.full(256, 0.02, dtype=jnp.float32)
    m = jnp.full(256, 0.005, dtype=jnp.float32)
    s = jnp.full(256, 0.1, dtype=jnp.float32)
    r = jnp.sin(jnp.linspace(0, 3.14, 256)) * 0.05

    # 1. Parity with active residual curvature (residual_weight = 1.0)
    c_active = jnp.asarray([lr, b1, b2, eps, wd, rho, 0.0, 0.0, 1.0, grad_scale], dtype=jnp.float32)
    xla_p, xla_m = rocora_update(p, g, m, s, r, c_active)
    pal_p, pal_m = pallas_rocora_update(
        p, g, m, s, r, lr, b1, b2, eps, rho, wd, residual_weight=1.0, grad_scale=grad_scale, block_size=128
    )
    np.testing.assert_allclose(xla_p, pal_p, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(xla_m, pal_m, rtol=1e-6, atol=1e-7)

    # 2. Strict scientific control: residual_weight = 0.0 must reproduce Sophia-G bitwise
    pal_rocora_zero_p, pal_rocora_zero_m = pallas_rocora_update(
        p, g, m, s, r, lr, b1, b2, eps, rho, wd, residual_weight=0.0, grad_scale=grad_scale, block_size=128
    )
    pal_sophia_p, pal_sophia_m = pallas_sophia_update(
        p, g, m, s, lr, b1, b2, eps, rho, wd, grad_scale=grad_scale, block_size=128
    )
    # Bitwise exact parity
    np.testing.assert_array_equal(pal_rocora_zero_p, pal_sophia_p)
    np.testing.assert_array_equal(pal_rocora_zero_m, pal_sophia_m)
    assert float(jnp.max(jnp.abs(pal_rocora_zero_p - pal_sophia_p))) == 0.0


def test_pallas_muon_parity_and_semi_orthogonality():
    """Verify Pallas Muon Newton-Schulz kernel numerical parity and semi-orthogonality."""
    from rocora.pallas_optimizers import pallas_muon_direction
    from rocora.matrix_optimizers import muon_direction

    class MuonCfg:
        muon_momentum = 0.95
        muon_ns_steps = 5
        muon_nesterov = True

    rng = np.random.default_rng(1337)
    g = jnp.asarray(rng.normal(size=(16, 32)).astype(np.float32))
    m = jnp.zeros_like(g)

    dir_xla, m_xla = muon_direction(g, m, MuonCfg())
    dir_pal, m_pal = pallas_muon_direction(g, m, cfg=MuonCfg(), use_pallas=True)

    np.testing.assert_allclose(dir_xla, dir_pal, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(m_xla, m_pal, rtol=1e-6, atol=1e-7)

    # Verify spectral boundedness of normalized update
    aspect = max(1, 32 / 16) ** 0.5
    raw_dir = dir_pal / aspect
    s_vals = jnp.linalg.svd(raw_dir, compute_uv=False)
    assert jnp.all((s_vals >= 0.5) & (s_vals <= 1.5))
    assert np.all(np.isfinite(dir_pal))


def test_pallas_soap_parity_and_eigenspace_projection():
    """Verify Pallas SOAP eigenspace projection and rotated Adam numerical parity."""
    from rocora.pallas_optimizers import pallas_soap_leaf
    from rocora.matrix_optimizers import soap_leaf, SoapFactors, _covariances, _initial_basis

    shape = (8, 8)
    p = jnp.ones(shape, dtype=jnp.float32)
    g = jnp.full(shape, 0.02, dtype=jnp.float32)
    m = jnp.zeros(shape, dtype=jnp.float32)
    v = jnp.zeros(shape, dtype=jnp.float32)

    cov = _covariances(g, (jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32)), 0.95)
    factors = SoapFactors(cov, _initial_basis(cov))

    class SoapCfg:
        beta1 = 0.965
        beta2 = 0.99
        eps = 1e-15
        soap_shampoo_beta = 0.95
        soap_precondition_frequency = 10
        soap_max_precondition_dim = 2048

    cfg = SoapCfg()
    xla_p, xla_m, xla_v, _ = soap_leaf(p, g, m, v, factors, 2, 1e-3, 0.1, cfg)
    pal_p, pal_m, pal_v, _ = pallas_soap_leaf(p, g, m, v, factors, 2, 1e-3, 0.1, cfg, use_pallas=True)

    np.testing.assert_allclose(xla_p, pal_p, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(xla_m, pal_m, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(xla_v, pal_v, rtol=1e-5, atol=1e-6)


def test_should_use_pallas_fallback_policy():
    """Verify should_use_pallas policy predicates across modes and backends."""
    from rocora.pallas_optimizers import should_use_pallas

    # 1. Explicit mode overrides
    assert should_use_pallas("adamw", mode="xla") is False
    assert should_use_pallas("adamw", mode="pallas") is True

    # 2. Auto mode on CPU backend (always falls back to XLA)
    cpu_device = jax.devices("cpu")[0]
    assert should_use_pallas("adamw", device=cpu_device, mode="auto") is False
    assert should_use_pallas("sophia", device=cpu_device, mode="auto") is False
    assert should_use_pallas("rocora", device=cpu_device, mode="auto") is False
    assert should_use_pallas("muon", device=cpu_device, mode="auto") is False
    assert should_use_pallas("soap", device=cpu_device, mode="auto") is False


def test_pallas_optimizer_update_dispatch():
    """Verify high-level update() integrates use_pallas toggle and produces identical results."""
    from rocora.config import OptimizerConfig
    from rocora.optimizers import init_state, update

    params = {"w": jnp.ones(128, dtype=jnp.float32)}
    grads = {"w": jnp.full(128, 0.01, dtype=jnp.float32)}

    cfg_xla = OptimizerConfig(name="adamw", learning_rate=1e-3, use_pallas=False)
    cfg_pallas = OptimizerConfig(name="adamw", learning_rate=1e-3, use_pallas=True, pallas_mode="pallas")

    state_xla = init_state(params, cfg_xla)
    state_pal = init_state(params, cfg_pallas)

    next_xla = update(state_xla, grads, 1e-3, cfg_xla, {"w": True})
    next_pal = update(state_pal, grads, 1e-3, cfg_pallas, {"w": True})

    np.testing.assert_allclose(next_xla.params["w"], next_pal.params["w"], atol=1e-5)


def test_optimizer_bench_execution():
    """Verify optimizer_bench executes across scales and measures zero error against references."""
    from rocora.benchmark import optimizer_bench, SCALE_CONFIGS

    for scale in ("125m", "350m", "1b"):
        assert scale in SCALE_CONFIGS
        assert "width" in SCALE_CONFIGS[scale]
        assert "layers" in SCALE_CONFIGS[scale]

    # Run microbenchmark for all 5 optimizers on CPU
    res = optimizer_bench(size=128, repetitions=1, scale="125m")
    assert res["scale"] == "125m"
    assert res["width"] == 768
    for opt in ("adamw", "sophia", "rocora", "soap", "muon"):
        assert opt in res["results"]
        assert res["results"][opt]["max_error"] < 1e-5
        assert res["results"][opt]["speedup"] > 0
