"""Independent numerical references, layer isolation, and exact checkpoint resume."""

from dataclasses import replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.checkpoint import load_checkpoint, save_checkpoint
from rocora.config import ExperimentConfig, OptimizerConfig
from rocora.matrix_optimizers import SoapFactors
from rocora.model import decay_mask
from rocora.optimizers import init_state, update


def _soap_reference(params, gradients, cfg, rate):
    """NumPy reference for official 2D SOAP, including QR and moment transport."""
    p = params.copy()
    m, v = np.zeros_like(p), np.zeros_like(p)
    cov = [np.zeros((n, n), np.float32) for n in p.shape]
    def project(x, q, back=False):
        return q[0] @ x @ q[1].T if back else q[0].T @ x @ q[1]
    q = None
    for step, g in enumerate(gradients):
        if q is None:
            cov = [(1-cfg.soap_shampoo_beta) * (g @ g.T),
                   (1-cfg.soap_shampoo_beta) * (g.T @ g)]
            q = [np.linalg.eigh(c)[1][:, ::-1] for c in cov]
            continue
        z = project(g, q)
        m = cfg.beta1*m + (1-cfg.beta1)*z
        v = cfg.beta2*v + (1-cfg.beta2)*z*z
        correction = np.sqrt(1-cfg.beta2**step)/(1-cfg.beta1**step)
        p = (p-rate*correction*project(m/(np.sqrt(v)+cfg.eps), q, True))*(1-rate*cfg.weight_decay)
        cov = [cfg.soap_shampoo_beta*cov[0]+(1-cfg.soap_shampoo_beta)*(g@g.T),
               cfg.soap_shampoo_beta*cov[1]+(1-cfg.soap_shampoo_beta)*(g.T@g)]
        parameter_moment = project(m, q, True)
        if step % cfg.soap_precondition_frequency == 0:
            new_q=[]
            for axis in range(2):
                order=np.argsort(-np.diag(q[axis].T@cov[axis]@q[axis]), kind="stable")
                v=np.take(v,order,axis=axis)
                new_q.append(np.linalg.qr(cov[axis]@q[axis][:,order])[0])
            q=new_q
        m=project(parameter_moment,q)
    return p


def test_soap_matches_independent_multistep_reference_with_basis_refresh():
    cfg=OptimizerConfig(name="soap",beta1=.8,beta2=.9,eps=1e-8,
                        soap_precondition_frequency=2,soap_shampoo_beta=.9)
    rng=np.random.default_rng(17)
    p=rng.normal(size=(3,4)).astype(np.float32)
    gradients=[rng.normal(size=p.shape).astype(np.float32) for _ in range(8)]
    state=init_state({"w":jnp.asarray(p)},cfg)
    advance=jax.jit(lambda s,g:update(s,{"w":g},.002,cfg,{"w":True}))
    state=advance(state,gradients[0])
    np.testing.assert_array_equal(state.params["w"],p)  # Bootstrap includes no decay.
    for g in gradients[1:]:
        state=advance(state,g)
    np.testing.assert_allclose(state.params["w"],_soap_reference(p,gradients,cfg,.002),rtol=2e-5,atol=2e-6)
    assert state.optimizer.core_events == state.optimizer.residual_events == 0


@pytest.mark.parametrize("arm",["soap","muon"])
def test_layer_stacks_equal_separate_matrix_updates_and_resume(tmp_path,arm):
    cfg=OptimizerConfig(name=arm,beta1=.9,beta2=.95,eps=1e-8,
                        soap_precondition_frequency=2,soap_max_precondition_dim=3)
    params={"layers":{"qkv":jnp.arange(24,dtype=jnp.float32).reshape(2,3,4)/50,
                      "norm1":jnp.ones((2,3))},"embedding":jnp.ones((5,3))}
    state=init_state(params,cfg)
    advance=jax.jit(lambda s,g:update(s,g,.003,cfg,decay_mask(params)))
    gradients=[]
    for i in range(5):
        gradients.append(jax.tree.map(lambda p:jax.random.normal(jax.random.PRNGKey(i+7),p.shape),params))
    for g in gradients[:3]:
        state=advance(state,g)
    experiment=ExperimentConfig(optimizer=cfg)
    path=tmp_path/"checkpoint.npz"
    save_checkpoint(path,state,experiment,2.5,"test")
    restored,_=load_checkpoint(path,init_state(params,cfg),experiment,"test")
    for g in gradients[3:]:
        state,restored=advance(state,g),advance(restored,g)
    for actual,expected in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(actual,expected)
    # Each layer is independently preconditioned, including unequal gradients.
    for layer in range(2):
        single={"layers":{"qkv":params["layers"]["qkv"][layer:layer+1],
                          "norm1":params["layers"]["norm1"][layer:layer+1]}}
        substate=init_state(single,cfg)
        for g in gradients:
            substate=update(substate,{"layers":{k:v[layer:layer+1] for k,v in g["layers"].items()}},
                            .003,cfg,decay_mask(single))
        np.testing.assert_allclose(state.params["layers"]["qkv"][layer],
                                   substate.params["layers"]["qkv"][0],rtol=2e-5,atol=2e-6)


def test_muon_hidden_matrix_orientation_and_auxiliary_adamw():
    cfg=OptimizerConfig(name="muon",learning_rate=.02,muon_aux_lr=.0006,
                        beta1=.9,beta2=.95,eps=1e-8)
    params={"layers":{"qkv":jnp.ones((1,2,8)),"norm1":jnp.ones((1,2))},
            "embedding":jnp.ones((3,2)),"position":jnp.ones((2,2))}
    gradients=jax.tree.map(jnp.ones_like,params)
    result=jax.jit(lambda s:update(s,gradients,.02,cfg,decay_mask(params)))(init_state(params,cfg))
    for name in ("embedding","position"):
        expected=np.ones(params[name].shape)*(1-.0006*cfg.weight_decay)-.0006
        np.testing.assert_allclose(result.params[name],expected,atol=1e-7)
    np.testing.assert_allclose(result.params["layers"]["norm1"],1-.0006,atol=1e-7)
    # A rank-one all-ones matrix remains rank one. A scalar polynomial independently
    # gives the finite-iteration singular value, including the original 2x scale
    # for torch's [output=8,input=2] orientation.
    x=np.float32(1)
    for _ in range(cfg.muon_ns_steps):
        x=3.4445*x-4.7750*x**3+2.0315*x**5
    measured=(1-.02*cfg.weight_decay-np.asarray(result.params["layers"]["qkv"]))/.02
    np.testing.assert_allclose(measured,x*2/4,atol=.06,rtol=.08)  # BF16 quintic error.
    assert np.all(np.isfinite(measured))


@pytest.mark.parametrize("shape",[(3,9),(9,3),(4,4)])
def test_muon_quintic_matches_singular_value_reference_across_momentum_steps(shape):
    from rocora.matrix_optimizers import muon_direction
    cfg=OptimizerConfig(name="muon")
    rng=np.random.default_rng(85)
    moment=np.zeros(shape,np.float32)
    actual_moment=jnp.zeros(shape)
    for _ in range(3):
        g=rng.normal(size=shape).astype(np.float32)
        moment=cfg.muon_momentum*moment+(1-cfg.muon_momentum)*g
        nesterov=(1-cfg.muon_momentum)*g+cfg.muon_momentum*moment
        # Orthogonal equivariance allows a completely separate SVD calculation:
        # evaluate the finite polynomial on singular values, not Gram matrices.
        u,s,vh=np.linalg.svd(nesterov.T,full_matrices=False)
        s=s/(np.linalg.norm(nesterov)+1e-7)
        for _ in range(cfg.muon_ns_steps):
            s=3.4445*s-4.7750*s**3+2.0315*s**5
        expected=((u*s)@vh).T*max(1,shape[1]/shape[0])**.5
        actual,actual_moment=muon_direction(jnp.asarray(g),actual_moment,cfg)
        precise,_=muon_direction(jnp.asarray(g),jnp.asarray(
            (moment-(1-cfg.muon_momentum)*g)/cfg.muon_momentum),cfg,dtype=jnp.float32)
        np.testing.assert_allclose(actual_moment,moment,rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(precise,expected,rtol=3e-5,atol=3e-6)
        # The unstable quintic intentionally runs in BF16. Measure its numerical
        # approximation against FP32 separately from formula correctness.
        assert np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected)<.12


def test_soap_skips_large_axes_and_preconditions_each_allowed_axis():
    cfg=OptimizerConfig(name="soap",soap_max_precondition_dim=3)
    state=init_state({"embedding":jnp.ones((9,3)),"layers":{"norm1":jnp.ones((2,3))}},cfg)
    f=state.optimizer.preconditioner
    assert isinstance(f["embedding"],SoapFactors)
    assert f["embedding"].covariance[0].shape==(0,0)
    assert f["embedding"].covariance[1].shape==(3,3)
    assert f["layers"]["norm1"].covariance[0].shape==(2,0,0)
    for arm in ("soap","muon"):
        with pytest.raises(ValueError):
            replace(cfg,name=arm,muon_ns_steps=0).validate()


def test_soap_matches_author_generated_reference_vectors():
    fixture=json.loads((Path(__file__).parent/"fixtures/author_optimizers.json").read_text())["soap"]
    cfg=OptimizerConfig(name="soap",beta1=.8,beta2=.9,eps=1e-8,
                        soap_precondition_frequency=2,soap_shampoo_beta=.9)
    state=init_state({"w":jnp.asarray(fixture["initial_parameters"])},cfg)
    advance=jax.jit(lambda s,g:update(s,{"w":g},.002,cfg,{"w":True}))
    for event in fixture["events"]:
        state=advance(state,jnp.asarray(event["gradient"]))
        np.testing.assert_allclose(state.params["w"],event["parameters"],rtol=2e-6,atol=1e-6)


@pytest.mark.parametrize("index",[0,1,2])
def test_muon_matches_author_generated_bf16_reference_vectors(index):
    from rocora.matrix_optimizers import muon_direction
    fixture=json.loads((Path(__file__).parent/"fixtures/author_optimizers.json").read_text())["muon"][index]
    cfg=OptimizerConfig(name="muon")
    advance=jax.jit(lambda g,m:muon_direction(g,m,cfg))
    moment=jnp.zeros(tuple(fixture["shape"]))
    for event in fixture["events"]:
        direction,moment=advance(jnp.asarray(event["gradient"]),moment)
        np.testing.assert_allclose(moment,event["momentum"],rtol=2e-6,atol=1e-7)
        np.testing.assert_allclose(direction,event["direction"],rtol=1e-5,atol=1e-6)


def test_masked_microbatch_accumulation_matches_direct_gradient():
    from rocora.config import ModelConfig
    from rocora.model import init_params, loss_sum
    from rocora.train import build_steps
    model=ModelConfig(width=4,layers=1,heads=1,sequence_length=2,vocab_size=8,
                      valid_vocab_size=8,attention="reference",loss_chunk=2)
    optimizer=OptimizerConfig(name="adamw",beta1=.9,beta2=.95,eps=1e-8)
    cfg=ExperimentConfig(model=model,optimizer=optimizer,global_batch_size=2,
                         train_tokens=100,warmup_tokens=0,microbatch_per_device=1)
    params=init_params(model)
    batch={"tokens":jnp.asarray([[0,1],[2,3]]),"targets":jnp.asarray([[1,2],[3,4]]),
           "mask":jnp.asarray([[1.,1.],[1.,0.]])}
    from rocora.config import learning_rate
    from rocora.optimizers import gradient_norm_and_scale
    grad=jax.grad(lambda p:loss_sum(p,batch,model)/3)(params)
    _,scale=gradient_norm_and_scale(grad,optimizer.grad_clip)
    expected=update(init_state(params,optimizer),grad,learning_rate(0,cfg),optimizer,
                    decay_mask(params),gradient_scale=scale)
    devices=jax.local_devices()[:1]
    state=jax.device_put_replicated(init_state(params,optimizer),devices)
    mapped={k:v[None] for k,v in batch.items()}
    actual,metrics=build_steps(cfg,params)[0](state,mapped)
    assert int(metrics["tokens"][0])==3
    for a,b in zip(jax.tree.leaves(actual.params),jax.tree.leaves(expected.params)):
        np.testing.assert_allclose(a[0],b,rtol=2e-3,atol=2e-5)
