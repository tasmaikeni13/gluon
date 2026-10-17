"""Budget accounting, safe selection, and production recipe coverage."""

import json
from pathlib import Path
import pytest

from rocora.config import OPTIMIZERS, load_config
from rocora.sweep import materialize, plan, select


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage,count,tokens,seeds",[
    ("scale125m",109_529_856,2_500_000_000,[1337,1338,1339,1340,1341]),
    ("scale350m",336_380_928,7_000_000_000,[1337]),
    ("scale1b",955_221_504,20_000_000_000,[1337]),
])
def test_all_requested_main_and_budget_recipes(stage,count,tokens,seeds):
    for arm in OPTIMIZERS:
        cfg=load_config(ROOT/f"configs/{stage}.json",arm)
        assert cfg.model.parameter_count==count and cfg.train_tokens==tokens
        assert list(cfg.seeds)==seeds and cfg.arms==OPTIMIZERS
        assert cfg.model.attention=="flash" and cfg.dataset=="HuggingFaceFW/fineweb-edu"
        if stage!="scale125m":
            assert cfg.minimum_train_corpus_tokens==tokens
    sweep=plan(ROOT/f"configs/sweeps/{stage}.json")
    assert sweep["trials_per_arm"]==6
    assert sweep["total_tokens"]==len(OPTIMIZERS)*sweep["tokens_per_arm"]
    shared={}
    for arm in OPTIMIZERS:
        trials=[t for t in sweep["trials"] if t["optimizer"]==arm]
        assert len(trials)==6
        assert len({t["identity"] for t in trials})==6
        assert all(t["config"]["train_tokens"]==sweep["trial_tokens"] for t in trials)
        shared[arm]=[t["config"]["optimizer"] for t in trials]
    for s,r in zip(shared["sophia"],shared["rocora"]):
        assert {k:v for k,v in s.items() if k!="name"}=={k:v for k,v in r.items() if k!="name"}
    for s,sh in zip(shared["sophia"],shared["sophia_h"]):
        assert {k:v for k,v in s.items() if k!="name"}=={k:v for k,v in sh.items() if k!="name"}


def _results(manifest,root):
    for trial in manifest["trials"]:
        # Deliberately give Sophia and Rocora different individually best trials.
        index=int(trial["id"].split("-t")[1])
        loss=(index-2)**2 if trial["optimizer"]=="sophia" else (index-4)**2
        rows=[{"type":"meta","optimizer":trial["optimizer"],"config":trial["config"],
               "code_sha256":"same-source","devices":16,"processes":4,"jax":"0.6.2",
               "device_kind":"TPU v4","data":{"revision":"fixed"},"reference":False},
              {"type":"eval","step":1,"tokens":manifest["trial_tokens"],"loss":loss,"train_seconds":1},
              {"type":"complete","budget_complete":True}]
        path=root/trial["run_name"]/"metrics.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")


def test_selection_preserves_shared_control_and_rejects_incomplete_or_changed_trials(tmp_path):
    manifest=materialize(ROOT/"configs/sweeps/scale350m.json",tmp_path/"configs")
    assert materialize(ROOT/"configs/sweeps/scale350m.json",tmp_path/"configs")==manifest
    _results(manifest,tmp_path/"outputs")
    selected=tmp_path/"selected.json"
    result=select(manifest,tmp_path/"outputs",selected)
    assert result["winners"]["sophia"]=="sophia-t02"
    assert result["winners"]["rocora"]=="rocora-t02"
    assert load_config(selected,"rocora").train_tokens==7_000_000_000
    assert load_config(selected,"sophia").optimizer.rho==load_config(selected,"rocora").optimizer.rho
    path=tmp_path/"outputs"/manifest["trials"][0]["run_name"]/"metrics.jsonl"
    original=path.read_text()
    path.write_text(original.replace('"budget_complete": true','"budget_complete": false'))
    with pytest.raises(ValueError,match="not completed"):
        select(manifest,tmp_path/"outputs",tmp_path/"other.json")
    path.write_text(original.replace('"reference": false','"reference": true'))
    with pytest.raises(ValueError,match="Reference/smoke"):
        select(manifest,tmp_path/"outputs",tmp_path/"other.json")
    path.write_text(original.replace('"same-source"','"changed-source"'))
    with pytest.raises(ValueError,match="differ in code"):
        select(manifest,tmp_path/"outputs",tmp_path/"other.json")


def test_sweep_refuses_to_replace_changed_trial(tmp_path):
    manifest=materialize(ROOT/"configs/sweeps/scale125m.json",tmp_path)
    p=Path(manifest["trials"][0]["config_path"])
    raw=json.loads(p.read_text())
    raw["train_tokens"]+=1
    p.write_text(json.dumps(raw))
    with pytest.raises(FileExistsError,match="Changed trial"):
        materialize(ROOT/"configs/sweeps/scale125m.json",tmp_path)


def test_five_arm_report_requires_all_arms_seeds_and_matched_controls(tmp_path):
    from rocora.report import summarize
    paths=[]
    for i,arm in enumerate(OPTIMIZERS):
        cfg=load_config(ROOT/"configs/scale350m.json",arm)
        rows=[{"type":"meta","optimizer":arm,"seed":1337,"config":cfg.to_dict(),
               "code_sha256":"test","devices":16,"processes":4,"jax":"0.6.2",
               "device_kind":"TPU v4","data":{"revision":"test"}},
              {"type":"eval","step":1,"tokens":cfg.train_tokens//2,"loss":4+i*.01,"train_seconds":1},
              {"type":"eval","step":2,"tokens":cfg.train_tokens,"loss":3+i*.01,"train_seconds":2},
              {"type":"complete","budget_complete":True}]
        path=tmp_path/f'{arm}.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        paths.append(path)
    assert set(summarize(paths)["arms"])==set(OPTIMIZERS)
    with pytest.raises(ValueError,match="arms differ"):
        summarize(paths[:-1])
    rocora_idx = OPTIMIZERS.index("rocora")
    rows=[json.loads(line) for line in paths[rocora_idx].read_text().splitlines()]
    rows[0]["config"]["optimizer"]["rho"]=.5
    paths[rocora_idx].write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError,match="confounded"):
        summarize(paths)
    assert summarize(paths,comparison="tuned")["comparison"]=="tuned"
    first=[json.loads(line) for line in paths[0].read_text().splitlines()]
    first[0]["config"]["seeds"]=[1337,1338]
    paths[0].write_text('\n'.join(json.dumps(r) for r in first)+'\n')
    with pytest.raises(ValueError,match="configured seed"):
        summarize(paths)


def test_large_recipe_rejects_a_short_corpus(tmp_path):
    from dataclasses import replace
    import numpy as np
    from rocora.data import validate_corpus
    np.save(tmp_path/"train.npy",np.zeros(2048,np.uint16))
    np.save(tmp_path/"valid.npy",np.zeros(2048,np.uint16))
    metadata={"dataset":"HuggingFaceFW/fineweb-edu","tokenizer":"llama",
              "train_tokens":2048,"valid_tokens":2048}
    (tmp_path/"metadata.json").write_text(json.dumps(metadata))
    cfg=load_config(ROOT/"configs/scale1b.json")
    cfg=replace(cfg,train_file=str(tmp_path/"train.npy"),validation_file=str(tmp_path/"valid.npy"),
                data_metadata=str(tmp_path/"metadata.json"))
    with pytest.raises(ValueError,match="requires at least 20000000000"):
        validate_corpus(cfg)


def test_validate_corpus_rejects_gpt2_and_accepts_llama(tmp_path):
    from dataclasses import replace
    import numpy as np
    from rocora.data import validate_corpus
    np.save(tmp_path/"train.npy",np.zeros(2048,np.uint16))
    np.save(tmp_path/"valid.npy",np.zeros(2048,np.uint16))
    cfg=load_config(ROOT/"configs/scale125m.json")
    cfg=replace(cfg,train_file=str(tmp_path/"train.npy"),validation_file=str(tmp_path/"valid.npy"),
                data_metadata=str(tmp_path/"metadata.json"),minimum_train_corpus_tokens=0)

    # GPT-2 metadata is rejected
    metadata_gpt2={"dataset":"HuggingFaceFW/fineweb-edu","tokenizer":"gpt2",
                   "train_tokens":2048,"valid_tokens":2048}
    (tmp_path/"metadata.json").write_text(json.dumps(metadata_gpt2))
    with pytest.raises(ValueError,match="GPT-2 tokenization is deprecated and rejected"):
        validate_corpus(cfg)

    # LLaMA metadata is accepted
    metadata_llama={"dataset":"HuggingFaceFW/fineweb-edu","tokenizer":"llama (NousResearch/Llama-2-7b-hf)",
                    "train_tokens":2048,"valid_tokens":2048}
    (tmp_path/"metadata.json").write_text(json.dumps(metadata_llama))
    val=validate_corpus(cfg)
    assert "llama" in val["tokenizer"]
