"""Verify token archive hashes, counts and vocabulary before a costly pod run."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .config import load_config
from .data import validate_corpus, TokenBatches


def inspect_corpus(cfg):
    metadata = validate_corpus(cfg)
    result = {"dataset":cfg.dataset,"files":{}}
    for name,path in (("train",cfg.train_file),("valid",cfg.validation_file)):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while chunk := stream.read(16*1024*1024):
                h.update(chunk)
        digest = h.hexdigest()
        expected = metadata.get(name+"_sha256")
        if expected and expected != digest:
            raise ValueError(f"{path} does not match its metadata hash")
        tokens = np.load(path,mmap_mode="r")
        for start in range(0,len(tokens),16_000_000):
            if int(tokens[start:start+16_000_000].max()) >= cfg.model.valid_vocab_size:
                raise ValueError(f"Out-of-vocabulary token in {path}")
        result["files"][name] = {"sha256":digest,"tokens":int(tokens.size),"metadata_hash_matched":bool(expected)}
    loader = TokenBatches(cfg.train_file,cfg.model.sequence_length,cfg.global_batch_size,
                         token_budget=cfg.train_tokens,skip_initial_document=cfg.train_skip_initial_document)
    result["skipped_train_prefix_tokens"] = loader.skipped_tokens
    result["usable_train_tokens"] = int(loader.tokens.size)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = inspect_corpus(load_config(args.config))
    print(json.dumps(result,sort_keys=True),flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
