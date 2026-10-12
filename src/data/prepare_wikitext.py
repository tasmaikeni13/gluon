"""Download WikiText-103 (raw) and tokenize with GPT-2 BPE into uint16 memmap bins.

Output: data_bin/train.bin, data_bin/val.bin, data_bin/meta.json
Layout: flat token stream; documents separated by the GPT-2 EOT token (50256).
WikiText's raw splits are line-based; blank-line-separated sections are kept as one
stream (standard LM setup), with EOT only between the original split boundaries not
needed — we insert EOT between top-level articles (lines starting with " = " that are
titles) is overkill; the standard recipe concatenates the whole split as one stream.
"""

import json
import os
import time

import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_bin")
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256


def tokenize_split(ds_split, name: str) -> None:
    t0 = time.time()
    # Concatenate raw lines into large chunks, encode chunk-wise in parallel batches.
    lines = ds_split["text"]
    print(f"[{name}] {len(lines)} lines, joining...", flush=True)
    text = "".join(lines)  # wikitext raw lines already contain their newlines
    print(f"[{name}] {len(text) / 1e6:.1f} MB text, encoding...", flush=True)
    chunk = 8_000_000  # chars per encode batch
    pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)]
    del text
    tok_lists = enc.encode_ordinary_batch(pieces, num_threads=32)
    n_tok = sum(len(t) for t in tok_lists) + 1
    arr = np.empty(n_tok, dtype=np.uint16)
    pos = 0
    for t in tok_lists:
        arr[pos : pos + len(t)] = t
        pos += len(t)
    arr[pos] = EOT
    pos += 1
    assert pos == n_tok
    out = os.path.join(OUT_DIR, f"{name}.bin")
    arr.tofile(out)
    print(f"[{name}] {n_tok} tokens -> {out} ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    print("loading wikitext-103-raw-v1 ...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenize_split(ds["train"], "train")
    tokenize_split(ds["validation"], "val")
    tokenize_split(ds["test"], "test")
    meta = {
        "dataset": "wikitext-103-raw-v1",
        "tokenizer": "gpt2 (tiktoken)",
        "vocab_size": enc.n_vocab,
        "eot": EOT,
        "dtype": "uint16",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DATA_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
