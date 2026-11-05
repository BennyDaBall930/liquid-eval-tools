"""Decode the frozen eval set (data/eval.pt + data/eval_meta.json) into readable JSONL, one line per sequence: the prompt
text, every model-generated span (the scored tokens) and the template text between spans (tool-result turns), with
special tokens kept so the chat-template structure is visible. CPU only, needs the LFM3 tokenizer (local snapshot or Hub).
Usage: python dump_eval.py [--category tool] [--out ../data/eval_decoded.jsonl]
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lfm3q  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--eval", default=os.path.join(ROOT, "data", "eval.pt"))
ap.add_argument("--meta", default=os.path.join(ROOT, "data", "eval_meta.json"))
ap.add_argument("--category", default=None, help="tool | chat | reason | raw (default: all)")
ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval_decoded.jsonl"))
args = ap.parse_args()

tok = lfm3q.load_tokenizer()
seqs = torch.load(args.eval)
meta = json.load(open(args.meta, encoding="utf-8"))
assert len(seqs) == len(meta), (len(seqs), len(meta))


def dec(ids):
    return tok.decode(ids, skip_special_tokens=False)


n = 0
scored = 0
with open(args.out, "w", encoding="utf-8", newline="\n") as f:
    for i, (ids, m) in enumerate(zip(seqs, meta)):
        if args.category and m["category"] != args.category:
            continue
        ids = ids.tolist()
        assert len(ids) == m["n_tokens"]
        segs = []
        if m["category"] == "raw":
            a, b = m["score"]
            segs.append({"part": "prefix_unscored", "range": [0, a], "text": dec(ids[:a])})
            segs.append({"part": "text", "range": [a, b], "scored": True, "text": dec(ids[a:b])})
            scored += b - a
        else:
            p = m["n_prompt"]
            segs.append({"part": "prompt", "range": [0, p], "text": dec(ids[:p])})
            prev = p
            for a, b in m["gen_ranges"]:
                if a > prev:
                    segs.append({"part": "template_between_turns", "range": [prev, a], "text": dec(ids[prev:a])})
                segs.append({"part": "model", "range": [a, b], "scored": True, "text": dec(ids[a:b])})
                scored += b - a
                prev = b
            if prev < len(ids):
                segs.append({"part": "tail", "range": [prev, len(ids)], "text": dec(ids[prev:])})
        rec = {"eval_idx": i, **m, "segments": segs}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} sequences, {scored} scored tokens -> {args.out}")
