"""Equivalence check: llama.cpp CPU sampling vs backend (GPU) sampling, greedy, single request at a time (batch 1 on both
servers so batching cannot change the numerics). Same GGUF, same flags otherwise. Writes bs_check.json and exits 0
only if every greedy output (reasoning + answer) is identical.
Usage: python bs_check.py <cpu_port> <gpu_port>
"""
import json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
cpu_port, gpu_port = sys.argv[1], sys.argv[2]
PROMPTS = [
    "Write a haiku about a lighthouse. Use only lowercase letters and do not use any commas.",
    "List exactly three benefits of drinking water as markdown bullet points, and end your answer with the phrase 'Stay hydrated.'",
    "Explain what a hash table is in two short paragraphs. The word 'bucket' must appear at least twice.",
]


def ask(port, prompt):
    body = {"model": "lfm3", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 768, "seed": 1234}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.perf_counter() - t0
    m = resp["choices"][0]["message"]
    return {"reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "",
            "tokens": resp["usage"]["completion_tokens"], "seconds": dt}


rows, ok = [], True
for p in PROMPTS:
    a, b = ask(cpu_port, p), ask(gpu_port, p)
    ta, tb = a["reasoning"] + "\n<ANSWER>\n" + a["content"], b["reasoning"] + "\n<ANSWER>\n" + b["content"]
    same = ta == tb
    first_diff = None if same else next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    ok &= same
    rows.append({"prompt": p, "identical": same, "first_diff_char": first_diff, "cpu": a, "gpu": b})
    print(f"identical={same} first_diff_char={first_diff} cpu {a['tokens']} tok {a['seconds']:.1f}s | gpu {b['tokens']} tok {b['seconds']:.1f}s")
json.dump({"identical_all": ok, "rows": rows}, open(os.path.join(ROOT, "bs_check.json"), "w", encoding="utf-8"), indent=1)
print("ALL IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
