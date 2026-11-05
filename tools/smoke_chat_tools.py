"""Smoke test an LFM3 OpenAI endpoint: one chat prompt + one tool-call prompt, card sampling
(temperature 0.1, top_k 50, repetition_penalty 1.1, fixed seed). Saves raw request + raw response JSON.

Server side (vLLM) needs: --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice

Usage: python smoke_chat_tools.py --url http://127.0.0.1:8131 --name liquid_nvfp4 --out results\\smoke_liquid_nvfp4.json
"""
import argparse, json, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252

SAMPLING = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1, "seed": 1234}

CHAT = [{"role": "user", "content": "In three sentences, explain why the sky is blue, then give the boiling "
                                    "point of water at sea level in both Celsius and Fahrenheit."}]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}]
TOOL_MSGS = [{"role": "user", "content": "What's the weather like in Indianapolis right now? Use fahrenheit."}]


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    return resp, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="lfm3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    out = {"name": a.name, "url": a.url, "sampling": SAMPLING, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    chat_req = {"model": a.model, "messages": CHAT, "max_tokens": a.max_tokens, **SAMPLING}
    resp, dt = post(a.url, chat_req)
    out["chat"] = {"request": chat_req, "response": resp, "latency_s": dt}
    m = resp["choices"][0]["message"]
    print(f"[{a.name}] chat ({resp['usage']['completion_tokens']} tok, finish={resp['choices'][0]['finish_reason']}):")
    print("  reasoning:", (m.get("reasoning") or m.get("reasoning_content") or "")[:200].replace("\n", " "))
    print("  content  :", (m.get("content") or "")[:400].replace("\n", " "))

    tool_req = {"model": a.model, "messages": TOOL_MSGS, "tools": TOOLS, "tool_choice": "auto",
                "max_tokens": a.max_tokens, **SAMPLING}
    resp, dt = post(a.url, tool_req)
    out["tool_call"] = {"request": tool_req, "response": resp, "latency_s": dt}
    m = resp["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    print(f"[{a.name}] tool ({resp['usage']['completion_tokens']} tok, finish={resp['choices'][0]['finish_reason']}):"
          f" {len(calls)} tool call(s)")
    for c in calls:
        print("  call:", c["function"]["name"], c["function"]["arguments"])
    if not calls:
        print("  content:", (m.get("content") or "")[:400].replace("\n", " "))
    ok_args = False
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
            ok_args = ok_args or (c["function"]["name"] == "get_weather" and "indianapolis" in str(args.get("city", "")).lower()
                                  and args.get("unit") == "fahrenheit")
        except Exception:
            pass
    out["tool_call_ok"] = ok_args
    print(f"[{a.name}] tool_call_ok={ok_args}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
