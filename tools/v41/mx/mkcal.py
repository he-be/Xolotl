#!/usr/bin/env python3
"""mkcal.py - calibration token streams for picking MXFP4 experts (Japanese-weighted, disjoint from rt/ eval corpora).
Conversations (from the Swift imatrix set, docs/pc/swift-quant/04-imatrix.md) rendered with DeepSeek-V4.1's encoder,
reasoning kept for every assistant turn (rendered turn by turn).  Writes cal/<part>-<k>.i32 of N tokens each."""
import json, random, sys
sys.path[:0] = ["/root/xolotl-1103/tools/v41/serve", "/root/xolotl-1103/tools/v41", "/root/xolotl-1103/tools", "/root/xolotl-1103"]
import serve_ds41 as SD
import strata_tokenizer as ST
import numpy as np
D = "/root/bench-ds41/iq1m/cal"; N = 16384
d = "/root/models/V41-XQ-1017-v4/tokenizer/"
vocab = json.load(open(d + "vocab.json")); toks = [None] * len(vocab)
for t, i in vocab.items(): toks[i] = t
merges = [l.rstrip("\n") for l in open(d + "merges.txt", encoding="utf-8") if l.strip() and not l.startswith("#version")]
T = ST.Tokenizer(toks, merges, json.load(open(d + "token_type.json")))
tpl = SD.Ds41Template()
BOS = T.encode("<｜begin▁of▁sentence｜>", parse_special=True)
def fix(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": c["function"]["arguments"] if isinstance(c["function"]["arguments"], str) else json.dumps(c["function"]["arguments"], ensure_ascii=False)}} for c in m["tool_calls"]]
        out.append(m)
    return out
def conv_ids(c):
    msgs = fix(c["messages"]); tools = c.get("tools")
    # render up to each assistant turn so every turn keeps its reasoning; take the new tail each time
    ids, prev = [], ""
    ends = [i for i, m in enumerate(msgs) if m["role"] == "assistant"] or [len(msgs) - 1]
    for i in ends:
        s = tpl.render(msgs[:i + 1], tools=tools, add_generation_prompt=False, reasoning_effort=c.get("reasoning_effort"))
        if not s.startswith(prev): ids += T.encode(s, parse_special=True); prev = s; continue
        ids += T.encode(s[len(prev):], parse_special=True); prev = s
    return ids[1:] if ids[:1] == BOS else ids
def stream(name, items, k):
    buf = []
    for x in items:
        buf += x
        while len(buf) >= N:
            np.array(BOS + buf[:N - 1], dtype=np.int32).tofile(f"{D}/{name}-{k}.i32"); print(name, k, flush=True); k += 1; buf = buf[N - 1:]
            return k
    return k
def chunks(name, gen, n):
    buf = []; k = 0
    for x in gen:
        buf += x
        if len(buf) >= N - 1:
            np.array(BOS + buf[:N - 1], dtype=np.int32).tofile(f"{D}/{name}-{k}.i32"); print(name, k, flush=True); k += 1; buf = []
            if k == n: return
    sys.exit(f"{name}: ran out at {k}")
def jl(p): return [json.loads(l) for l in open(f"{D}/src/{p}", encoding="utf-8")]
R = random.Random(20261105)
plan = {"ja_agent": 2, "ja_logic": 1, "jawiki": 1, "en_code": 1, "v6": 1}   # x 16k tokens
for kind in ("ja_agent", "ja_logic", "en_code"):
    cs = jl(f"{kind}.jsonl"); R.shuffle(cs)
    chunks(kind, (conv_ids(c) for c in cs), plan[kind])
pool = open(f"{D}/src/jawiki_pool.txt", encoding="utf-8").read().split("\n\n"); R.shuffle(pool)
chunks("jawiki", (T.encode(a + "\n\n") for a in pool), plan["jawiki"])
v6 = json.load(open(f"{D}/src/v6_conversations.json", encoding="utf-8")); cv = v6["conversations"] + v6["extension_conversations"]; R.shuffle(cv)
prose = open(f"{D}/src/v6_prose.txt", encoding="utf-8").read()
def v6gen():
    for c in cv[:60]: yield conv_ids(c if isinstance(c, dict) and "messages" in c else {"messages": c})
    yield T.encode(prose)
chunks("v6", v6gen(), plan["v6"])
