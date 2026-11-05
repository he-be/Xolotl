#!/usr/bin/env python3
"""fetch.py - pick the MXFP4 set and range-fetch those experts from UD-Q4_K_XL.
  select: sel.json = {"A": [[l,e],...] (XQ v4's MXFP4 experts), "C1": [...] (A + rank-1017-v4 order up to 134.15 GiB)}
  fetch : chunks of consecutive experts per (layer, role) -> OUT/L{l:02d}_{role}_{e0:03d}-{e1:03d}.bin (resumable)."""
import json, os, struct, sys, time, urllib.request, concurrent.futures as cf
D = "/root/bench-ds41/iq1m/mx"; OUT = "/root/models/mx-hot"
REV = "85f81fb970add013033f25e94006713ca2c0aaf2"
BASE = f"https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/{REV}/"
MX1 = 5120 * 2304 * 17 // 32           # one expert, one role
G = 2**30
IQ = {"IQ1_S": 884736000/384, "IQ2_XXS": 1167851520/384, "IQ2_XS": 1309409280/384, "IQ2_S": 1450967040/384}
def iq(l):
    g, d = ("IQ1_S","IQ1_S") if l < 5 else ("IQ1_S","IQ2_XXS") if l < 20 else ("IQ2_XXS","IQ2_S") if l < 30 else \
           ("IQ2_XXS","IQ2_XS") if l < 38 else ("IQ2_XS","IQ2_XS")
    return 2*IQ[g] + IQ[d]
def select():
    xq = json.load(open("/root/models/V41-XQ-1017-v4/V41-XQ-1017-v4.gguf.experts.json"))["layers"]
    A = [(int(l), e) for l, v in xq.items() for e, t in enumerate(v["gu_type"]) if t == 39]
    r = open("/root/models/V41-XQ-1017-v4/rank-1017-v4.bin", "rb").read()
    p = struct.unpack("<%di" % (len(r)//4), r); rank = list(zip(p[0::2], p[1::2]))
    base = sum(iq(l)*384 for l in range(40)); add = sum(3*MX1 - iq(l) for l, e in A); C = list(A); s = set(A)
    for l, e in rank:
        if (l, e) in s: continue
        if base + add + 3*MX1 - iq(l) > 134.15*G: break
        add += 3*MX1 - iq(l); C.append((l, e))
    json.dump({"A": sorted(A), "C1": sorted(C)}, open(f"{D}/sel.json", "w"))
    print("A", len(A), "C1", len(C), "total GiB %.2f" % ((base+add)/G))
def chunks(sel, maxn=16):
    by = {}
    for l, e in sel: by.setdefault(l, []).append(e)
    for l, es in sorted(by.items()):
        es.sort(); run = [es[0]]
        for e in es[1:]:
            if e == run[-1] + 1 and len(run) < maxn: run.append(e)
            else: yield l, run[0], run[-1]; run = [e]
        yield l, run[0], run[-1]
def fetch1(job):
    T, l, e0, e1, role = job
    f, ty, shape, off, nb = T[f"blk.{l}.ffn_{role}_exps.weight"]
    assert ty == 39 and nb == 384*MX1
    path = f"{OUT}/L{l:02d}_{role}_{e0:03d}-{e1:03d}.bin"; want = (e1-e0+1)*MX1
    if os.path.exists(path) and os.path.getsize(path) == want: return 0
    a = off + e0*MX1; b = off + (e1+1)*MX1 - 1
    for t in range(20):
        try:
            req = urllib.request.Request(BASE + f, headers={"Range": f"bytes={a}-{b}", "User-Agent": "mx-fetch"})
            data = urllib.request.urlopen(req, timeout=300).read()
            if len(data) != want: raise IOError(f"short {len(data)} != {want}")
            open(path + ".tmp", "wb").write(data); os.replace(path + ".tmp", path); return want
        except Exception as ex:
            print("retry", path, t, ex, flush=True); time.sleep(15)
    raise SystemExit("failed " + path)
def fetch(which):
    os.makedirs(OUT, exist_ok=True)
    T = json.load(open(f"{D}/q4xl_tensors.json")); sel = [tuple(x) for x in json.load(open(f"{D}/sel.json"))[which]]
    jobs = [(T, l, e0, e1, r) for l, e0, e1 in chunks(sel) for r in ("gate", "up", "down")]
    tot = len(sel)*3*MX1; done = 0; t0 = time.time()
    print(f"{len(jobs)} ranges, {tot/1e9:.1f} GB", flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        for i, n in enumerate(ex.map(fetch1, jobs)):
            done += n
            if i % 50 == 0: print(f"{i}/{len(jobs)} {done/1e9:.1f} GB {done/1e6/max(1,time.time()-t0):.1f} MB/s", flush=True)
    print("DONE", time.ctime(), flush=True)
if __name__ == "__main__":
    select() if sys.argv[1] == "select" else fetch(sys.argv[2])
