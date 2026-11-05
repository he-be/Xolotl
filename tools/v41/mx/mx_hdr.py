#!/usr/bin/env python3
"""hdr.py - read GGUF headers of unsloth UD-Q4_K_XL (rev 85f81fb) by HTTP range; write q4xl_tensors.json
{name: [shard_file, type_id, shape, abs_offset, nbytes]} for the routed-expert tensors."""
import json, struct, sys, urllib.request
REV = "85f81fb970add013033f25e94006713ca2c0aaf2"
REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "gguf-hdr"})
    return urllib.request.urlopen(req, timeout=120).read()
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REV}/UD-Q4_K_XL"))
files = sorted(x["path"] for x in tree if x["path"].endswith(".gguf"))
print(files, file=sys.stderr)
TS = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:8,11:8,12:8}  # scalar sizes (u8,i8,u16,i16,u32,i32,f32,bool->1 fixed below)
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
# ggml block (elems, bytes) for the types we may meet
BLK = {0:(1,4),1:(1,2),8:(32,34),12:(256,144),14:(256,210),39:(32,17),30:(1,2),26:(1,4)}
out = {}
for f in files:
    url = BASE + f
    n = 8 << 20
    buf = get(url, 0, n - 1)
    while True:
        try:
            p = 0
            def rd(fmt):
                global p
                v = struct.unpack_from(fmt, buf, p); p += struct.calcsize(fmt); return v
            def rs():
                (l,) = rd("<Q"); global p; s = buf[p:p+l].decode("utf-8", "replace"); p += l; return s
            def skipval(t):
                global p
                if t == 8: rs()
                elif t == 9:
                    (at, cnt) = rd("<IQ")
                    if at == 8:
                        for _ in range(cnt): rs()
                    else: p += SZ[at] * cnt
                else: p += SZ[t]
            magic, ver, nt, nkv = rd("<4sIQQ")
            align = 32
            for _ in range(nkv):
                k = rs(); (t,) = rd("<I")
                if k == "general.alignment": (align,) = rd("<I")
                else: skipval(t)
            infos = []
            for _ in range(nt):
                name = rs(); (nd,) = rd("<I"); shape = list(rd("<%dQ" % nd)); (ty, off) = rd("<IQ")
                infos.append((name, shape, ty, off))
            if p > len(buf): raise IndexError
            break
        except (struct.error, IndexError):
            n *= 2; buf = get(url, 0, n - 1)
    data0 = (p + align - 1) // align * align
    for name, shape, ty, off in infos:
        if "_exps" in name:
            ne = 1
            for s in shape: ne *= s
            e, b = BLK.get(ty, (None, None))
            out[name] = [f, ty, shape, data0 + off, ne // e * b if e else None]
    print(f, "tensors", len(infos), "data0", data0, file=sys.stderr)
json.dump(out, open(sys.argv[1], "w"), indent=0)
from collections import Counter
print(Counter(v[1] for v in out.values()), len(out))
