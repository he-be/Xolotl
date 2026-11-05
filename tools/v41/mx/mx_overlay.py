#!/usr/bin/env python3
"""mx_overlay.py SET OUT [--layers A-B] - write the expert overlay GGUF (STRATA_GGUF_OVERLAY) for UD-IQ1_M:
per layer with chosen experts, I32 maps blk.L.ffn_exps_{gu,down}_type [384] (39 = MXFP4 for the chosen, the IQ1_M
layer's own types otherwise) and blk.L.ffn_{gate,up,down}_exps.mxfp4.weight holding the chosen experts in expert
order (bytes from mx_fetch.py's chunk files = UD-Q4_K_XL = the checkpoint's FP4).  alignment 4096."""
import json, os, struct, sys, glob
D = "/root/bench-ds41/iq1m/mx"; SRC = "/root/models/mx-hot"
BASE = "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf"
MX1 = 5120 * 2304 * 17 // 32; NE = 384; AL = 4096
T = {"IQ1_S": 19, "IQ2_XXS": 16, "IQ2_XS": 17, "IQ2_S": 22}
def iqt(l):
    g, d = ("IQ1_S","IQ1_S") if l < 5 else ("IQ1_S","IQ2_XXS") if l < 20 else ("IQ2_XXS","IQ2_S") if l < 30 else \
           ("IQ2_XXS","IQ2_XS") if l < 38 else ("IQ2_XS","IQ2_XS")
    return T[g], T[d]
which, out = sys.argv[1], sys.argv[2]
lo, hi = 0, 39
if "--layers" in sys.argv: lo, hi = map(int, sys.argv[sys.argv.index("--layers") + 1].split("-"))
sel = {}
for l, e in json.load(open(f"{D}/sel.json"))[which]:
    if lo <= l <= hi: sel.setdefault(l, []).append(e)
# chunk files: (layer, role) -> {e: (path, index)}
chunk = {}
for p in glob.glob(f"{SRC}/L*_*_*-*.bin"):
    b = os.path.basename(p)[:-4]; L, role, rng = b.split("_"); e0, e1 = map(int, rng.split("-"))
    if os.path.getsize(p) != (e1 - e0 + 1) * MX1: continue
    for e in range(e0, e1 + 1): chunk.setdefault((int(L[1:]), role), {})[e] = (p, e - e0)
# XQ v4 keeps its MXFP4 experts as the same bytes (checked byte-equal on layer 0): the local source when not fetched
XQF = "/root/models/V41-XQ-1017-v4/V41-XQ-1017-v4.gguf"
xq = json.load(open(XQF + ".experts.json"))["layers"]
for l, v in xq.items():
    for e, t in enumerate(v["gu_type"]):
        if t == 39:
            for role in ("gate", "up", "down"):
                chunk.setdefault((int(l), role), {}).setdefault(e, (XQF, ("abs", v["off"][str(e)][role])))
tensors = []   # (name, shape, type, nbytes, writer)
for l in sorted(sel):
    es = sorted(sel[l]); s = set(es); gt, dt = iqt(l)
    for role in ("gate", "up", "down"):
        miss = [e for e in es if e not in chunk.get((l, role), {})]
        if miss: sys.exit(f"layer {l} {role}: {len(miss)} experts not fetched yet")
    gu = struct.pack("<384i", *[39 if e in s else gt for e in range(NE)])
    dn = struct.pack("<384i", *[39 if e in s else dt for e in range(NE)])
    tensors.append((f"blk.{l}.ffn_exps_gu_type", [NE], 26, len(gu), gu))
    tensors.append((f"blk.{l}.ffn_exps_down_type", [NE], 26, len(dn), dn))
    for role in ("gate", "up", "down"):
        shape = [2304, 5120, len(es)] if role == "down" else [5120, 2304, len(es)]
        tensors.append((f"blk.{l}.ffn_{role}_exps.mxfp4.weight", shape, 39, len(es) * MX1, [chunk[(l, role)][e] for e in es]))
def gstr(s): b = s.encode(); return struct.pack("<Q", len(b)) + b
kv = [("general.alignment", 4, struct.pack("<I", AL)), ("overlay.base", 8, gstr(BASE)),
      ("overlay.source", 8, gstr(f"unsloth/DeepSeek-V4.1-Flash-1017-GGUF@85f81fb UD-Q4_K_XL experts, set {which}"))]
hdr = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kv)))
for k, t, v in kv: hdr += gstr(k) + struct.pack("<I", t) + v
off = 0; offs = []
for name, shape, ty, nb, _ in tensors:
    offs.append(off); hdr += gstr(name) + struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape) + struct.pack("<IQ", ty, off)
    off = (off + nb + AL - 1) // AL * AL
data0 = (len(hdr) + AL - 1) // AL * AL
tmp = out + ".tmp"
with open(tmp, "wb") as f:
    f.write(hdr); f.write(b"\0" * (data0 - len(hdr)))
    for (name, shape, ty, nb, w), o in zip(tensors, offs):
        assert f.tell() == data0 + o
        if isinstance(w, bytes): f.write(w)
        else:
            cache = {}
            for p, i in w:
                if p not in cache: cache = {p: open(p, "rb")}
                fh = cache[p]; fh.seek(i[1] if isinstance(i, tuple) else i * MX1); buf = fh.read(MX1)
                assert len(buf) == MX1; f.write(buf)
        pad = (-(f.tell() - data0)) % AL
        if pad: f.write(b"\0" * pad)
os.replace(tmp, out)
n = sum(len(v) for v in sel.values())
print(f"{out}: layers {len(sel)} experts {n} tensors {len(tensors)} size {os.path.getsize(out)/2**30:.2f} GiB")
