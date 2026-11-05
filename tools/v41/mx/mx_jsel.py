#!/usr/bin/env python3
"""mx_jsel.py - pick MXFP4 experts by Japanese-weighted calibration routing (cal/*.routes, disjoint from eval corpora);
report coverage on the 6 eval corpora; add sets "J" (A's budget) and "J2" (v4-equivalent budget) to mx/sel.json."""
import json, numpy as np, sys
sys.path.insert(0, "/root/bench-ds41/iq1m")
K=6; NE=384; MX=3*5120*2304*17/32
IQ={"IQ1_S":884736000/384,"IQ2_XXS":1167851520/384,"IQ2_XS":1309409280/384,"IQ2_S":1450967040/384}
def iq(l):
    g,d=("IQ1_S","IQ1_S") if l<5 else ("IQ1_S","IQ2_XXS") if l<20 else ("IQ2_XXS","IQ2_S") if l<30 else ("IQ2_XXS","IQ2_XS") if l<38 else ("IQ2_XS","IQ2_XS")
    return 2*IQ[g]+IQ[d]
def load(p):
    b=open(p,"rb").read(); o=0; cnt=np.zeros((40,NE))
    while o<len(b):
        l,p0,n=np.frombuffer(b,np.int32,3,o); o+=12; ids=np.frombuffer(b,np.int32,n*K,o); o+=8*n*K; np.add.at(cnt[l],ids,1)
    return cnt
D="/root/bench-ds41/iq1m"
W={"ja_agent-0":.15,"ja_agent-1":.15,"ja_logic-0":.15,"jawiki-0":.2,"en_code-0":.2,"v6-0":.15}
u=sum(w*load(f"{D}/cal/{c}.routes")/load(f"{D}/cal/{c}.routes").sum() for c,w in W.items())
E6={c:load(f"{D}/{c}-base.routes") for c in ["ja-api","ja-novel","ja-fast","ja-exact","html-api","en-readme"]}
cost=np.array([[MX-iq(l)]*NE for l in range(40)]).ravel()
sel=json.load(open(f"{D}/mx/sel.json"))
def cov(s):
    m=np.zeros(40*NE,bool); m[s]=True
    return " ".join("%s %.0f%%"%(c,100*E6[c].ravel()[m].sum()/E6[c].sum()) for c in E6)
A=[l*NE+e for l,e in sel["A"]]; print("A          ", cov(A))
G=2**30
for name,bud in (("J",9.66*G),("J2",19.24*G),("Jh",4.83*G),("Jq",2.42*G)):
    idx=np.argsort(-(u.ravel()/cost)); k=np.searchsorted(np.cumsum(cost[idx]),bud); s=idx[:k]
    print("%-3s %5.2f GiB n=%d"%(name,bud/G,k), cov(s))
    sel[name]=sorted([[int(i//NE),int(i%NE)] for i in s])
    xq=set(map(tuple,sel["A"])); print("    overlap with A", len(xq & set(map(tuple,sel[name]))), "per-layer n", np.bincount(s//NE,minlength=40).tolist())
json.dump(sel,open(f"{D}/mx/sel.json","w"))
