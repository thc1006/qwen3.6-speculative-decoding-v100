#!/usr/bin/env python3
"""Generate the figures (PNG) from the raw JSON. Usage: python scripts/plots.py [data_dir]
Writes figures/{speedup,acceptance,per_category,amdahl}.png"""
import json, glob, math, os, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
os.makedirs("figures", exist_ok=True)
C = {"dense": "#1B6CA8", "moe": "#C0392B", "fd": "#27AE60", "mtp": "#8E44AD"}
NS = [1, 2, 3, 4, 6, 8, 11, 15]
prm = json.load(open("prompts/prompts40.json"))
CATS = {i: c for i, (c, _) in enumerate(prm)}

def load(sub, cfg):
    r = {}
    for f in glob.glob(os.path.join(DATA, sub, f"{cfg}_p*.json")):
        i = int(f.split("_p")[-1].split(".")[0])
        try:
            t = json.load(open(f)).get("timings", {})
            if t.get("predicted_n", 0) > 0: r[i] = t
        except Exception: pass
    return r
def gm(xs): return math.exp(sum(math.log(x) for x in xs)/len(xs))
def curve(sub, pfx, ns):
    b = load(sub, f"{pfx}baseline"); xs = []; sp = []; ac = []
    for n in ns:
        c = load(sub, f"{pfx}n{n}"); com = set(b) & set(c)
        if not com: continue
        xs.append(n); sp.append(gm([c[i]["predicted_per_second"]/b[i]["predicted_per_second"] for i in com]))
        dn = sum(c[i]["draft_n"] for i in com); da = sum(c[i]["draft_n_accepted"] for i in com)
        ac.append(100*da/dn)
    return xs, sp, ac, st.mean([b[i]["predicted_per_second"] for i in b])

# ---- Fig 1: speedup vs n-max ----
plt.figure(figsize=(8, 5))
xd, sd, ad, bd = curve("camp3", "dense_", NS)
xm, sm, am, bm = curve("camp3", "moe_", NS)
xf, sf, af, bf = curve("campfd2", "fd_", NS)
xt, stp, at, bt = curve("campmtp2", "mtp_", [1, 2, 3, 4, 6])
plt.plot(xd, sd, "-o", color=C["dense"], lw=2.2, label=f"Dense 27B (base {bd:.0f})")
plt.plot(xm, sm, "-s", color=C["moe"], lw=2.2, label=f"MoE 35B-A3B (base {bm:.0f})")
plt.plot(xf, sf, "--^", color=C["fd"], lw=1.8, label=f"fast-dense 4B (base {bf:.0f}, confounded)")
plt.plot(xt, stp, ":D", color=C["mtp"], lw=1.8, label="MTP, dense 27B")
plt.axhline(1.0, color="0.4", ls="--", lw=1)
plt.text(11.3, 1.02, "break-even", fontsize=9, color="0.3")
plt.xlabel("--spec-draft-n-max"); plt.ylabel("geomean speedup (paired)")
plt.title("DFlash speedup vs draft depth  (2×V100, non-thinking greedy)")
plt.xticks(NS); plt.ylim(0.4, 1.7); plt.grid(alpha=.3); plt.legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig("figures/speedup.png", dpi=150); plt.close()

# ---- Fig 2: acceptance overlay ----
plt.figure(figsize=(6.5, 4.5))
plt.plot(xd, ad, "-o", color=C["dense"], lw=2.2, label="dense")
plt.plot(xm, am, "-s", color=C["moe"], lw=2.2, label="MoE")
plt.xlabel("--spec-draft-n-max"); plt.ylabel("draft acceptance (%)")
plt.title("Acceptance is near-identical dense vs MoE\n(baselines differ 2.57×)")
plt.xticks(NS); plt.ylim(0, 90); plt.grid(alpha=.3); plt.legend(fontsize=10)
plt.tight_layout(); plt.savefig("figures/acceptance.png", dpi=150); plt.close()

# ---- Fig 3: per-category bars at n8 ----
def percat(sub, pfx, n):
    b = load(sub, f"{pfx}baseline"); c = load(sub, f"{pfx}n{n}"); by = {}
    for i in set(b) & set(c):
        by.setdefault(CATS[i], []).append(c[i]["predicted_per_second"]/b[i]["predicted_per_second"])
    return {k: gm(v) for k, v in by.items()}
order = ["code", "math", "factual", "howto", "analysis", "creative"]
pd = percat("camp3", "dense_", 8); pm = percat("camp3", "moe_", 8)
import numpy as np
x = np.arange(len(order)); w = 0.38
plt.figure(figsize=(7, 4.5))
plt.bar(x - w/2, [pd[k] for k in order], w, color=C["dense"], label="dense")
plt.bar(x + w/2, [pm[k] for k in order], w, color=C["moe"], label="MoE")
plt.axhline(1.0, color="0.4", ls="--", lw=1)
plt.xticks(x, order, rotation=30, ha="right"); plt.ylabel("geomean speedup"); plt.ylim(0, 1.7)
plt.title("Per-category speedup at n=8 (workload-dependent: code/math win, creative loses)", fontsize=11)
plt.grid(alpha=.3, axis="y"); plt.legend(fontsize=10)
plt.tight_layout(); plt.savefig("figures/per_category.png", dpi=150); plt.close()

# ---- Fig 4: Amdahl v vs baseline ----
def vfit(sub, pfx, ns):
    b = load(sub, f"{pfx}baseline"); xs = []; ys = []
    for n in ns:
        c = load(sub, f"{pfx}n{n}"); com = set(b) & set(c)
        if not com: continue
        spv = gm([c[i]["predicted_per_second"]/b[i]["predicted_per_second"] for i in com])
        dn = sum(c[i]["draft_n"] for i in com); da = sum(c[i]["draft_n_accepted"] for i in com)
        tau = da/(dn/n)+1; xs.append(n); ys.append(tau/spv)
    nn = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(a*a for a in xs); sxy = sum(a*b2 for a, b2 in zip(xs, ys))
    v = (nn*sxy-sx*sy)/(nn*sxx-sx*sx); return v, st.mean([b[i]["predicted_per_second"] for i in b])
pts = [("dense 27B", *vfit("camp3", "dense_", NS), C["dense"], "o"),
       ("MoE 35B-A3B", *vfit("camp3", "moe_", NS), C["moe"], "s"),
       ("dense 4B", *vfit("campfd2", "fd_", NS), C["fd"], "^")]
plt.figure(figsize=(7, 4.2))
bs = [p[2] for p in pts]; vs = [p[1] for p in pts]
_n = len(bs); _sx = sum(bs); _sy = sum(vs); _sxx = sum(a*a for a in bs); _sxy = sum(a*b for a, b in zip(bs, vs))
_sl = (_n*_sxy-_sx*_sy)/(_n*_sxx-_sx*_sx); _ic = (_sy-_sl*_sx)/_n
plt.plot([25, 160], [_ic+_sl*25, _ic+_sl*160], "--", color="0.6", lw=1, zorder=1, label="OLS fit (3 pts)")
for name, v, base, col, mk in pts:
    plt.scatter([base], [v], s=90, color=col, marker=mk, zorder=3)
    plt.annotate(f"{name}\nv={v:.3f}", (base, v), textcoords="offset points", xytext=(8, -16), fontsize=9)
plt.xlabel("baseline decode speed (tok/s)"); plt.ylabel("marginal cost v (baseline-tokens)")
plt.title("Amdahl: marginal cost v rises with baseline speed\n(v·T_base absolute cost is flat → denominator effect)")
plt.grid(alpha=.3); plt.xlim(20, 165)
plt.tight_layout(); plt.savefig("figures/amdahl.png", dpi=150); plt.close()
print("wrote figures/{speedup,acceptance,per_category,amdahl}.png")
