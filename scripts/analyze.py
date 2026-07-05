#!/usr/bin/env python3
"""
Reproduce every headline number in this repo from the raw per-request JSON.
No third-party deps (pure stdlib). Usage:  python scripts/analyze.py [data_dir]
Writes results/summary_tables.md and prints the same to stdout.

Metric definitions (see METHODOLOGY.md):
  speedup(prompt)   = dflash.predicted_per_second / baseline.predicted_per_second   (paired, same prompt index)
  geomean speedup   = exp(mean(log(per-prompt speedups)))
  %net-loss         = fraction of prompts with speedup < 1.0
  acceptance        = sum(draft_n_accepted) / sum(draft_n)   (pooled over prompts)
  tau (pooled)      = sum(draft_n_accepted)/(sum(draft_n)/n_max) + 1   (mean accepted tokens per verify step)
  C                 = tau / geomean_speedup  == T_step / T_base   (per-step cost in baseline-token units)
  cost fit          = ordinary least squares  C(n) = d + v*n  over the swept n_max
  95% CI            = 10000-sample bootstrap on the log per-prompt speedups
Fixed budget: every request uses ignore_eos + n_predict=384, so predicted_n==384 everywhere (checked).
"""
import json, glob, math, os, sys, random, statistics as st
random.seed(12345)
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
OUT  = []
def emit(s=""):
    print(s); OUT.append(s)

CATS = None
def cats():
    global CATS
    if CATS is None:
        p = os.path.join(os.path.dirname(DATA), "..", "prompts", "prompts40.json")
        if not os.path.exists(p): p = "prompts/prompts40.json"
        CATS = {i: c for i, (c, _) in enumerate(json.load(open(p)))}
    return CATS

def load(sub, cfg):
    r = {}
    for f in glob.glob(os.path.join(DATA, sub, f"{cfg}_p*.json")):
        i = int(f.split("_p")[-1].split(".")[0])
        try:
            t = json.load(open(f)).get("timings", {})
            if t.get("predicted_n", 0) > 0:
                r[i] = t
        except Exception:
            pass
    return r

def gm(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs))
def bootci(rs, B=10000):
    L = [math.log(x) for x in rs]; n = len(L); m = []
    for _ in range(B):
        m.append(math.exp(sum(L[random.randrange(n)] for _ in range(n)) / n))
    m.sort(); return m[int(.025 * B)], m[int(.975 * B)]
def linfit(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    v = (n*sxy - sx*sy) / (n*sxx - sx*sx); d = (sy - v*sx) / n
    yb = sy/n; ssr = sum((y-(d+v*x))**2 for x, y in zip(xs, ys)); sst = sum((y-yb)**2 for y in ys)
    return d, v, 1 - ssr/sst

def sweep(sub, prefix, ns, label, bl_cfg="baseline"):
    b = load(sub, f"{prefix}{bl_cfg}")
    if not b: emit(f"### {label}: (no data)"); return None
    bt = st.mean([b[i]["predicted_per_second"] for i in b])
    bad = sum(1 for i in b if b[i].get("predicted_n") != 384)
    emit(f"### {label}  — baseline {bt:.2f} tok/s (n={len(b)} prompts, predicted_n!=384: {bad})")
    emit(f"| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |")
    emit(f"|---|---|---|---|---|---|---|")
    Cs = []
    for n in ns:
        c = load(sub, f"{prefix}n{n}")
        com = sorted(set(b) & set(c))
        if not com: continue
        sp = [c[i]["predicted_per_second"]/b[i]["predicted_per_second"] for i in com]
        e2e = [c[i]["predicted_n"]/((c[i]["prompt_ms"]+c[i]["predicted_ms"])/1000) /
               (b[i]["predicted_n"]/((b[i]["prompt_ms"]+b[i]["predicted_ms"])/1000)) for i in com]
        dn = sum(c[i]["draft_n"] for i in com); da = sum(c[i]["draft_n_accepted"] for i in com)
        lo, hi = bootci(sp); nl = 100*sum(1 for s in sp if s < 1)/len(sp)
        emit(f"| {n} | {gm(sp):.3f} | {st.median(sp):.3f} | {nl:.0f}% | {da/dn:.1%} | {gm(e2e):.3f} | [{lo:.3f}, {hi:.3f}] |")
        tau = da/(dn/n) + 1; Cs.append((n, tau/gm(sp)))
    if len(Cs) >= 2:
        d, v, r2 = linfit([n for n, _ in Cs], [cc for _, cc in Cs])
        emit(f"\ncost fit C(n)=d+v·n :  d={d:.3f}  v={v:.4f}  R²={r2:.3f}  (v = marginal per-step cost in baseline-token units)")
    return bt, Cs

def percat(sub, prefix, n, label):
    b = load(sub, f"{prefix}baseline"); c = load(sub, f"{prefix}n{n}")
    if not b or not c: return
    by = {}
    for i in set(b) & set(c):
        by.setdefault(cats()[i], []).append(c[i]["predicted_per_second"]/b[i]["predicted_per_second"])
    emit(f"\n{label} n{n} per-category geomean speedup:")
    for cat in sorted(by):
        emit(f"  - {cat:10s} {gm(by[cat]):.3f}× (n={len(by[cat])})")

def losslessness(sub, prefix, ns, label):
    def txt(sub, cfg, i):
        m = json.load(open(os.path.join(DATA, sub, f"{cfg}_p{i}.json")))["choices"][0]["message"]
        return (m.get("content") or "") + (m.get("reasoning_content") or "")
    emit(f"\n{label} — DFlash-greedy vs baseline-greedy identical-output count (of 40):")
    for n in ns:
        same = tot = 0
        for i in range(40):
            try:
                if txt(sub, "baseline" if prefix == "" else f"{prefix}baseline", i) == txt(sub, f"{prefix}n{n}", i):
                    same += 1
                tot += 1
            except Exception:
                pass
        emit(f"  - n{n}: {same}/{tot} identical")

NS_FULL = [1, 2, 3, 4, 6, 8, 11, 15]
emit("# Reproduced summary tables\n")
emit("_(regenerate with `python scripts/analyze.py`; every number below comes from the raw JSON in `data/raw/`)_\n")
emit("## 1. Main sweep — dense vs MoE, matched IQ4_XS target + Q8_0 drafter, non-thinking greedy\n")
sweep("camp3", "dense_", NS_FULL, "DENSE Qwen3.6-27B-IQ4_XS")
emit(""); percat("camp3", "dense_", 8, "DENSE")
emit("\n---\n")
sweep("camp3", "moe_", NS_FULL, "MoE Qwen3.6-35B-A3B-IQ4_XS")
emit(""); percat("camp3", "moe_", 8, "MoE")
emit("\n## 2. Fast-dense CONTROL (Qwen3-4B, single-GPU, self-converted drafter) — CONFOUNDED, see caveats\n")
sweep("campfd2", "fd_", NS_FULL, "fast-dense Qwen3-4B")
emit("\n## 3. Acceptance-equality (dense vs MoE, the key de-confounder)\n")
bd = load("camp3", "dense_baseline"); bm = load("camp3", "moe_baseline")
emit("| n-max | dense acc | MoE acc | Δ (pp) |\n|---|---|---|---|")
for n in NS_FULL:
    cd = load("camp3", f"dense_n{n}"); cm = load("camp3", f"moe_n{n}")
    ad = sum(cd[i]["draft_n_accepted"] for i in cd)/max(sum(cd[i]["draft_n"] for i in cd), 1)
    am = sum(cm[i]["draft_n_accepted"] for i in cm)/max(sum(cm[i]["draft_n"] for i in cm), 1)
    emit(f"| {n} | {ad:.1%} | {am:.1%} | {(ad-am)*100:+.1f} |")
emit("\n## 4. MTP vs DFlash (same Qwen3.6-27B-IQ4_XS target)\n")
sweep("campmtp2", "mtp_", [1, 2, 3, 4, 6], "MTP (native nextn)")
emit("(compare to DENSE DFlash table above at n=1..6)")
emit("\n## 5. Thinking vs non-thinking regime (n subset)\n")
sweep("camp3t", "dense_", [1, 3, 6, 8, 15], "DENSE thinking-on")
sweep("camp3t", "moe_", [1, 3, 6, 8, 15], "MoE thinking-on")
emit("(compare to the non-thinking main-sweep tables above)")
emit("\n## 6. temp=0.7 (sampling) crossover check\n")
sweep("campt07", "dense_", [3, 6, 8], "DENSE temp0.7")
sweep("campt07", "moe_", [3, 6, 8], "MoE temp0.7")
emit("\n## 7. Losslessness (bit-identical output count; placement/routing artifact, not lossy design)\n")
losslessness("campfd2", "fd_", [1, 4, 8], "fast-dense (single-GPU)")
losslessness("camp3", "dense_", [1, 4, 15], "DENSE (2-GPU split)")
losslessness("camp3", "moe_", [1, 4], "MoE (2-GPU split)")

os.makedirs("results", exist_ok=True)
open("results/summary_tables.md", "w").write("\n".join(OUT) + "\n")
sys.stderr.write("\nwrote results/summary_tables.md\n")
