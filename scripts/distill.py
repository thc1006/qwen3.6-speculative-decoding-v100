#!/usr/bin/env python3
"""
Distill the raw per-request JSON into one analysis-ready CSV.
Usage: python scripts/distill.py [data_dir] > data/distilled/all_records.csv
Columns: phase, model, config, n_max, prompt_idx, category, baseline_flag,
         predicted_per_second, predicted_n, prompt_n, prompt_ms, predicted_ms,
         draft_n, draft_n_accepted
"""
import json, glob, os, sys, csv
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
PROMPTS = "prompts/prompts40.json"
cats = {i: c for i, (c, _) in enumerate(json.load(open(PROMPTS)))}
# phase-dir -> (phase_label, regime/sampling notes)
PHASES = {
    "camp3":   "main_sweep_nonthinking_greedy",
    "campfd2": "fast_dense_control_nonthinking_greedy_singleGPU",
    "camp3t":  "thinking_greedy",
    "campt07": "nonthinking_temp0.7_topp0.8_seed42",
    "campmtp2":"mtp_nonthinking_greedy",
}
w = csv.writer(sys.stdout)
w.writerow(["phase", "model", "config", "n_max", "prompt_idx", "category", "baseline_flag",
            "predicted_per_second", "predicted_n", "prompt_n", "prompt_ms", "predicted_ms",
            "draft_n", "draft_n_accepted"])
for sub, phase in PHASES.items():
    for f in sorted(glob.glob(os.path.join(DATA, sub, "*_p*.json"))):
        base = os.path.basename(f)[:-5]              # e.g. dense_n8_p12
        name, pidx = base.rsplit("_p", 1)
        pidx = int(pidx)
        # name like dense_n8 / moe_baseline / fd_n1 / mtp_baseline
        parts = name.split("_")
        model = parts[0]                              # dense/moe/fd/mtp
        cfg = "_".join(parts[1:])                     # baseline / n8
        is_base = cfg == "baseline"
        nmax = 0 if is_base else int(cfg[1:])
        try:
            t = json.load(open(f)).get("timings", {})
        except Exception:
            continue
        if t.get("predicted_n", 0) <= 0:
            continue
        w.writerow([phase, model, cfg, nmax, pidx, cats.get(pidx, "?"),
                    int(is_base),
                    round(t.get("predicted_per_second", 0), 4), t.get("predicted_n"),
                    t.get("prompt_n"), round(t.get("prompt_ms", 0), 3), round(t.get("predicted_ms", 0), 3),
                    t.get("draft_n", 0), t.get("draft_n_accepted", 0)])
