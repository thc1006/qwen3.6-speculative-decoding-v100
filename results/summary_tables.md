# Reproduced summary tables

_(regenerate with `python scripts/analyze.py`; every number below comes from the raw JSON in `data/raw/`)_

## 1. Main sweep — dense vs MoE, matched IQ4_XS target + Q8_0 drafter, non-thinking greedy

### DENSE Qwen3.6-27B-IQ4_XS  — baseline 37.65 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 1.261 | 1.270 | 0% | 84.1% | 1.248 | [1.241, 1.279] |
| 2 | 1.411 | 1.410 | 0% | 72.5% | 1.392 | [1.368, 1.455] |
| 3 | 1.397 | 1.389 | 0% | 61.0% | 1.378 | [1.331, 1.464] |
| 4 | 1.463 | 1.456 | 5% | 52.8% | 1.441 | [1.374, 1.555] |
| 6 | 1.249 | 1.224 | 20% | 39.8% | 1.235 | [1.152, 1.353] |
| 8 | 1.087 | 1.013 | 45% | 31.2% | 1.078 | [0.988, 1.194] |
| 11 | 1.061 | 0.986 | 52% | 23.5% | 1.052 | [0.957, 1.180] |
| 15 | 0.995 | 0.903 | 58% | 17.4% | 0.987 | [0.889, 1.111] |

cost fit C(n)=d+v·n :  d=1.529  v=0.1609  R²=0.917  (v = marginal per-step cost in baseline-token units)


DENSE n8 per-category geomean speedup:
  - analysis   0.839× (n=6)
  - code       1.501× (n=7)
  - creative   0.717× (n=7)
  - factual    1.080× (n=7)
  - howto      1.041× (n=6)
  - math       1.560× (n=7)

---

### MoE Qwen3.6-35B-A3B-IQ4_XS  — baseline 96.63 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 1.137 | 1.140 | 2% | 82.4% | 1.126 | [1.118, 1.154] |
| 2 | 1.210 | 1.224 | 10% | 69.2% | 1.195 | [1.166, 1.253] |
| 3 | 1.136 | 1.137 | 25% | 58.8% | 1.124 | [1.081, 1.193] |
| 4 | 1.187 | 1.156 | 25% | 50.9% | 1.172 | [1.109, 1.270] |
| 6 | 1.063 | 1.018 | 48% | 38.5% | 1.052 | [0.974, 1.160] |
| 8 | 0.829 | 0.788 | 68% | 30.3% | 0.826 | [0.754, 0.914] |
| 11 | 0.765 | 0.714 | 72% | 22.6% | 0.763 | [0.692, 0.847] |
| 15 | 0.683 | 0.615 | 80% | 16.6% | 0.683 | [0.612, 0.762] |

cost fit C(n)=d+v·n :  d=1.565  v=0.2592  R²=0.961  (v = marginal per-step cost in baseline-token units)


MoE n8 per-category geomean speedup:
  - analysis   0.661× (n=6)
  - code       1.154× (n=7)
  - creative   0.550× (n=7)
  - factual    0.812× (n=7)
  - howto      0.764× (n=6)
  - math       1.193× (n=7)

## 2. Fast-dense CONTROL (Qwen3-4B, single-GPU, self-converted drafter) — CONFOUNDED, see caveats

### fast-dense Qwen3-4B  — baseline 147.88 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 0.978 | 0.966 | 62% | 69.0% | 0.976 | [0.956, 1.000] |
| 2 | 0.988 | 0.969 | 58% | 57.3% | 0.985 | [0.946, 1.030] |
| 3 | 0.953 | 0.938 | 65% | 47.8% | 0.951 | [0.897, 1.014] |
| 4 | 0.943 | 0.897 | 65% | 40.6% | 0.940 | [0.875, 1.015] |
| 6 | 0.817 | 0.766 | 70% | 30.9% | 0.815 | [0.742, 0.900] |
| 8 | 0.462 | 0.418 | 98% | 24.6% | 0.463 | [0.414, 0.519] |
| 11 | 0.472 | 0.406 | 95% | 19.0% | 0.473 | [0.417, 0.537] |
| 15 | 0.489 | 0.425 | 92% | 15.5% | 0.490 | [0.430, 0.558] |

cost fit C(n)=d+v·n :  d=1.459  v=0.4168  R²=0.873  (v = marginal per-step cost in baseline-token units)

## 3. Acceptance-equality (dense vs MoE, the key de-confounder)

| n-max | dense acc | MoE acc | Δ (pp) |
|---|---|---|---|
| 1 | 84.1% | 82.4% | +1.7 |
| 2 | 72.5% | 69.2% | +3.3 |
| 3 | 61.0% | 58.8% | +2.2 |
| 4 | 52.8% | 50.9% | +1.9 |
| 6 | 39.8% | 38.5% | +1.3 |
| 8 | 31.2% | 30.3% | +1.0 |
| 11 | 23.5% | 22.6% | +1.0 |
| 15 | 17.4% | 16.6% | +0.7 |

## 4. MTP vs DFlash (same Qwen3.6-27B-IQ4_XS target)

### MTP (native nextn)  — baseline 38.09 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 1.401 | 1.418 | 0% | 87.5% | 1.384 | [1.382, 1.418] |
| 2 | 1.544 | 1.538 | 0% | 77.3% | 1.521 | [1.503, 1.583] |
| 3 | 1.567 | 1.602 | 0% | 67.6% | 1.542 | [1.505, 1.629] |
| 4 | 1.598 | 1.631 | 0% | 59.1% | 1.571 | [1.515, 1.683] |
| 6 | 1.360 | 1.385 | 8% | 45.3% | 1.343 | [1.267, 1.456] |

cost fit C(n)=d+v·n :  d=1.081  v=0.2719  R²=0.994  (v = marginal per-step cost in baseline-token units)
(compare to DENSE DFlash table above at n=1..6)

## 5. Thinking vs non-thinking regime (n subset)

### DENSE thinking-on  — baseline 37.66 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 1.280 | 1.271 | 0% | 86.8% | 1.268 | [1.268, 1.293] |
| 3 | 1.475 | 1.440 | 0% | 67.1% | 1.454 | [1.433, 1.522] |
| 6 | 1.350 | 1.288 | 0% | 45.6% | 1.333 | [1.281, 1.429] |
| 8 | 1.231 | 1.154 | 20% | 38.4% | 1.218 | [1.152, 1.321] |
| 15 | 1.156 | 1.080 | 42% | 22.2% | 1.144 | [1.067, 1.256] |

cost fit C(n)=d+v·n :  d=1.599  v=0.1616  R²=0.888  (v = marginal per-step cost in baseline-token units)
### MoE thinking-on  — baseline 96.30 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 1.156 | 1.151 | 0% | 85.1% | 1.145 | [1.144, 1.168] |
| 3 | 1.213 | 1.189 | 2% | 65.9% | 1.198 | [1.175, 1.254] |
| 6 | 1.146 | 1.114 | 25% | 44.5% | 1.133 | [1.083, 1.215] |
| 8 | 0.903 | 0.841 | 72% | 35.7% | 0.899 | [0.843, 0.970] |
| 15 | 0.760 | 0.708 | 80% | 20.2% | 0.758 | [0.699, 0.830] |

cost fit C(n)=d+v·n :  d=1.629  v=0.2632  R²=0.949  (v = marginal per-step cost in baseline-token units)
(compare to the non-thinking main-sweep tables above)

## 6. temp=0.7 (sampling) crossover check

### DENSE temp0.7  — baseline 37.68 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 3 | 1.374 | 1.379 | 2% | 59.4% | 1.356 | [1.309, 1.441] |
| 6 | 1.212 | 1.196 | 28% | 37.9% | 1.200 | [1.115, 1.320] |
| 8 | 1.072 | 1.016 | 48% | 30.6% | 1.063 | [0.973, 1.181] |

cost fit C(n)=d+v·n :  d=1.305  v=0.2369  R²=0.999  (v = marginal per-step cost in baseline-token units)
### MoE temp0.7  — baseline 96.55 tok/s (n=40 prompts, predicted_n!=384: 0)
| n-max | geomean | median | %net-loss | acceptance | e2e_geomean | 95% CI |
|---|---|---|---|---|---|---|
| 3 | 1.124 | 1.080 | 22% | 57.7% | 1.112 | [1.066, 1.182] |
| 6 | 1.039 | 0.995 | 50% | 37.2% | 1.029 | [0.955, 1.129] |
| 8 | 0.805 | 0.734 | 68% | 29.1% | 0.803 | [0.733, 0.888] |

cost fit C(n)=d+v·n :  d=1.347  v=0.3312  R²=0.949  (v = marginal per-step cost in baseline-token units)

## 7. Losslessness (bit-identical output count; placement/routing artifact, not lossy design)


fast-dense (single-GPU) — DFlash-greedy vs baseline-greedy identical-output count (of 40):
  - n1: 40/40 identical
  - n4: 5/40 identical
  - n8: 0/40 identical

DENSE (2-GPU split) — DFlash-greedy vs baseline-greedy identical-output count (of 40):
  - n1: 13/40 identical
  - n4: 18/40 identical
  - n15: 5/40 identical

MoE (2-GPU split) — DFlash-greedy vs baseline-greedy identical-output count (of 40):
  - n1: 1/40 identical
  - n4: 0/40 identical
