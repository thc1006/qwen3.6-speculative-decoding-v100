# Methodology

## Hardware & build
- **2× NVIDIA Tesla V100 32 GB (sm_70 / Volta)**, ~900 GB/s HBM2. A production vLLM was co-resident but idle during all runs (baseline throughput CV < 0.1 %).
- **llama.cpp `b9860`**, built **from source for sm_70** (the official prebuilt CUDA images/binaries drop Volta → PTX-JIT failure on V100). Clean toolchain via a user-space conda CUDA 12.4:
  ```bash
  conda create -y -n build --override-channels -c nvidia/label/cuda-12.4.0 -c conda-forge cuda-toolkit
  cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_COMPILER=<conda>/bin/nvcc
  cmake --build build -j
  ```
  (Do **not** use Ubuntu's `nvidia-cuda-toolkit` — it pulls a driver lib and breaks NVML/`nvidia-smi`.)

## Models (all GGUF)
| role | model | quant | source |
|---|---|---|---|
| dense target | Qwen3.6-27B | IQ4_XS | unsloth |
| dense drafter | Qwen3.6-27B-DFlash | Q8_0 | williamliao |
| MoE target | Qwen3.6-35B-A3B | UD-IQ4_XS | unsloth |
| MoE drafter | Qwen3.6-35B-A3B-DFlash | Q8_0 | lym00 |
| fast-dense target | Qwen3-4B (base) | Q4_K_M | unsloth |
| fast-dense drafter | Qwen3-4B-DFlash | bf16 | **self-converted** from z-lab safetensors with this PR's `convert_hf_to_gguf.py --target-model-dir` |
| MTP | Qwen3.6-27B (native nextn) | IQ4_XS-mtp | froggeric |

Note: many community DFlash **GGUFs are incompatible** with b9860 — old converters emit `dflash.target_layer_ids`
(new build wants `dflash.target_layers`) or are missing tensors (`enc.output_norm.weight`). We converted the fast-dense drafter
from the z-lab safetensors with this PR's converter to guarantee a valid drafter. gpt-oss-20b DFlash could **not** be loaded on
b9860 (tensor-count mismatch 91 vs expected 123, both community and self-converted) so no second MoE point is included.

## Measurement protocol
- Server: `llama-server ... -ngl 99 -ngld 99 -c 4096 -fa on --jinja`, endpoint `POST /v1/chat/completions`.
- **Non-thinking primary regime**: request `chat_template_kwargs={"enable_thinking": false}` (real answer tokens; this is the
  less-favorable, production-chat regime). Thinking regime uses `enable_thinking: true`.
- **Fixed generation budget**: `ignore_eos: true, n_predict: 384`. 384 < the natural non-thinking EOS length of the prompt set
  (min 405), so baseline and DFlash each emit **exactly 384 real tokens** → fair paired timing, no degenerate post-EOS tail.
- **Greedy** (`temperature: 0`) for the headline; a `temperature: 0.7, top_p: 0.8, seed: 42` pass for the sampling check.
- **DFlash**: `--spec-type draft-dflash --spec-draft-model <drafter> --spec-draft-n-max N` (drafter on GPU; `-ngld 0` = CPU drafter is a 0.5× net-loss). **MTP**: `--spec-type draft-mtp` with **no** `-md` (the nextn head is inside the target GGUF).
- **40 prompts**, 6 categories (7 code / 7 math / 7 factual / 6 how-to / 7 creative / 6 analysis), generative (long natural outputs).
- **Randomized config order** per model (`shuf`) so GPU thermal/clock drift cannot correlate with n-max; each config is a **fresh server** (single request in flight; `n_parallel=1` behaviour).
- Big models (27B/35B) run 2-GPU tensor-split (they don't fit one 32 GB card next to the resident vLLM); the 4B control runs single-GPU. **This placement difference is a confound for the 4B control** (see caveats).

## Metric definitions
- `speedup(prompt) = dflash.predicted_per_second / baseline.predicted_per_second` (paired by prompt index; greedy so both deterministic per side).
- **geomean speedup** = `exp(mean(log(per-prompt speedups)))`; also report **median** and **%net-loss** (fraction < 1.0). 95 % CI = 10 000-sample bootstrap on the log speedups.
- **end-to-end** speedup uses `predicted_n / ((prompt_ms + predicted_ms)/1000)` (includes prompt processing).
- **acceptance** = `Σ draft_n_accepted / Σ draft_n` (pooled).
- **τ (pooled)** = `Σ draft_n_accepted / (Σ draft_n / n_max) + 1` = mean accepted tokens per verify step.
- **C** = `τ / speedup` = `T_step / T_base` (per-step cost in baseline-token units). **Cost fit** `C(n)=d+v·n` by OLS; `v` = marginal per-step cost in baseline-token units, `d` = fixed intercept.

## Caveats (also in README)
1. **V100-scoped**: magnitudes and crossover-n are bandwidth-dependent; do not extrapolate (cf. the PR author's 2.69× dense on DGX Spark).
2. **Fast-dense-4B control is confounded**: single-GPU (vs 2-GPU for the big models), ~15 pp lower acceptance, and a self-converted bf16 drafter. Treat as *suggestive*; the clean mechanism evidence is the internal dense-vs-MoE contrast (equal acceptance, MoE's absolute drafter cost not inflated).
3. **Not bit-lossless** (throughput only, quality not evaluated): greedy DFlash diverges from plain greedy under 2-GPU tensor-split (FP reduction order) + MoE routing; a single-GPU run at n=1 was 40/40 identical, so this is a placement/routing artifact, not a lossy acceptance rule (`p_min=0`).
4. **Non-thinking is the primary regime**; thinking traces draft better, so these are a lower bound for reasoning-heavy use.
5. **Expert-activation counts were not measured**: the Amdahl reading is a hypothesis; it does not test or exclude the expert-union mechanism.
6. Single sweep (greedy is deterministic; per-prompt spread, not run-to-run noise, dominates — hence 40 prompts + CIs rather than reps). `temp 0.7` adds a sampling pass but was not multiply-seeded per prompt.
