# K3 model support, modalities & optimization guide

A model-centric companion to [`README.md`](README.md) (which is organized by patch) and
[`TODO.md`](TODO.md) (the full changelog). This file answers three questions at a glance:

1. **What runs** on the SpacemiT K3, on **which core**, in **which compute format** — [Matrix A](#matrix-a--supported-architectures).
2. **What has actually been validated** on real K3 hardware, by **modality** and **architecture** — [Matrix B](#matrix-b--validated-on-k3-hardware).
3. **What each fork optimization adds, why it helps, and how it works** — [Optimizations](#optimizations-what-why-how).

---

## How the K3 runs a model (the mental model)

The K3 SoC has two RISC-V core complexes:

| Cores | Complex | Capabilities | Used for compute? |
|---|---|---|---|
| 8–15 | **A100** AI cores | RVV + **IME2** int8 matrix engine + TCM | **Yes** — all graph compute is pinned here (`cpu_mask ff00`; graph threads auto-clamp to the 8 preferred cores) |
| 0–7 | **X100** app cores | RVV only (no IME2) | **No** — idle by design; offloading probed and rejected (patch 16, [TODO](TODO.md)) |

So the "core" column below is **A100 for every model** — the meaningful lever is not *which*
core but *which compute path* the weights dispatch to on the A100:

| Weight format in GGUF | Dispatch on A100 | Speed | Notes |
|---|---|---|---|
| K-quants + legacy int quants: `Q2_K` `Q3_K` `Q4_0` `Q4_1` `Q4_K` `Q5_0` `Q5_1` `Q5_K` `Q6_K` `Q8_0` | **IME2 int8 engine** | fastest | the K3's reason for existing (repack set in `spacemit/ime.cpp`) |
| **IQ**-quants (`IQ2_XXS`, `IQ4_XS`, …) | RVV vector | slower | not in the IME2 repack set — prefer a K-quant |
| `f16` | RVV `zvfh` vector | fast | vectorised, but not IME2 |
| `bf16` | **scalar** (no `zvfbfwma`) | catastrophic | retyped away at load — see below |
| IQ-quant **MoE experts** | RVV (no IME2 repack) | slower | measured: Qwen3.6 UD-Q2_K_XL stores ~99% experts as IQ2/IQ3 → RVV; a straight `Q2_K`/`Q3_K` GGUF keeps experts on IME2 at ~same size |

Because bf16 is scalar-only on this `-march`, the fork **retypes bf16 weights at load** so they
never hit the scalar path: vision mmproj (patches 21/22/27) and MTP/draft heads (patch 26) are
converted to `q8_0` (IME2) or `f16` (RVV) automatically. The main text model is never touched.

---

## Matrix A — supported architectures

"Supported" = the arch loads and runs on the K3 through the standard `clip.cpp`/mmproj +
`llama` path in this fork. **Core is A100 (8–15) for all rows.** "Compute path" is the *typical*
dispatch for a normal quant of that model.

### Text / MTP models

| Family | GGUF arch | Typical quant → compute path | MTP | Validated? |
|---|---|---|---|---|
| Qwen 3.5 (0.8B/2B/4B/9B) | `qwen35` | Q4_K/Q6_K → IME2 int8 | ✅ post-norm tap (patch 5) | ✅ text + MTP |
| Qwen 3.5 MoE | `qwen35moe` | Q4_K → IME2 (non-IQ experts) | ✅ same tap dispatch | ✅ wired |
| Qwen 3.6 27B | `qwen35` | Q3_K → IME2 int8 | ✅ (inherits qwen35) | — arch supported (35B-A3B validated instead) |
| Qwen 3.6 35B-A3B (MoE) | `qwen35moe` | K-quant experts → IME2; **IQ experts → RVV** | ✅ (inherits qwen35moe) | ✅ text + MTP |
| Gemma 4 E2B / E4B / 12B | `gemma4` | Q4_0 (QAT) → IME2 int8 | ✅ tuned n_max (patch 10) | ✅ text + MTP |
| Gemma 4 26B-A4B (MoE) | `gemma4` | Q4_0 → IME2 int8 | ✅ wired | — text only |
| Gemma 4 Assistant (E2B/E4B/12B) | `gemma4` | Q4_0 → IME2 int8 | ✅ `--model-draft` path (patch 7) | ✅ wired |
| Llama 3.1 8B / 3.2 3B | `llama` | Q4_K → IME2 int8 | — | — text (baseline) |
| GLM-4.7-Flash | `glm4` | on-disk IQ2_XXS → RVV (use a K-quant for IME2) | — | — (looping fix already in fork) |
| Cydonia 24B (Mistral) | `llama` | Q4_K → IME2 int8 | — | — |

### Multimodal projectors (mmproj / clip.cpp path)

| Model | Projector | Modality | mmproj dtype → vision path | Validated? |
|---|---|---|---|---|
| Gemma 4 E2B / E4B | `gemma4v` + `gemma4a` | vision + audio | bf16 → q8_0/IME2 (Tier 2) | ✅ image + audio + video |
| Gemma 4 12B Unified | `gemma4uv` / `gemma4ua` | vision + audio | bf16 → q8_0/IME2 | ✅ vision (patch 23) |
| Qwen 3.5 | (F16 mmproj) | vision | F16 → q8_0/IME2 (Tier 3) | ✅ image + video |
| Qwen3-VL 8B | `qwen3vl` (deepstack) | vision | F16/Q8_0 → IME2 | ✅ image (downscale input) |
| Qwen2-VL | `qwen2vl` | vision | (SMT patch preproc #9) | — arch supported |
| MiniCPM-V | `minicpmv` | vision | (RISC-V fix #15) | — arch supported |
| LFM2 | `lfm2` / `lfm2a` | vision | (SMT vision #10) | — arch supported |

> The upstream `clip.cpp` in this fork registers ~40 projector types (Pixtral, InternVL,
> Idefics3, GLM4V, Phi4, Kimi-VL, etc.). They are **code-supported** by inheritance but **not
> K3-validated** — treat anything not in Matrix B as untested on this hardware.

### Second (ONNX) multimodal path — not built here

There is a separate SpacemiT **ONNX-Runtime wrapper path** (`mtmd-cli-smt`,
`smt-vision-wrapper`, `smt-audio-wrapper`) gated behind `LLAMA_SERVER_SMT_VISION=ON` +
`SPACEMIT_ORT_DIR`. Our build does **not** enable it (no ORT SDK, no `.onnx` encoders on disk),
so it is entirely **untested by us**. SpacemiT's newest commits (#19 Gemma4-audio, #20 Qwen3-VL,
#21 cleanup) target only this path and have **no effect on the A100/X100 mmproj path** above.

---

## Matrix B — validated on K3 hardware

Only models actually run on the K3 with this fork's patches. "Result" is a functional
smoke (loads + coherent output), not a benchmark unless noted.

| Model | Quant | Modality | GGUF arch | Result |
|---|---|---|---|---|
| Qwen 3.5 4B | Q4_K_M | text | `qwen35` | ✅ coherent |
| Qwen 3.5 4B MTP | Q4_K_M | text + MTP | `qwen35` | ✅ 47.6% accept (patch 6b) |
| Qwen 3.5 4B | Q4_K_M + F16 mmproj | image + video | `qwen35` | ✅ coherent; F16→q8_0 Tier 3 |
| Gemma 4 E2B | Q4_0 QAT | text + MTP | `gemma4` | ✅ net-positive n_max=2 |
| Gemma 4 E2B | Q4_0 + bf16 mmproj | image + audio + video | `gemma4` | ✅ all three |
| Gemma 4 E4B | Q4_0 QAT | vision | `gemma4` | ✅ workshop scene accurate |
| Gemma 4 12B | Q4_0 QAT | text + MTP + vision | `gemma4`/`gemma4uv` | ✅ +315% tg; tiny-image caveat (patch 24) |
| Huihui Gemma 4 12B | Q4_K + bf16 MTP head | MTP (drafter dtype) | `gemma4` | ✅ q8_0 7.19 t/s, 98.95% accept |
| Qwen3-VL 8B Thinking | Q4_K_M + Q8_0 mmproj | image | `qwen3vl` | ✅ accurate scene; downscale input (ViT scales w/ resolution) |
| Qwen 3.6 35B-A3B | UD-Q2_K_XL | text + MTP | `qwen35moe` | ✅ coherent; MTP fires (nextn tap); slow — "Dynamic" quant's experts are **IQ2/IQ3 → RVV, not IME2** (use a straight Q2_K/Q3_K GGUF for IME2 experts) |

Modalities covered on the mmproj path: **text ✅ · image ✅ · audio ✅ · video ✅**. No mtmd
modality is missing.

---

## Optimizations: what, why, how

Condensed and model-centric; full rationale + measurements in [`TODO.md`](TODO.md).

### MTP (Multi-Token Prediction) speculative decode — patches 2–7, 10
- **What:** drives a model's built-in MTP head as a self-speculating drafter (no second model).
- **Why:** on a bandwidth-bound K3, verifying several drafted tokens per decode step amortises
  the memory traffic — net faster *when the drafter is cheap and accurate*.
- **How:** captures the trunk's post-output-norm hidden state (`t_h_nextn`) into the MTP head via
  the `common_speculative_process()` hook; arch-aware tap dispatch picks the right tensor per
  model. Tuned `--spec-draft-n-max` per model (patch 10). Qwen 3.5 4B: 4.8% → 47.6% accept.

### Drafter dtype retype (bf16 → q8_0/f16) — patch 26
- **What:** a draft/MTP head that ships as bf16 is retyped at load to `q8_0` (IME2) or `f16`.
- **Why:** bf16 `vec_dot` is scalar on the K3 — a bf16 drafter is so slow it erases MTP's win.
- **How:** default `q8_0` when IME2 is live, else `f16`; main model untouched; draft forced to
  `use_mmap=false`; f32 conversion row-block-chunked to bound RAM. Measured on a genuine-bf16
  12B head: **2.49 → 7.19 t/s (2.89×)** at identical 98.95% accept. Override `LLAMA_DRAFT_BF16_TO`.

### Vision encode retype (Tiers 1/2/3) — patches 21/22/27
- **What:** vision mmproj weights are retyped so the CLIP encode dispatches to a fast engine —
  bf16→f16 (Tier 1, RVV), bf16→q8_0 (Tier 2, IME2), F16→q8_0 (Tier 3, IME2, default-on).
- **Why:** bf16 encode is scalar (~24× slower); even F16 misses the IME2 int8 engine.
- **How:** self-gates on CPU capability (`ggml_cpu_vec_dot_is_simd` + an IME2 repack probe).
  Gemma 4 bf16: 287551 ms → 11834 ms (f16) → ~5900 ms (q8_0). Tier 3 reroutes 96 F16 tensors on
  Qwen3.5-4B; win is resolution-dependent (~8–12% at native, flat when downscaled); 199-image A/B
  found 0 accuracy regressions. Override `LLAMA_VISION_{BF16_TO_F16,BF16_TO_Q8_0,F16_TO_Q8_0}`.

### Video (frame-sequence) — patch 25
- **What:** `--video foo.mp4` extracts frames (ffmpeg) and feeds them as an image sequence.
- **Why/how:** surgical port of upstream #24269; the ceiling is LLM **prefill** of ~2200 vision
  tokens/frame (grows super-linearly with frames), **not** the encode — so practical K3 video is
  a few frames of a short clip. See [`README.md`](README.md) / [`TODO.md`](TODO.md).

### Probe instrumentation — patch 14
- `GGML_OP_TIMING=1` records per-tensor wall-clock in the CPU dispatcher. Any future fusion patch
  must show ≥2% of decode wall-clock in the target region before it is worth coding — the gate
  that (correctly) killed patches 16/17.

---

## Running each modality

```bash
export LD_LIBRARY_PATH="$PWD/build/bin"
CLI=build/bin/llama-mtmd-cli          # multimodal (needs --mmproj)

# text + MTP  → route through llama-cli / llama-server (MTP-aware); NOT llama-completion
# image
$CLI -m model.gguf --mmproj mmproj.gguf --image pic.jpg -p "Describe this image." --jinja -t 8 -fa 1
# audio
$CLI -m model.gguf --mmproj mmproj.gguf --audio clip.mp3 -p "Transcribe this audio." --jinja -t 8 -fa 1
# video
$CLI -m model.gguf --mmproj mmproj.gguf --video clip.mp4 -p "Describe the video." --jinja -t 8 -fa 1 -ub 512
```

Graph threads **auto-clamp to the 8 A100/IME2 preferred cores** (patch 28,
`ggml_backend_cpu_riscv64_spacemit_max_perfer_threads`), so an over-provisioned `-t` no
longer aborts the affinity path — passing `-t 8` is optional but recommended for explicit,
deterministic runs. `rm -f /dev/shm/tcm_sync_standalone` before a run if a prior spacemit
process aborted mid-barrier (see [`README.md`](README.md) "Known issues").
