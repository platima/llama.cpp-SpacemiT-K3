# Platima SpacemiT-K3 fork of llama.cpp

A fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) tracking the SpacemiT MTMD release line, with MTP (Multi-Token Prediction / speculative decoding) and multimodal wiring for the SpacemiT BananaPi BPI-F3 / Muse Pi K3 RISC-V SoC.

Work branch: **`platima-mtmd`**. The `master` branch tracks upstream `ggml-org/llama.cpp:master` with none of these patches applied — it's kept clean so rebasing onto a newer upstream tip is straightforward.

> **Companion docs:** [`MODELS.md`](MODELS.md) is the model-centric guide (which model runs on which core, in which compute format, and what's actually been validated on K3 hardware). This README is organized by feature; [`TODO.md`](TODO.md) is the full patch-by-patch changelog.

## Hardware target

SpacemiT K3 SoC:

- Cores 0–7: **X100** general-purpose RISC-V application cores (RVV, no IME2 matrix instructions).
- Cores 8–15: **A100** AI cores (RVV + IME2 + TCM).

The SpacemiT backend pins all compute threads to cores 8–15 (`cpu_mask: ff00`, `perfer_core_arch_id: a064`). X100 cores are unused for compute — see patch 16 for the empirical reason.

Build check: `llama-cli --version` must show `use_ime2: 1` in the startup banner. If it doesn't, the system `libggml` is shadowing the in-tree one — set `LD_LIBRARY_PATH` to the build's `bin/` directory first.

## MTP support

### Architecture status

| Architecture | Status | Notes |
|---|---|---|
| Qwen 3.5 4B / 9B | net-positive | Post-output-norm tap (patches 4/5); D2D-copy reduction backport (patch 9, +43–53% tg) |
| Qwen 3.5 MoE | wired | Same tap dispatch as dense Qwen 3.5 |
| Gemma 4 E2B | net-positive at `n_max=2` | Tuned in patch 10 sweep |
| Gemma 4 E4B | net-positive at `n_max=3` | Tuned in patch 10 sweep |
| Gemma 4 12B | net-positive at `n_max=8` | Tuned in patch 10 sweep, +315% sustained tg |
| Gemma 4 Assistant (E2B / E4B / 12B) | wired | Patch 7 fixes the `--model-draft` MTP path |

### Drafter dtype retype (patch 26)

Many MTP/draft heads ship as **bf16**, whose `vec_dot` is scalar on the K3 `-march` (no
`zvfbfwma`) — catastrophically slow, which eats MTP's net speedup. Patch 26 re-types a
*draft/MTP model's* 2D bf16 weights at load: `q8_0` onto the IME2 int8 engine when live,
else vectorised `f16`. **The main model is never touched.** Auto-selected by default
(`q8_0` when IME2 present, else `f16`); override with `LLAMA_DRAFT_BF16_TO=off|f16|q8_0`.
The retype forces the draft load to `use_mmap=false`, and the f32 conversion is
row-block chunked to bound peak staging RAM.

A3 probe (Huihui-gemma-4-12B Q4_K main + a genuine-bf16 12B MTP head, `-n 500 ×3`,
temp 0): bf16 baseline 2.49 t/s → f16 6.21 t/s → **q8_0 7.19 t/s** — all at identical
98.95% accept (377/381). q8_0 wins (+15.8% over f16, 2.89× over bf16), so it is the
IME2-present default. f16/f32 drafters already run fine and are left alone.

### Multimodal

| Modality | Status |
|---|---|
| Vision (mmproj) on Gemma 4 E2B / E4B (`gemma3` projector) | working via `llama-mtmd-cli --jinja --image ...` |
| Vision (mmproj) on Gemma 4 12B Unified (`gemma4uv` projector) | working — loads after the #24077 cherry-pick; see note below |
| Audio (mmproj) on Gemma 4 | working via `llama-mtmd-cli --jinja --audio ...` |
| Video (frame sequences) on Gemma 4 / Qwen 3.5 | working via `llama-mtmd-cli --jinja --video ...` (patch 25); see note below |

End-to-end smoke test results are in [`TODO.md`](TODO.md) under "Functional test".

#### Video (frame-sequence) support — patch 25

Ported upstream #24269 surgically (the fork predates the lazy/placeholder-media base
that PR sits on). Adds an ffmpeg-shelling video helper in `mtmd-helper.{h,cpp}` (vendored
`sheredom/subprocess.h`), so `llama-mtmd-cli --video foo.mp4` (or the `/video` command)
extracts frames and feeds them as an image sequence via the existing `load_media`
image→video fallback. Verified on Gemma 4 E2B/12B and Qwen 3.5 4B (2 s clips, coherent
multi-frame descriptions).

**Ceiling is prefill, not encode (P3, 2026-07-07).** Video wall-clock is dominated by the
LLM **prefill of the vision tokens** (~2200–2300 tokens/frame at 720p), which grows
super-linearly in frame count (growing-KV attention) — *not* by the CLIP encode. The IME2
`LLAMA_VISION_F16_TO_Q8_0` retype (below) makes each frame's encode ~8–12% faster but does
not touch prefill, so practical K3 video today is a few frames of a short clip: a 4B
2-frame 720p run completes (baseline 170.9 s vs retype 149.7 s), while 8 frames is
impractical (~30–40 min). Heavy models (27B) are additionally forced to `-ub 128` by
memory. Details in [`TODO.md`](TODO.md) under "mtmd video port".

#### Gemma 4 12B Unified vision (`gemma4uv`) — patch 23

The 12B "Unified" QAT model uses an **encoder-free** projector (`gemma4uv` for vision,
`gemma4ua` for audio): raw patches → conv `patch_embd` → position embeddings → a single
`mm.input_projection` linear → straight into the decoder (`n_layer=0`). Its 11-tensor /
175 MB mmproj is therefore *complete*, not truncated, and `patch_size` becomes 48 by
design (file's 16 × n_merge 3, the merge folded into the conv layer).

Our fork was ~2 months behind upstream and threw `unknown projector type: gemma4uv`.
The runtime fix is upstream commit `a731805ce` *"mtmd, model: allow skip build_vit()
(#24077)"* — **not** the widely-cited PR #24118, which only touches `conversion/gemma.py`.
Cherry-picked onto branch `platima-mtmd-gemma4uv` (off the Tier-2 vision branch). Builds
clean; 12B encodes a normal image in ~55 ms.

**Known limitation — tiny images misread (not a port bug).** On the K3, the 12B model
misreads a very small 236×214 "Hi" test image (sees a "narrow strip / bottles") while
x86 upstream reads it correctly. This was exhaustively bisected and is **not** a
SpacemiT-port defect — every fork-specific cause was ruled out: threading race
(`-t 1` == `-t 8`, byte-identical), `ggml_im2col` (plain scalar C, identical to x86), all
SpacemiT RVV vision kernels (routed the whole vision encode to generic ggml-cpu), f16
matmul precision (forced patch + input projection to f32), the IME2 decoder (disabled IME
mul_mat, ran the whole 12B on generic Q4_K), and preprocessing/geometry/layout (dumped
the preprocessed tensor — a perfect clean "Hi", correct 336×336 / 7×7 grid, correct
HWC→planar de-interleave). The only remaining K3-vs-x86 difference is inherent RISC-V
vectorised FP (RVV reduction order/rounding vs x86 AVX; no fast-math flag involved). It
only bites because upstream itself documents this encoder-free model as performing "quite
poor with small images" — tiny low-information inputs sit on its decision boundary and
sub-ulp numeric noise tips the read. **Normal images work fine** (`Test3.jpg` reads
correctly: "a laboratory… a person").

**The lever is spatial composition, not resolution (measured, patch-24 investigation).**
Bicubic-upscaling the tiny "Hi" to 256²/384²/512² does **not** fix the read — token count
scales with size (`set_limit_image_tokens(40,280)`) but the 12B still perceives "vertical
shapes / a thin strip" at every size. What *does* fix it is **padding** the small image
onto a larger canvas so the glyphs occupy a margin of whitespace (matching how text
appears in training photos): the read flips to a correct "Hi". So the mitigation is
*center-on-canvas-with-margin*, not a plain upscale. We chose **document-only** (no code):
a general auto-pad would distort non-text / non-white-background inputs, and the failure
condition (a tiny full-frame glyph image) is a synthetic edge case — real photos and
screenshots carry their own margins. Upstream's own mitigation
(`set_limit_image_tokens(40,280)`, `clip.cpp`) is already in our fork. Repro:
`platima-spacemit/run_gemma4uv_size_sweep.sh`.

### Tool MTP awareness

| Tool | MTP-aware? |
|---|---|
| `llama-server` | yes (already, shared with `common_speculative_draft/_accept`) |
| `llama-cli` | yes (inherits server-context MTP wiring) |
| `llama-speculative-simple` | yes (patches 2, 6b, 6c, 7) |
| `llama-completion` | no — rejects `--spec-type` by design (patches 8, 19) |
| `llama-bench` | not wired |

`llama-completion` has no speculative-decode loop, so the spec flags stay rejected rather than parse-and-ignore (which would silently drop them). Run MTP through `llama-cli`, which routes via the MTP-aware `server_context`.

## Probe instrumentation (patch 14)

`GGML_OP_TIMING=1` enables an env-gated per-named-tensor wall-clock recorder in the CPU graph dispatcher. Filters to `mtp_*` / `h_*` prefixes by default; use `GGML_OP_TIMING_ALL=1` to record every named tensor. Prints a sorted table at exit. Zero overhead when unset.

Any future fusion patch must show ≥ 2% of decode wall-clock in the target region via this probe before coding.

```bash
GGML_OP_TIMING=1 llama-speculative-simple ...  2>&1 | grep -A100 GGML_OP_TIMING
```

## Vision encode speedup (patches 21 / 22 / 27)

Gemma 4 mmproj vision weights ship as **bf16**. The SpacemiT toolchain `-march`
(`rv64gcv_zfh_zvfh_…`) has vectorised F16 (`zfh`/`zvfh`) but **no** bf16 vector
extension (`zvfbfwma`), so `ggml_vec_dot_bf16` falls to a scalar element-by-element
path — making the CLIP encode painfully slow.

**Tier 1 — bf16→F16 (portable).** Re-types 2D bf16 vision weights to F16 at load
(data converted bf16→f32→f16), routing the encoder mul_mats onto the vectorised
`ggml_vec_dot_f16` RVV kernel.

**Tier 2 — bf16→q8_0 (SpacemiT K3 IME2).** Quantizes 2D bf16 vision weights to `q8_0`
and places them in the spacemit repack buffer so the encoder mul_mats dispatch onto the
IME2 int8 matrix engine. Takes precedence over Tier 1 for the bf16 2D weights it can
handle (`ne[0] % 32 == 0`); Tier 1 still covers any that don't fit the q8_0 block size.

**Tier 3 — F16→q8_0 (SpacemiT K3 IME2, patch 27).** The same q8_0/IME2 reroute, but for
mmproj files that are *already F16* (e.g. Qwen 3.5). F16 mul_mats otherwise run on the
vectorised RVV `zvfh` path and **never** reach IME2; quantizing the 2D F16 weights to
`q8_0` into the spacemit repack buffer dispatches them onto the int8 engine (96 vision
tensors rerouted on Qwen3.5-4B mmproj-F16). **Default-on** where IME2 is live (matching
the bf16 path), env `LLAMA_VISION_F16_TO_Q8_0`. The win is **resolution-dependent**:
~8–12% faster CLIP encode at native (~1600px) resolution or on video, and net-flat once
images are downscaled small enough that the encode stops being the wall-clock bottleneck
(model load + LLM prefill + generation dominate). A 199-image Qwen3.5-4B greedy-decode A/B
found **0 accuracy regressions** (weights are identical regardless of resolution), which
is why it ships default-on. Opt out with `LLAMA_VISION_F16_TO_Q8_0=0`.

### Auto-gating (no flags needed)

Both tiers self-gate on CPU capability — no env var required:

- **Tier 1** keys off a new ggml predicate, `ggml_cpu_vec_dot_is_simd(enum ggml_type)`
  (declared in `ggml-cpu.h`, defined in `vec.cpp`). It mirrors the per-type SIMD
  guards inside `ggml_vec_dot_f16`/`_bf16`. Tier 1 enables when F16 is vectorized but
  bf16 is scalar (true on the K3 `-march`; false on x86/ARM where bf16 upconverts to
  vectorised f32, so the retype correctly stays off there). This is portable, not a
  hardcoded arch check.
- **Tier 2** locates the `CPU_RISCV64_SPACEMIT` extra buffer type and probes that a
  small q8_0 tensor actually repacks onto it (`tensor->extra` set after alloc),
  confirming the IME2 int8 engine is live. (`use_ime2` isn't externally exported, hence
  the probe.)

Override with `LLAMA_VISION_BF16_TO_F16` / `LLAMA_VISION_BF16_TO_Q8_0` /
`LLAMA_VISION_F16_TO_Q8_0` set to `0`/`false`/`off` to force a tier off, or any other
value to force it on. Models whose mmproj is already F16 (e.g. Qwen 3.5) have no 2D bf16
weights, so **Tiers 1 and 2** are a no-op there — but **Tier 3** (F16→q8_0, default-on)
does reroute those F16 weights onto IME2 (96 tensors on Qwen3.5-4B), so an F16 mmproj is
no longer left on the RVV path.

### Measured (IME2 build, auto-gated, `-t 8 --jinja`)

Gemma 4 E4B, `Test.png`, across the manual tier flags:

| | CLIP encode | reads image |
|---|---|---|
| bf16 (Tier 1 & 2 forced off) | 287551 ms | "Hi" ✓ |
| F16 (Tier 1) | 11834 ms (**~24×**) | "Hi" ✓ |
| q8_0 (Tier 2, = auto-gated default on K3) | ~5900 ms (**~1.9× over F16**) | "Hi" ✓ |

Auto-gated test matrix, `Test.png` / `Test3.jpg` (larger, complex workshop scene):

| Model | mmproj | Test.png | Test3.jpg | vision→IME2 | result |
|---|---|---|---|---|---|
| Gemma 4 E2B | bf16 | 5895 ms | 5514 ms | 112 tensors | accurate (workshop, ductwork, toolbox) |
| Gemma 4 E4B | bf16 | 5893 ms | 5474 ms | 112 tensors | accurate (HVAC, glass partition, yellow drill) |
| Qwen 3.5 0.8B | F16 | 501 ms | 21556 ms | 0 (f16 path) | accurate, also caught the Vecteezy watermark |

q8_0 quantization did not regress Gemma description quality. The Vecteezy watermark is
caught only by Qwen — a model-capability difference (Gemma misses it across *all* tiers,
including unquantized bf16), not a quantization artifact. F16 carries more mantissa than
bf16, so Tier 1 is near-lossless; Tier 2 q8_0 held quality on these tests.

Implemented on branch `platima-mtmd-tier2-ime2-vision`. Graph threads auto-clamp to the 8
A100 preferred cores (patch 28), so `-t 8` is optional (still recommended for deterministic
runs); `--jinja` is needed for Gemma 4 — both pre-existing, unrelated to these tiers.

## Build

```bash
./platima-spacemit/build.sh           # configure + build all targets
./platima-spacemit/build.sh clean     # remove build/ first
./platima-spacemit/build.sh <target>  # single target (e.g. llama-cli)
```

The script runs cmake from the repo root and applies the SpacemiT toolchain flags (`-march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d`, auto-vectorisation disabled) via `cmake/riscv64-spacemit-linux-gnu-gcc.cmake`. Skipping the toolchain file silently uses system GCC defaults and produces a working-but-suboptimal binary.

Verify:

```bash
LD_LIBRARY_PATH=$PWD/build/bin $PWD/build/bin/llama-cli --version
# Expect: 'use_ime2: 1' in the startup banner.
```

## Bench

```bash
./platima-spacemit/bench.sh
```

Runs a 4-way matrix (FA on/off × MTP on/off) on the Qwen 3.5 0.8B model, checks per-thread CPU affinity (to catch compute threads leaking onto X100 cores), and emits `tg` / `pp` figures.

The benchmark model is downloaded automatically on first run from HuggingFace:
- **Model** (533 MB): [Qwen3.5-0.8B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
- **mmproj** (for multimodal runs, not used in the 4-way text bench): [mmproj-F16.gguf](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)

Variance discipline for any K3 MTP perf claim: **`-n 500 × 3`**. Short benches (`-n 60`) are warmup-contaminated.

Per-run measurements accumulate in [`results.log`](results.log).

## Patch history

See [`TODO.md`](TODO.md). Shipped patches: 1–14, 18 (Gemma4-assistant fit-probe log downgraded ERROR→DEBUG — the "MTP silently falls back" report was a misdiagnosis; MTP already works), 20 (Gemma 4 vision garbled-output fix — the custom IME2 transpose-cont kernel corrupts the vision encoder's F32 `ggml_cont(ggml_transpose(...))`; `GGML_OP_CONT` is now routed to generic CPU), 21 (vision encode faster, Tier 1 — bf16 mmproj weights re-typed to the vectorised F16 RVV path, ~24×), 22 (vision encode Tier 2 — bf16 weights quantized to q8_0 onto the IME2 int8 engine, a further ~1.9×; both tiers now auto-gated on CPU capability via the new `ggml_cpu_vec_dot_is_simd` predicate — see above), 23 (Gemma 4 12B Unified `gemma4uv` vision enabled via the upstream #24077 cherry-pick; tiny-image misread diagnosed as inherent RISC-V FP × model fragility, not a port bug — see above), 25 (mtmd video / frame-sequence support ported from #24269 — see above), 26 (drafter bf16 → q8_0/f16 retype at load, default-on; keeps a bf16 MTP head off the scalar path — see above), 27 (vision encode Tier 3 — F16 mmproj weights → q8_0 onto IME2, default-on; ~8–12% at native resolution, 199-image A/B found 0 accuracy regressions — see above), 28 (SpacemiT base 0.1.3 → 0.1.6 merge — cherry-picks #8 FunASR audio, #9 Qwen2-VL preproc, #10 LFM2 vision, #11 TCM barrier, #13 worker lanes, #15 MiniCPM-V RISC-V fix, #18 hybrid-recurrent multimodal; also adds graph-thread auto-clamp so `-t 8` is optional — see [`TODO.md`](TODO.md)). Deferred/dismissed: 15 (buffer-unification refactor — dismissed; see TODO), 16 (X100 sampling threadpool), 17 (trunk-graph probe — ROPE-RVV and Q4_1 HP-unlock both fail the ≥2%-of-decode gate), 19 (`llama-completion` spec args — the tool has no speculative loop). Document-only: 24 (tiny-image misread is spatial composition, not resolution — an auto-pad would distort non-text inputs; see TODO). Each entry records what was tried and why it was kept or dropped.

The `--version` stamp in `common/arg.cpp` prints the current patch level so a runtime check identifies exactly which patches a deployed binary carries.

Patches 22 and 23 were developed on the feature branches `platima-mtmd-tier2-ime2-vision`
and `platima-mtmd-gemma4uv`, both now merged into `platima-mtmd` (fast-forward; the
branches remain on origin for reference).

## Known issues

- `CPU_RISCV64_SPACEMIT: open(/dev/tcm_sync_mem) failed, errno=2` at startup. Expected on stock K3 firmware — `/dev/tcm_sync_mem` is a cross-core barrier-synchronisation device absent from the shipped kernel. The heap fallback is functionally correct and the log line is intentionally loud so any regression in the fallback stays visible.
- **Heap-fallback barrier is not crash-safe (`ime.cpp:1747: wait tcm buffer failed`).** When `/dev/tcm_sync_mem` is absent, the backend falls back to a shared-memory barrier at `/dev/shm/tcm_sync_standalone`. If a spacemit process aborts (or is killed) mid-barrier, it leaves that file in a wedged state, and **every subsequent run then aborts** at `ime.cpp:1747` across all 8 A100 cores — regardless of model size (a 0.8B model that ran fine will start crashing too). Larger graphs (e.g. 9B text models) appear more likely to trip the initial failure, especially under sustained back-to-back loads. Recovery: clear the stale file with `sudo rm /dev/shm/tcm_sync_standalone` (it is root-owned in a sticky dir, so a non-root user cannot remove it) or reboot. This is pre-existing backend behaviour, unrelated to the vision tiers, but worth knowing before a long benchmark batch.
- `version: 9481 (161be67d6)` (the upstream-style stamp) reports the local cherry-pick tip, not an upstream commit. The fork-specific `--version` lines print the upstream base (`354ebac8c`) separately to disambiguate.

## Relationship to upstream

`master` tracks `ggml-org/llama.cpp:master`; `platima-mtmd` carries the SpacemiT base plus the patches above. Pulling in newer upstream changes is a separate workflow (see "Rebase onto a personal fork" in `TODO.md`) and not done routinely.

Changes in this directory and in patches stamped `platima-spacemit:` are not intended to be PR'd back upstream. Cherry-picks of upstream commits (e.g. patch 9, `e95dae18d`) are already in upstream and were backported here.
