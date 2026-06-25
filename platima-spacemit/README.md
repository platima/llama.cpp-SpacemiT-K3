# Platima SpacemiT-K3 fork of llama.cpp

A fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) tracking the SpacemiT MTMD release line, with MTP (Multi-Token Prediction / speculative decoding) and multimodal wiring for the SpacemiT BananaPi BPI-F3 / Muse Pi K3 RISC-V SoC.

Work branch: **`platima-mtmd`**. The `master` branch tracks upstream `ggml-org/llama.cpp:master` with none of these patches applied — it's kept clean so rebasing onto a newer upstream tip is straightforward.

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

### Multimodal

| Modality | Status |
|---|---|
| Vision (mmproj) on Gemma 4 | working via `llama-mtmd-cli --jinja --image ...` |
| Audio (mmproj) on Gemma 4 | working via `llama-mtmd-cli --jinja --audio ...` |

End-to-end smoke test results are in [`TODO.md`](TODO.md) under "Functional test".

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

## Vision encode speedup (patch 21)

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

Override with `LLAMA_VISION_BF16_TO_F16` / `LLAMA_VISION_BF16_TO_Q8_0` set to
`0`/`false`/`off` to force a tier off, or any other value to force it on. Models whose
mmproj is already F16 (e.g. Qwen 3.5) have no 2D bf16 weights, so both tiers are a no-op
there (verified: 0 vision tensors rerouted, encoder stays on the f16/RVV path).

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

Implemented on branch `platima-mtmd-tier2-ime2-vision`. Requires `-t 8` (the spacemit
affinity path aborts above 8 threads) and `--jinja` for Gemma 4 — both pre-existing,
unrelated to these tiers.

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

See [`TODO.md`](TODO.md). Shipped patches: 1–14, 18 (Gemma4-assistant fit-probe log downgraded ERROR→DEBUG — the "MTP silently falls back" report was a misdiagnosis; MTP already works), 20 (Gemma 4 vision garbled-output fix — the custom IME2 transpose-cont kernel corrupts the vision encoder's F32 `ggml_cont(ggml_transpose(...))`; `GGML_OP_CONT` is now routed to generic CPU), 21 (vision encode faster — bf16 mmproj weights re-typed/quantized for the vectorised F16 or IME2-int8 path; Tier 1 bf16→F16 ~24×, Tier 2 bf16→q8_0 a further ~1.9×; now auto-gated on CPU capability via the new `ggml_cpu_vec_dot_is_simd` predicate — see above). Deferred/dismissed: 15 (buffer-unification refactor), 16 (X100 sampling threadpool), 17 (trunk-graph probe — ROPE-RVV and Q4_1 HP-unlock both fail the ≥2%-of-decode gate), 19 (`llama-completion` spec args — the tool has no speculative loop). Each entry records what was tried and why it was kept or dropped.

The `--version` stamp in `common/arg.cpp` prints the current patch level so a runtime check identifies exactly which patches a deployed binary carries.

## Known issues

- `CPU_RISCV64_SPACEMIT: open(/dev/tcm_sync_mem) failed, errno=2` at startup. Expected on stock K3 firmware — `/dev/tcm_sync_mem` is a cross-core barrier-synchronisation device absent from the shipped kernel. The heap fallback is functionally correct and the log line is intentionally loud so any regression in the fallback stays visible.
- `version: 9481 (161be67d6)` (the upstream-style stamp) reports the local cherry-pick tip, not an upstream commit. The fork-specific `--version` lines print the upstream base (`354ebac8c`) separately to disambiguate.

## Relationship to upstream

`master` tracks `ggml-org/llama.cpp:master`; `platima-mtmd` carries the SpacemiT base plus the patches above. Pulling in newer upstream changes is a separate workflow (see "Rebase onto a personal fork" in `TODO.md`) and not done routinely.

Changes in this directory and in patches stamped `platima-spacemit:` are not intended to be PR'd back upstream. Cherry-picks of upstream commits (e.g. patch 9, `e95dae18d`) are already in upstream and were backported here.
