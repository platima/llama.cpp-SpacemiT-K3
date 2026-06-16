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
| `llama-completion` | rejects `--spec-type` with an error (patch 8 close-out) |
| `llama-bench` | not wired |

## Probe instrumentation (patch 14)

`GGML_OP_TIMING=1` enables an env-gated per-named-tensor wall-clock recorder in the CPU graph dispatcher. Filters to `mtp_*` / `h_*` prefixes by default; use `GGML_OP_TIMING_ALL=1` to record every named tensor. Prints a sorted table at exit. Zero overhead when unset.

Any future fusion patch must show ≥ 2% of decode wall-clock in the target region via this probe before coding.

```bash
GGML_OP_TIMING=1 llama-speculative-simple ...  2>&1 | grep -A100 GGML_OP_TIMING
```

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

See [`TODO.md`](TODO.md). Shipped patches: 1–16 (patches 15 and 16 are deferred/dismissed with empirical close-out rationale). Each entry records what was tried and why it was kept or dropped.

The `--version` stamp in `common/arg.cpp` prints the current patch level so a runtime check identifies exactly which patches a deployed binary carries.

## Known issues

- `CPU_RISCV64_SPACEMIT: open(/dev/tcm_sync_mem) failed, errno=2` at startup. Expected on stock K3 firmware — `/dev/tcm_sync_mem` is a cross-core barrier-synchronisation device absent from the shipped kernel. The heap fallback is functionally correct and the log line is intentionally loud so any regression in the fallback stays visible.
- `version: 9481 (161be67d6)` (the upstream-style stamp) reports the local cherry-pick tip, not an upstream commit. The fork-specific `--version` lines print the upstream base (`354ebac8c`) separately to disambiguate.

## Relationship to upstream

`master` tracks `ggml-org/llama.cpp:master`; `platima-mtmd` carries the SpacemiT base plus the patches above. Pulling in newer upstream changes is a separate workflow (see "Rebase onto a personal fork" in `TODO.md`) and not done routinely.

Changes in this directory and in patches stamped `platima-spacemit:` are not intended to be PR'd back upstream. Cherry-picks of upstream commits (e.g. patch 9, `e95dae18d`) are already in upstream and were backported here.
