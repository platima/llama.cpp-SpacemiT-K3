# Platima SpacemiT-K3 fork of llama.cpp

A fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) tracking the SpacemiT MTMD release line, with additional Multi-Token Prediction (MTP) and multimodal wiring for use on the SpacemiT BananaPi BPI-F3 / Muse Pi K3 RISC-V SoC.

The work branch is **`platima-mtmd`**. The `master` branch tracks upstream `ggml-org/llama.cpp:master` and intentionally carries none of these patches — it exists so the rebase onto a newer upstream tip stays clean.

## Hardware target

SpacemiT K3 SoC:

- Cores 0–7: **X100** general-purpose RISC-V application cores (RVV, no IME2 matrix instructions).
- Cores 8–15: **A100** AI cores (RVV + IME2 + TCM).

The SpacemiT backend pins all compute threads to cores 8–15 (`cpu_mask: ff00`, `perfer_core_arch_id: a064`). The X100 cores are unused for compute by design — see patch 16 for the empirical reason.

Build verification: `llama-cli --version` must print `use_ime2: 1` in the startup banner. If it doesn't, the system `libggml` is shadowing the in-tree one — set `LD_LIBRARY_PATH` to the build's `bin/` directory.

## What this fork adds on top of the SpacemiT base

### MTP (speculative decoding via multi-token prediction)

| Architecture | Status | Notes |
|---|---|---|
| Qwen 3.5 4B / 9B | net-positive | Post-output-norm tap (patches 4/5); D2D-copy reduction backport (patch 9, +43–53% tg) |
| Qwen 3.5 MoE | wired | Same tap dispatch as dense Qwen 3.5 |
| Gemma 4 E2B | net-positive at `n_max=2` | Tuned in patch 10 sweep |
| Gemma 4 E4B | net-positive at `n_max=3` | Tuned in patch 10 sweep |
| Gemma 4 12B | net-positive at `n_max=4` | Tuned in patch 10 sweep |
| Gemma 4 Assistant (E2B / E4B / 12B) | wired | Patch 7 fixes the `--model-draft` MTP path |

Multi-modal:

| Modality | Status |
|---|---|
| Vision (mmproj) on Gemma 4 | working via `llama-mtmd-cli --jinja --image ...` |
| Audio (mmproj) on Gemma 4 | working via `llama-mtmd-cli --jinja --audio ...` |

End-to-end smoke test results are recorded in [`TODO.md`](TODO.md) under "Functional test".

### Tool-level MTP awareness

| Tool | MTP-aware? |
|---|---|
| `llama-server` | yes (already, shared with `common_speculative_draft/_accept`) |
| `llama-cli` | yes (patch 8 — via shared server-context paths) |
| `llama-speculative-simple` | yes (patches 2, 6b, 6c, 7) |
| `llama-completion` | rejects `--spec-type` with an error (patch 8 close-out) |
| `llama-bench` | not wired |

### Probe instrumentation (patch 14)

`GGML_OP_TIMING=1` enables an env-gated per-named-tensor wall-clock recorder in the CPU graph dispatcher. By default filters to `mtp_*` / `h_*` prefixes (override with `GGML_OP_TIMING_ALL=1`). `atexit()` prints a sorted table. Zero overhead when the env variable is unset. Intended to enforce a discipline: any future fork-side fusion patch should show its target region ≥ 2% of decode wall-clock before coding.

## Build

```bash
./platima-spacemit/build.sh           # configure + build all targets
./platima-spacemit/build.sh clean     # remove build/ first
./platima-spacemit/build.sh <target>  # single target (e.g. llama-cli)
```

The script enforces the SpacemiT toolchain flags (`-march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d`, no auto-vectorization) via `cmake/riscv64-spacemit-linux-gnu-gcc.cmake`. Bypassing it (`cmake -B build` without the toolchain file) silently uses system GCC defaults and produces a working-but-suboptimal binary.

Verify the build:

```bash
LD_LIBRARY_PATH=$PWD/build/bin $PWD/build/bin/llama-cli --version
# Expect: 'use_ime2: 1' in the startup banner.
```

## Bench

```bash
./platima-spacemit/bench.sh
```

Runs a 4-way matrix (FA on/off × MTP on/off) on a small Qwen 3.5 model, captures per-thread CPU affinity histograms (to catch any threads leaking onto X100 cores), and emits `tg` / `pg` figures. Variance discipline for any K3 MTP perf claim is **`-n 500 × 3`**; short benches (`-n 60`) are warmup-contaminated.

Per-run measurements accumulate in [`results.log`](results.log).

## Patch history

See [`TODO.md`](TODO.md). Shipped patches: 1–14. Patch 15 (buffer-unification refactor) is deferred; patch 16 (X100 sampling threadpool) and patch 17 (trunk-graph probe — ROPE-RVV and Q4_1 HP-unlock both fail the ≥2%-of-decode gate) are dismissed. Each entry in TODO.md records the empirical close-out rationale.

The `--version` stamp in `common/arg.cpp` mirrors the same patch list and prints on every binary's `--version` invocation, so a runtime check tells you exactly which patches a deployed binary carries.

## Known issues

- `CPU_RISCV64_SPACEMIT: open(/dev/tcm_sync_mem) failed, errno=2` at startup. Expected on stock K3 firmware — `/dev/tcm_sync_mem` is a cross-core barrier-synchronization device not present in the shipped kernel. The fallback path (heap allocation) is functionally correct, and the log line is left loud so any regression in the fallback is visible.
- `version: 9481 (161be67d6)` (the upstream-style stamp) reports the local cherry-pick tip, not an upstream commit. The fork-specific `--version` lines now print the upstream base separately (`354ebac8c`) to disambiguate.

## Relationship to upstream

This is a fork in the GitHub sense — `master` tracks `ggml-org/llama.cpp:master`, the `platima-mtmd` branch carries the SpacemiT base plus the patches listed above. Pulling newer upstream changes is a separate workflow (see the "Rebase onto a personal fork" section in `TODO.md`) and not done routinely.

Changes in this directory and in patches stamped `platima-spacemit:` are not intended to be PR'd back upstream. The MTP wiring patches (e.g. patch 9 cherry-pick of `e95dae18d`) are already upstream and were backported here.
