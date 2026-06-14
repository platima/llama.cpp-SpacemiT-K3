## Context

This is SpacemiT's fork of llama.cpp (branch `spacemit-mtmd`, tag `v0.1.3`) which adds a
custom RISC-V backend for the SpacemiT K3's A100 AI cores, using IME2 matrix instructions and
TCM (tightly-coupled memory) integration. I'm compiling natively on the K3 itself (riscv64).

I'm cherry-picking upstream Gemma 4 MTP (Multi-Token Prediction / speculative decoding) commits
onto this fork and there are merge conflicts from roughly 100 commits of divergence. There is a
cherry-pick of `04eb4c446` (llama: add Gemma4 MTP) currently in progress with conflicts. After
it is resolved I also need `7d2b45b4f` (mtp: support for gemma-4 E2B and E4B assistants) applied.

## Conflicted files

Mostly:

- `common/speculative.cpp`
- `src/llama-context.cpp`
- `src/llama-graph.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-model.cpp`
- `src/llama-model.h`
- `src/llama-hparams.h`
- `src/llama-ext.h`
- `src/models/gemma4.cpp`
- `tools/server/server-context.cpp`
- some Python conversion files (`conversion/__init__.py`, `conversion/gemma.py`)

## Hard constraints for conflict resolution

1. Add upstream's MTP / speculative-decoding functionality, but do NOT remove or break
   SpacemiT's existing A100 / IME2 / TCM code. Where a conflict is in the model compute or
   graph path that SpacemiT modified for the A100 backend, prefer SpacemiT's side and only
   layer in the MTP-specific additions from upstream. Take upstream's side only for genuinely
   MTP-specific new code.

2. The fork already contains Qwen3.5 MTP code (`mtp_on_hybrid_qwen35` in `src/llama-model.cpp`,
   around line 2000). Preserve it, do not clobber it.

3. A previous attempt on a different (newer) base produced `malloc(): invalid size (unsorted)`
   heap corruption during the generation / decode path. That came from taking too much of
   upstream's graph / context generation path, which conflicted with SpacemiT's heap-fallback
   barrier code (the fork falls back to heap when `/dev/tcm_sync_mem` is unavailable). Avoid
   reintroducing that: keep SpacemiT's memory / barrier handling intact.

4. Commit after each successfully completed cherry-pick so there are clean checkpoints to roll
   back to if a later step or the build fails. Do not bundle both cherry-picks into one giant
   unreviewable resolution.

## Build (native on the K3)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DGGML_CPU_REPACK=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_RVV=ON \
    -DGGML_RV_ZVFH=ON \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON
cmake --build build --parallel $(nproc) --config Release
```

## Verification

After resolving all conflicts and completing both cherry-picks:

1. Confirm the build compiles cleanly.
2. Run `./build/bin/llama-cli --version` and confirm the startup output still shows
   `use_ime2: 1` (this confirms the A100 backend is intact and was not broken by the merge).
3. Report every conflict resolution where the correct choice was ambiguous, so I can review
   those specific decisions.

## Notes / known-good facts

- The fork's runtime always logs `open(/dev/tcm_sync_mem) failed, errno=2` followed by
  `falling back to heap`. This is expected on current hardware (the driver is missing) and is
  NOT an error to fix. The heap fallback must remain safe.
- `--spec-type draft-mtp` is the correct (post-2026-05-13) flag name; upstream renamed it from
  `--spec-type mtp`. The base tag v0.1.3 is from after that rename.
- Drafter files already downloaded for testing:
  - Gemma: `mtp-gemma-4-E4B-it.gguf`
  - Qwen3.5: `Qwen3.5-4B-Q4_K_M-MTP.gguf`
