# TODO — SpacemiT K3 llama.cpp follow-ups

Tracking items deferred from the Gemma4 MTP cherry-pick work.

## Patch history (on top of release 0.1.3 / upstream base `354ebac8c`)

- **patch 1**: Cherry-pick of Gemma4 MTP (#23398) and Gemma4 E2B/E4B assistants
  (#24282) from upstream; SpacemiT toolchain flags codified in `build.sh` and
  `cmake/riscv64-spacemit-linux-gnu-gcc.cmake`; `--version` stamp added.
- **patch 2**: Make `examples/speculative-simple` MTP-aware. When
  `--spec-type draft-mtp` is set without `--model-draft` (or with `--model-draft`
  pointing at the same file as the target), create the MTP draft context
  against the target model instead of loading a second copy. Mirrors
  `tools/server/server-context.cpp:1062-1083`. Fixes a ~5 GB → ~15 GB host RAM
  blowup observed on Qwen3.5-4B-Q4_K_M-MTP during warmup, which was OOM-killing
  the run on the K3's 16 GB budget.
- **patch 3**: Investigation patch — confirmed that the ~3% MTP accept rate on
  Qwen3.5-4B-MTP is a tap-mismatch bug in `common/speculative.cpp`, not a model
  or binary issue. `common_speculative_impl_draft_mtp` unconditionally used the
  `nextn` (post-output-norm) tap, while Qwen3.5/MoE graphs populate only
  `t_h_pre_norm`. Verified by cross-tool comparison: `llama-server` (4.1%) and
  patched `llama-speculative-simple` (2.2%) — both broken the same way. No
  code changes; the investigation findings drove patch 4.
- **patch 4**: Arch-aware MTP tap dispatch. Adds
  `llama_model_mtp_uses_nextn(const llama_model *)` returning true for
  Gemma4/Gemma4Assistant, false for Qwen3.5/MoE. `common_speculative_impl_draft_mtp`
  now picks `set/get_embeddings_pre_norm*` vs `set/get_embeddings_nextn*` at
  init and at the three read sites based on the target model. Infrastructure
  shipped, but the pre_norm hypothesis was disproven on bench: Qwen3.5-4B-MTP
  accept rate *dropped* to 1.3% (3/232) vs 2.2% pre-patch. Dispatch fires
  correctly per init log. Conclusion: pre_norm is not what the trained gguf
  expects either — the right tap is post-output-norm (matching upstream
  `166fe2949`), which neither of our existing taps currently exposes for
  Qwen3.5. Patch is kept (dispatch is correct infrastructure even if the
  truth-table needs flipping later).
- **patch 5**: Expose post-output-norm tap for Qwen3.5. Adds
  `res->t_h_nextn = cur` after `build_norm(cur, model.output_norm, ...)` in
  both `src/models/qwen35.cpp` (trunk + MTP) and `src/models/qwen35moe.cpp`
  (trunk + MTP). The trunk-graph emit is gated on `cparams.embeddings_nextn`
  so non-MTP inference keeps the pre-existing reduce-before-norm fast path.
  `llama_model_mtp_uses_nextn` flipped back to return `true` for QWEN35/MoE
  so the patch 4 dispatch now routes Qwen3.5 to the (newly populated) nextn
  APIs. Matches upstream `166fe2949 (qwen35: use post-norm hidden state for MTP)`.
- **patch 6a**: Counter-gated L2-magnitude probes (`probe_l2`) in
  `common_speculative_impl_draft_mtp` at four sites: trunk-row, verify-row,
  draft-input, and MTP-output. Auto-stops after 12 calls per run to limit
  spam. The probes were the diagnostic that revealed patch 6b.
- **patch 6b**: Wire `common_speculative_process()` into
  `examples/speculative-simple`. Previously the example bypassed the
  spec-impl's `process()` hook by calling `llama_decode(ctx_dft, batch_tgt)`
  directly. That worked for plain draft-model speculation (process() for that
  impl is just a decode), but for MTP it meant the trunk's hidden state was
  never captured into `pending_h` → MTP head ran on zero-vector h input. With
  the hook wired, Qwen3.5-4B-MTP accept jumped 4.8% → 47.6% and tg 1.84 →
  4.72 t/s. Still below the 7 t/s non-MTP baseline but no longer net-negative
  by a large margin; remaining gap is per-draft MTP graph runtime cost.
- **patch 6c**: Cold-start fix in `examples/speculative-simple`. After
  patch 6b, iter 2+ had non-zero h propagating through pending_h, but iter 1
  still ran `draft()` against the constructor-zero `pending_h` because no
  `process()` had run yet. Inserted a pre-loop warmup that decodes `id_last`
  on `ctx_tgt`, drives `common_speculative_process(spec, warmup)` to
  capture h(id_last) into pending_h, then rolls back the KV slot on both
  `ctx_tgt` and `ctx_dft` so iter 1 re-decodes `id_last` cleanly. Verified
  via the patch 6a probes: first-iter `draft/input-to-mtp` now reports
  L2≈153.79 (was 0.0). Perf is within run-to-run noise of patch 6b
  (one recovered cycle out of ~22 over a 60-token run), but correctness
  is restored end-to-end.

## Scope (per user, 2026-06-14)

Only these tools need MTP + MMPROJ support for the project goals (Gemma 4
and Qwen 3.5 inference, K3 A100 cores):

- `llama-server` — production endpoint. **Already MTP-aware** (the only tool
  that drives `common_speculative_draft/_accept` correctly and handles
  MTP-against-target without loading the model twice).
- `llama-cli` — interactive use. **MTP loop not wired.** Currently only
  preserves `params.speculative` defaults (`tools/cli/cli.cpp:68`). Needs the
  full draft/accept loop from `speculative-simple` plus the MTP-on-target
  branch from patch 2.
- `llama-bench` — perf benchmarking. **MTP loop not wired.** Same situation.
- `llama-completion` — non-interactive single-prompt. **MTP loop not wired.**
  Has zero references to `common_speculative_*` (`tools/completion/completion.cpp`).
  Footgun: parses `--spec-type` and silently sets `cparams.n_rs_seq=4` which
  widens DeltaNet recurrent state 5× for no benefit. Either wire MTP in or
  reject `--spec-type` with a clear error.

These are the future patch 3+ items.

## MTP acceptance rate (4B, Qwen3.5-4B-Q4_K_M-MTP) — tap-mismatch confirmed

First validated 4B MTP runs (2026-06-14):

| Tool                              | tg       | accept | drafted | accepted |
|-----------------------------------|----------|--------|---------|----------|
| `llama-speculative-simple` (patch 2) | 1.74 t/s | 2.2%   | 224     | 5        |
| `llama-server`                    | 1.93 t/s | 4.1%   | 195     | 8        |

Both tools share `common_speculative_impl_draft_mtp` and produce the same
dismal accept rate (~3%). **This rules out a binary-specific bug** and
strongly points at the hidden-state tap: the fork pins Qwen3.5 to
pre-output-norm via `mtp_on_hybrid_qwen35` (see `src/llama-model.cpp`),
while the upstream commit `166fe2949 (qwen35: use post-norm hidden state
for MTP)` switched to post-norm. The shipped Qwen3.5-4B-Q4_K_M-MTP gguf was
presumably trained / packaged for post-norm.

At ≤5% accept, MTP is net-negative on this hardware (baseline non-MTP
Qwen3.5-4B is 7 t/s, MTP run is ~1.7 t/s — 4× slowdown).

### Patch 3 (investigation, DONE 2026-06-14)

**Confirmed bug location: `common/speculative.cpp:495-496`.**

`common_speculative_impl_draft_mtp` unconditionally calls
`llama_set_embeddings_nextn(...)` and unconditionally reads via
`llama_get_embeddings_nextn*` (lines 597, 631, 692), regardless of target
arch. But the per-model graph code reveals the tap split:

| Arch                                | Sets graph output      | File                                  |
|-------------------------------------|------------------------|---------------------------------------|
| Qwen3.5, Qwen3.5MoE                 | `t_h_pre_norm` (only)  | `src/models/qwen35.cpp:212,630`, `qwen35moe.cpp:235,725` |
| Gemma4, Gemma4 Assistant            | `t_h_nextn` (only)     | `src/models/gemma4.cpp:385`, `gemma4-assistant.cpp:199` |

So for Qwen3.5, `t_h_nextn` is `nullptr` → `embd_nextn` is never populated →
speculative impl reads zero/garbage rows → drafts are noise → ~3% accept on
both `llama-server` and patched `llama-speculative-simple`. Not a model-
quality issue; not the binary; the tap is wired wrong for one of the two
supported arches.

### Patch 4 (shipped 2026-06-14, option 1 — minimal arch-aware dispatch)

Implementation:

- New API `llama_model_mtp_uses_nextn(const llama_model *)` declared in
  `src/llama-ext.h`, implemented in `src/llama-model.cpp` next to
  `llama_model_n_embd_out`. Returns `true` for `LLM_ARCH_GEMMA4` and
  `LLM_ARCH_GEMMA4_ASSISTANT`, `false` for `LLM_ARCH_QWEN35` and
  `LLM_ARCH_QWEN35MOE`. Default for unknown archs is `true` (preserves
  pre-patch behavior on any future arch we haven't classified).
- `common_speculative_impl_draft_mtp` (`common/speculative.cpp`) caches the
  result as `uses_nextn` at init from the target model's arch, then dispatches:
  - init (was lines 495-496): picks `set_embeddings_nextn` vs `set_embeddings_pre_norm`
  - prefill catch-up read: `get_embeddings_nextn` vs `get_embeddings_pre_norm`
  - per-row verify reads: `get_embeddings_nextn_ith` vs `get_embeddings_pre_norm_ith`
  - per-row draft reads on ctx_dft: same `_ith` dispatch
- Log line at init prints the chosen tap so future runs make the decision
  visible at a glance.

Validation (2026-06-14):

- **Qwen3.5-4B-Q4_K_M-MTP**: accept dropped to 1.293% (3/232 drafted), tg
  1.67 t/s. Worse than patch 2's 2.2%. Dispatch log line confirms
  `mtp tap: pre_norm (target arch dictates)` so the wiring is correct —
  the hypothesis "Qwen3.5 wants pre-output-norm" is what failed.
- **Gemma4 E2B/E4B drafters**: regression-test still outstanding. Low
  priority until we have a working Qwen3.5 path; the nextn path is the
  pre-patch behavior for Gemma4 so no regression is expected.

### Patch 5 (shipped 2026-06-14, partial fix — only ~2x improvement)

Implementation matches the proposal:

- `src/models/qwen35.cpp` and `qwen35moe.cpp`: trunk graphs now compute
  `build_norm(..., model.output_norm, ...)` BEFORE the row reduction and set
  `res->t_h_nextn = cur` so the tap captures the full-batch post-output-norm
  hidden state. Gated on `cparams.embeddings_nextn` to preserve the
  reduce-before-norm fast path for non-MTP inference.
- Same files' MTP graphs set `res->t_h_nextn = cur` immediately after
  `mtp_shared_head_norm` so subsequent draft iterations read post-head-norm.
- `llama_model_mtp_uses_nextn` flipped back to return `true` for QWEN35/MoE.

Bench (Qwen3.5-4B-Q4_K_M-MTP):

| Build                  | accept | drafted | accepted | tg t/s |
|------------------------|--------|---------|----------|--------|
| patch 2 (junk nextn)   | 2.2%   | 224     | 5        | 1.74   |
| patch 4 (pre_norm)     | 1.3%   | 232     | 3        | 1.67   |
| patch 5 (post_norm)    | 4.8%   | 208     | 10       | 1.84   |

The ordering (post-norm > zeros > pre-norm) confirms patch 5's tap is the
*correct direction* per upstream `166fe2949`, and it's the only build of the
three where the MTP head is producing meaningfully-informative drafts. But
4.8% is still ~6x short of the ≥30% acceptance typical for trained MTP
heads at n=4. Speculative is still net-negative vs the 7 t/s non-MTP
baseline. Something else is wrong beyond just the tap location.

### Patch 6 (investigation → fix, DONE 2026-06-14)

Patch 6a added counter-gated L2 magnitude probes to
`common_speculative_impl_draft_mtp` at four sites. First bench
immediately exposed the bug: `draft/input-to-mtp L2=0.000` —
the MTP head was being fed all-zero vectors. None of the trunk-side
probes (`process/trunk-row0`, `process/verify-row0`) fired at all.

Root cause: `examples/speculative-simple/speculative-simple.cpp` did
`llama_decode(ctx_dft, batch_tgt)` directly after the target decode,
bypassing `common_speculative_process()`. For the legacy draft-model spec
impl that direct decode was correct (its `process()` is just a decode), so
nobody noticed. For MTP, `process()` is where the trunk's hidden state
gets captured into `pending_h`. Skipping it left `pending_h` at its
constructor zero-init forever, so every draft fed the MTP head a zero
vector.

Patch 6b replaces the direct decode with `common_speculative_process()`,
mirroring the `tools/server/server-context.cpp:3550` pattern. Verified
with the same probes: after the fix, `trunk-row0` and `verify-row0` both
report L2≈153.7, and the next draft's `input-to-mtp` reads L2≈153.7 too —
clean propagation. Accept rate 4.8% → 47.6%, tg 1.84 → 4.72 t/s.

### Patch 6c (shipped 2026-06-14, cold-start fix)

Patch 6b made `process()` populate `pending_h` per target decode, but
**iter 1's `draft()` runs before any `process()` has run**, so it still
reads the constructor-zeroed `pending_h`. The patch 6a probes confirmed
this by showing `draft/input-to-mtp L2=0.000` only on the very first
iteration after the fix landed.

Patch 6c inserts a pre-loop warmup in `examples/speculative-simple` just
after `llama_batch_init(...)`:

1. Build a 1-token batch with `id_last` at `n_past`.
2. `llama_decode(ctx_tgt, warmup)` — populates the trunk's h tap.
3. `common_speculative_process(spec, warmup)` — captures h(id_last) into
   `pending_h` (this call also internally decodes on `ctx_dft`).
4. `llama_memory_seq_rm` on both `ctx_tgt` and `ctx_dft` at `[n_past, -1)`
   to undo the KV writes; `pending_h` survives because it lives in the
   impl struct, not the KV cache.

After the rollback, iter 1's `draft()` consumes the primed `pending_h`
and the loop body re-decodes `id_last` at the same `n_past` as normal.
Probe confirms first-iter `draft/input-to-mtp L2=153.79`. The perf
delta is in run-to-run noise (one extra useful draft cycle over ~22),
but the path is now correct on cycle 1 too.

### Remaining gap to non-MTP baseline (7 t/s)

At 4.72 t/s the MTP path is still ~33% slower than non-MTP. With 47.6%
accept and `n_draft=4`, average accepted-per-cycle ≈ `1 + 4·0.476 ≈ 2.9`,
which would offset ~half of a draft cycle's cost. The per-draft MTP graph
runtime is the bottleneck — each `draft/mtp-out` probe stamped ~30 ms
between iterations. Possible follow-ups (none scoped yet):

1. **IME2 coverage of the MTP block.** Check whether the MTP block's
   eh_proj / attn / ffn matmuls are hitting the A100 fast path or
   falling back to a slower kernel.
2. **`n_max` tuning.** 4 was chosen as a default; with ~50% accept,
   `n_max=2` or `3` might give a better speed/accuracy tradeoff.
3. **Backend sampling.** Re-enabling once the "more than one output per
   seq" issue is solved would skip ~94 ms of CPU sampler time per run.

### Patch 5/6 follow-ups (deferred)

- **Gemma4 E2B/E4B regression test** — confirm Gemma4 drafters still see
  the same accept rate they did before patches 4/5/6. The patch 6b fix
  also benefits Gemma4 (now its `process()` actually runs); expected to
  be a net positive but should be confirmed.

## TCM sync-mem heap fallback

At startup `llama-cli` logs:

```
CPU_RISCV64_SPACEMIT: alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap
```

Status: **expected on current K3 firmware, no fix known.** `/dev/tcm_sync_mem`
is a separate cross-core barrier-synchronization device, distinct from the
TCM allocator itself (TCM proper is healthy — startup also reports
`tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0`, i.e. 3 MB
of real TCM). The heap fallback is functionally correct.

Known facts to check before any "fix":
- The device node simply doesn't exist on stock K3 firmware (`errno=2 = ENOENT`).
- A kernel module exposing `/dev/tcm_sync_mem` would presumably let `mmap()`
  return real TCM-backed pages for the inter-core init barrier; this is a
  driver-side change, not a llama.cpp change.
- Do not "fix" by silencing the log — the fallback path needs to stay loud so
  it's visible if/when shared-mem barriers regress.

## K3 A100 / X100 improvements observed during the merge

- **X100 cores are unused.** The current SpacemiT backend (`ggml-cpu/spacemit/`)
  targets the A100 AI cores only (`perfer_core_arch_id: a064`, `cpu_mask: ff00`,
  i.e. cores 8–15). The K3 also has X100 application cores (0–7) which are
  currently only used for the main thread. Worth exploring whether IME2-less
  RVV paths on the X100s could pick up auxiliary work (rope, norms, sampling).
- **Pre-norm + nextn dual hidden-state extraction.** This patchset now keeps
  *two* parallel hidden-state output buffers (`embd_pre_norm` sized `n_embd`,
  `embd_nextn` sized `n_embd_out`). They cover different MTP drafters —
  Qwen3.5 uses pre-output-norm, Gemma4 uses post-output-norm via
  `nextn_proj_post`. Once both code paths settle, consider whether they can
  be unified behind a single "h_for_drafter" tap with a per-arch flag, to
  avoid the duplicated reserve/output-reorder bookkeeping in `llama-context`.
- **`mtp_on_hybrid_qwen35` is fork-only.** Qwen3.5 MTP wiring lives in
  `src/llama-model.cpp` and isn't in upstream. If we ever rebase onto a much
  newer upstream commit it will need to be re-applied; keep a short note of
  the exact lines so the rebase isn't archaeology.
- **`deepstack_mapping_arr` is missing.** Granite4 Vision (upstream commit
  `64086f2b2`) added this field; a log-print referencing it was removed from
  `llama-model.cpp` during the cherry-pick. If we ever rebase past that
  commit, restore the field plus the print.
- **`f_attention_scale` hard-coded to 1.0 for Gemma4 assistant.** Inherited
  from upstream's reference impl. Verify this is the right value once we
  actually have a Gemma4 assistant gguf in hand to run.
- **Cherry-pick base drift.** `version:` currently shows `9481 (161be67d6)`
  but `161be67d6` is a *local* cherry-pick tip, not an upstream commit. The
  custom `--version` line now also shows the upstream base
  (`354ebac8c`) and the SpacemiT release tag separately so this is no longer
  misleading.

## Rebase onto a personal fork (`github.com/platima/llama.cpp-spacemit`)

Goal: a personal fork tracking a *newer* upstream `llama.cpp` than SpacemiT
themselves are tracking, with the SpacemiT backend + MTP/MTMD changes merged
on top.

Doability — high level, plausible, but with caveats:

1. **Backend code is self-contained.** `ggml/src/ggml-cpu/spacemit/`,
   the `GGML_CPU_RISCV64_SPACEMIT` gates, and the IME2 kernels are isolated
   enough to rebase as a feature branch.
2. **Risk areas** (where SpacemiT has cross-cutting changes that conflict
   with upstream churn):
   - `src/llama-context.cpp` / `llama-graph.cpp` — pre-norm + nextn paths,
     TCM/heap barrier handling.
   - `src/llama-kv-cache*.cpp` — shared-KV plumbing for MTP.
   - `tools/server/server-context.cpp` — `process_chunk(..., smt_ctx, ...)`
     signature divergence and the LingBot-MAP pipeline.
   - `common/speculative.cpp` — pre-norm vs nextn API choice in
     `common_speculative_impl_draft_mtp`.
3. **Suggested approach**:
   - Fork at the *current upstream tip*, not at the SpacemiT release.
   - Apply SpacemiT commits as a series of focused feature patches:
     (a) backend + build flags, (b) MTMD/mmproj server bits,
     (c) Qwen3.5 MTP wiring, (d) any LingBot-MAP server changes.
   - Validate after each patch with `./build/bin/llama-cli --version`
     (`use_ime2: 1`) and the smoke test in §Functional test below.
4. **Versioning**: the patch-level constant in `common/arg.cpp` will need
   to be bumped per merge round so `--version` stays meaningful.

## Functional test (still to do — task #18)

End-to-end smoke test on K3:
- Gemma 4 E4B with `mtp-gemma-4-E4B-it.gguf` drafter, `--spec-type draft-mtp`.
- Qwen 3.5 4B with `Qwen3.5-4B-Q4_K_M-MTP.gguf` drafter, `--spec-type draft-mtp`.
- Flash attention on.
- TCM enabled (verify "tcm is available" at startup; the sync-mem fallback
  above is fine).
- MTMD/mmproj: audio + image + text input via `llama-mtmd-cli` /
  `llama-server`.

Pass criteria: all three input modalities work without heap corruption
(remember: previous over-merge produced
`malloc(): invalid size (unsorted)` during decode) and MTP speculative
decoding produces non-trivial acceptance rates.
