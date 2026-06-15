# TODO — SpacemiT K3 llama.cpp follow-ups

Tracking items deferred from the Gemma4 MTP cherry-pick work.

## Patch history on branch `platima-mtmd` (on top of SpacemiT release 0.1.3 / upstream base `354ebac8c`)

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
- **patch 9**: Backport of upstream `e95dae18d (Remove padding and multiple
  D2D copies for MTP, #24086)`. Refactors `ggml_gated_delta_net`:
  state tensor now 4D `[S_v, S_v, H, n_seqs]` (was `(S_v*S_v*H, K, n_seqs)`
  with a padding hack); K (snapshot slot count) is now an op param;
  multi-snapshot per-slot `ggml_cpy` loop in `build_recurrent_attn`
  collapsed to a single strided copy. Hexagon-side conflict resolved
  by taking upstream verbatim (Hexagon backend not built on SpacemiT K3).
  Result on Qwen 3.5: 4B tg 4.67 → 7.16 t/s (+53%), 9B Q3_K_S tg 3.90 →
  5.60 t/s (+43%). MTP is now NET-POSITIVE on both Qwen sizes (was
  net-negative even at 9B before). Gemma E2B sanity check: no change
  (Gemma doesn't use DeltaNet). Single-commit cherry-pick of `c9a10c1d8`.
- **patch 7**: Gemma 4 regression fix in `examples/speculative-simple`.
  Patch 2's MTP-on-target branch set `cparams.ctx_other = ctx_tgt` and
  `cparams.n_rs_seq = 0` on the new MTP draft context, but the *other*
  branch (when `--model-draft` points at a separately-loaded MTP gguf —
  the Gemma 4 Assistant case) only set `cparams.ctx_type` and skipped the
  other two. Gemma4Assistant's init throws "requires ctx_other to be set"
  in that path, so every Gemma 4 MTP run with `--model-draft` died at
  context creation. Mirrors the server pattern at
  `tools/server/server-context.cpp:1043-1061`. Verified end-to-end on
  Gemma 4 E2B (unsloth UD-Q4_K_XL + mtp-gemma-4-E2B-it.gguf): runs
  cleanly, accept = 15% (21/140), tg 11.85 t/s vs 12.54 non-MTP baseline.
  Confirms patches 4/5/6 did not regress the Gemma path — but slightly
  net-negative for MTP on E2B (per-draft graph cost dominates on the
  smaller 1536 embedding dimension).

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

### Upstream survey for MTP perf fixes (2026-06-14)

Scanned `ggml-org/llama.cpp` commits from base `354ebac8c` to `upstream/master`
(160 commits ahead) for anything that might obviate the planned patch 9
(MTP net-negative on small models). Touched paths surveyed: `common/speculative.cpp`,
`src/models/qwen35*.cpp`, `src/models/gemma4*.cpp`, `examples/speculative-simple/`,
`tools/server/server-context.cpp`, `src/llama-context.cpp`, `src/llama-graph.cpp`.

| Upstream commit | What it does | Relevance to patch 9 |
|-----------------|--------------|----------------------|
| `e95dae18d` Remove padding and multiple D2D copies for MTP (#24086) | Refactors `ggml_gated_delta_net` to take only initial state (D,1,n_seqs); K passed as op param. Removes a padding hack + multi-copy D2D pattern in favour of a single strided `ggml_cpy`. | **Qwen3.5 only** — the GDN op is the Qwen3.5 DeltaNet recurrent path, not used by Gemma4 (plain transformer). Does not address the Gemma E2B net-negative case. Could be a small win for Qwen3.5 MTP runtime if backported, but the API change is invasive (touches all 9 backends + delta-net-base.cpp). Defer. |
| `a66d50588` graph: guard iswa kq_mask on its own buffer (#24294) | Defensive null-buffer guard for SWA-only draft heads (StepFun MTP specifically). | Not applicable — Gemma4 SWA isn't a "draft-only" head in our setup. |
| `88a39274e` spec: add EAGLE3 (#18039) | New speculative impl alongside MTP. Adds `common_speculative_impl_draft_eagle3` but does NOT modify `common_speculative_impl_draft_mtp`. | No MTP perf change. Large/intrusive backport if we ever want EAGLE3, but unrelated to E2B's gap. |
| `260862b8c` arg: fix double mtp downloads (#24128) | Skip mtp/mmproj auto-discovery for sub-models (draft/mmproj/vocoder). | Arg-parse hygiene, not perf. Probably worth applying anyway. |
| `7acb4e8cd` hparams: refactor `hparams.n_layer` (#24060) | `n_layer` → `n_layer()`, `swa_layers` → `is_swa_impl`, `n_layer_kv_from_start` reads from `n_layer_all`. | Pure refactor. Would conflict with our patches if/when we rebase. |

**Conclusion**: there is no upstream commit that addresses the Gemma E2B
MTP net-negative case directly. The only MTP-tagged perf commit
(`e95dae18d`) targets DeltaNet, which only Qwen3.5 uses. The E2B gap is
genuinely per-draft transformer-block runtime on a 1536-dim embedding,
which the existing three follow-ups (IME2 coverage, `n_max` tuning,
backend sampling re-enablement) are still the right framing for. Patch 9
would need to be original investigation work, not a cherry-pick. Given
that E4B (+30%) and 12B (+356%) are already net-positive on the same
binary, patch 9 may be deprioritisable depending on whether E2B has to
be MTP-positive specifically.

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

### Qwen 3.5 MTP chat-template + size-scaling test (DONE 2026-06-14)

The Qwen 3.5 4B "MTP head trained but accept only ~45%" mystery was
investigated against the model README. Two leads tested:

1. **Chat template** (P1 — cheap test). Wrapped prompt in the Qwen3
   chat format (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`)
   and re-ran 4B MTP at temp=0. Output became coherent thinking-mode
   text ("Thinking Process: 1. Analyze the Request..."). But accept
   rate moved from 47.6% → 45.5% — **within run-to-run noise**.
   Chat template fixes output quality but does NOT fix accept rate.
   Sampling per README's thinking-mode recommendation (temp=1.0,
   top_p=0.95, top_k=20, presence_penalty=1.5) dropped accept to 16.9%
   as expected (sampling variance ≠ greedy match).

2. **Size scaling** (sanity check against the Gemma pattern). Benched
   Qwen 3.5 9B (Q3_K_S) MTP vs non-MTP:
   - 9B non-MTP: 5.33 t/s
   - 9B MTP: 3.90 t/s, accept 52.5% (42/80)
   - **Δ = −27% — still NET-NEGATIVE at 9B**.

   The accept rate climbs modestly with size (45%@4B → 52%@9B) but stays
   in the 50% range. Compare Gemma: 15%@E2B → 24%@E4B → 96%@12B.

**Key reframe**: Qwen MTP underperformance is NOT a small-model artifact
(Gemma E4B at embd=2048 is already +30% net-positive; Qwen 9B at
embd=3584 is still net-negative). The bottleneck is **Qwen-specific**,
in the DeltaNet/GDN recurrent path. Upstream `e95dae18d` (Remove padding
and multiple D2D copies for MTP, #24086) refactors `ggml_gated_delta_net`
to remove a padding hack and consolidate multiple D2D copies into a
single strided `ggml_cpy` — in exactly that path. The earlier upstream
survey dismissed it as "Qwen-only and we only care about Gemma";
correction: **Qwen is the problem case, so e95dae18d IS the right
backport candidate for patch 9**. Promote it.

### Next priorities (queued for after patch 9 backport)

**Patch 9 shipped 2026-06-14** (commit `c9a10c1d8`). Qwen 4B MTP +53% tg,
9B +43% tg; both archs now MTP-positive. The "Qwen-specific MTP gap" is
resolved.

- **P2: Wire MTP into `llama-cli` and `llama-completion`** —
  **investigated 2026-06-14, no code change needed.** Findings:
  - `llama-cli` (`tools/cli/cli.cpp`) is a thin client over
    `server_context`; instantiates one and submits a task. The MTP
    wiring in `tools/server/server-context.cpp:920-1083` (spec_mtp
    detection → `ctx_type=LLAMA_CONTEXT_TYPE_MTP`, `n_rs_seq=0`,
    `ctx_other=ctx_tgt`) is inherited automatically. Already MTP-aware.
  - `llama-completion` (`tools/completion/completion.cpp`) has no spec
    hook in its main loop, but isn't a footgun: the `--spec-type` arg
    is `.set_examples({SPECULATIVE, SERVER, CLI})` at
    `common/arg.cpp:3672`, so it's filtered out at registration for
    `LLAMA_EXAMPLE_COMPLETION` (filter at `arg.cpp:1075`). Both the
    CLI-flag path and the `LLAMA_ARG_SPEC_TYPE` env-var path skip it
    (env loop at `arg.cpp:500` only iterates registered options). A
    user running `llama-completion --spec-type draft-mtp` gets
    `error: invalid argument: --spec-type` before model load — no
    silent `n_rs_seq` widening. If completion ever does need MTP,
    port the spec loop from `speculative-simple` proper (substantial
    refactor: interactive, ga_n, session cache, conversation,
    antiprompts all interact with the sample site at line 718).
  - Prior memory note ("both silently set n_rs_seq=4") was wrong.
    Closed without patch; ship-state confirmed correct.
- **P3: IME2 coverage audit on the MTP block — DONE 2026-06-14.**
  Instrumented `ggml_backend_riscv64_spacemit_buffer_init_tensor`
  (`ggml/src/ggml-cpu/spacemit/ime.cpp:1395-1416`) with an env-gated
  per-tensor dispatch log. Set `GGML_SPACEMIT_DISPATCH_LOG=1` and lines
  of the form `SPM_DISPATCH|<name>|<type>|<ne[0]>|<ne[1]>|<ime|rvv>`
  emit to stderr at buffer init time (model load), one per tensor that
  enters the SpacemiT buffer. Single short load per arch captures full
  coverage data.

  Findings:
  - **Qwen 3.5 4B Q4_K_M (single-model MTP, layer 32 = MTP block)**:
    209 tensors enter the SpacemiT buffer, 209 get IME repack traits
    (100% IME-eligible, 0 RVV fallbacks). MTP block coverage:
    - `blk.32.attn_q.weight` Q4_K [2560, 8192] → ime
    - `blk.32.attn_k.weight` Q4_K [2560, 1024] → ime
    - `blk.32.attn_v.weight` Q6_K [2560, 1024] → ime
    - `blk.32.attn_output.weight` Q4_K [4096, 2560] → ime
    - `blk.32.ffn_gate.weight` Q4_K [2560, 9216] → ime
    - `blk.32.ffn_down.weight` Q6_K [9216, 2560] → ime
    - `blk.32.ffn_up.weight` Q4_K [2560, 9216] → ime
    - `blk.32.nextn.eh_proj.weight` Q8_0 [5120, 2560] → ime
    - Shared LM head: tied with `token_embd` Q6_K [2560, 248320] → ime
  - **Gemma 4 E2B Q4_K_XL + `mtp-gemma-4-E2B-it.gguf` drafter**:
    300 tensors total (277 target + 23 drafter), 100% IME-eligible.
    Drafter MTP-specific weights:
    - `nextn.pre_projection.weight` Q4_0 [3072, 256] → ime
    - `nextn.post_projection.weight` Q4_0 [256, 1536] → ime

  **Conclusion**: every MTP matmul on both archs hits the IME2 path
  already. The earlier conditional ("if every op already hits IME2 then
  patch 9's headroom is in algorithm, not kernel choice") is now
  confirmed. Further per-draft graph speed has to come from
  algorithmic changes — fewer ops per draft (fuse eh_proj + hnorm?),
  larger draft batches amortising fixed overhead, or reducing the
  number of recurrent-state slots we feed through `ggml_cpy`. There is
  no IME dispatch headroom to chase.

  Instrumentation kept in-tree as a debug aid (env-gated, no
  overhead when unset).

### Patch 5/6/7 follow-ups

- **Gemma4 E2B regression test — DONE 2026-06-14.** Surfaced patch 7 (the
  Gemma path was crashing at init, not silently regressing). With patch 7
  applied: tg 11.85 t/s, accept 15%, coherent output. Net slightly below
  the 12.54 t/s non-MTP baseline; the gap is per-draft MTP graph cost on
  E2B's 1536 embedding dim. Patches 4/5/6 themselves did not regress
  Gemma's accept rate (patch 4's dispatch routes Gemma to nextn just as
  the pre-patch code did unconditionally).
- **Gemma4 E4B regression test — DONE 2026-06-14 (patch 8 in user
  nomenclature).** With patch 7 applied: tg 9.787 t/s vs 7.53 t/s non-MTP
  baseline = **+30% net win**. accept 24.2% (30/124), coherent output.
  E4B's 2048 embedding dim is large enough that per-token decode cost
  outweighs per-draft MTP graph cost — crosses the win/loss threshold
  E2B (embd=1536) sits just below. Confirms patch 7 holds end-to-end on
  the recommended `mtp-gemma-4-E4B-it.gguf`.
- **Gemma4 12B opportunistic bench — DONE 2026-06-14.** Same patch-7
  binary on `unsloth-gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
  with `mtp-gemma-4-12B-it.gguf` drafter: tg 11.32 t/s vs 2.48 t/s
  non-MTP = **+356% (4.5× speedup)**, accept 96.2% (50/52). Caveat: raw
  prompt with `-no-cnv` produces garbage `"1.\n1.\n1."` repeating output
  on the 12B Qat (worked on E2B/E4B with same flags) — 12B likely needs
  chat-template wrapping. Speed comparison still valid since same garbage
  appears with or without MTP.

### Size-dependent MTP win threshold (Gemma 4, patch 7 build, 2026-06-14)

| Model | embd dim | non-MTP tg | MTP tg | Δ        | accept |
|-------|----------|------------|--------|----------|--------|
| E2B   | 1536     | 12.54 t/s  | 11.85  | **−5.5%** | 15.0%  |
| E4B   | 2048     |  7.53 t/s  |  9.79  | **+30%**  | 24.2%  |
| 12B   | 3840     |  2.48 t/s  | 11.32  | **+356%** | 96.2%  |

Crossover sits between E2B and E4B. Two compounding effects: (a) larger
embedding makes each non-MTP decode slower in absolute terms (denominator
grows), and (b) accept rate climbs sharply with model capacity (likely
because the MTP head was trained as a function of the larger trunk's
distribution). Patch 9 (MTP net-negative on E2B) is therefore a
small-model-specific problem; if E2B is acceptable as a non-MTP-only
target the patch 9 work could be deprioritised.

### `--spec-draft-n-max` tuning sweep (DONE 2026-06-14, post-patch-9)

P3-followup. Swept `--spec-draft-n-max ∈ {2, 3, 4}` across every MTP-capable
arch on this fork. All runs `-t 8 --no-mmap -fa 1 --temp 0
--no-spec-draft-backend-sampling -n 60` (Qwen: chat-template prompt; Gemma:
raw "Write a short poem about a robot.").

| Model     | n=2 tg  | n=3 tg  | n=4 tg  | non-MTP | winner | vs non-MTP |
|-----------|---------|---------|---------|---------|--------|------------|
| Qwen 4B   |  7.540  | **8.162** |  7.164  |  ~7.0  | **3**  | +16%       |
| Qwen 9B   |  6.088  | **6.571** |  5.595  |  5.33  | **3**  | +23%       |
| Gemma E2B |**13.406**| (skip)  | 11.837  | 12.54  | **2**  | +6.9% (flipped) |
| Gemma E4B |  9.045  | **10.223**|  9.787  |  7.53  | **3**  | +36%       |
| Gemma 12B | (skip)  | (skip)  |  11.32  |  2.48  | **8**¹ | +315%¹     |

**Key finding**: the upstream default `params.speculative.draft.n_max = 3`
(`common/common.h:303`, changed from 16 → 3 in `b7c91edac` upstream merge)
is already correct for **3 of 5 archs**. Prior `bench.sh` and results-log
rows passed `--spec-draft-n-max 4` everywhere, which was uniformly worse than
the default on Qwen 4B/9B and Gemma E4B. **No code change required — just
stop overriding the default unless the arch falls into the high/low edges.**

- **Edges that should override:** Gemma E2B → `2` (flips MTP net-negative to
  net-positive vs non-MTP baseline). Gemma 12B → `4` (96% accept rate means
  longer drafts almost always commit; the lost cycle at `n_max=3` would shed
  the +356% margin).
- **Why low-accept archs prefer `n_max=2`:** when accept rate <~30%, the
  marginal token at draft step `k>2` is rarely committed, so it's pure
  per-draft graph cost with no payoff. Shortening the draft horizon recovers
  that cost.
- **Why mid-accept archs prefer `n_max=3`:** sweet spot. Long enough that
  the expected `1 + n_max·p_accept` accepted-per-cycle dominates the fixed
  per-cycle setup, short enough that marginal-step waste isn't yet a tax.
- **Why 12B prefers `n_max=8` (¹CONFIRMED big win — see patch 10):** the
  n=5/6/8 follow-up sweep (patch 10, 2026-06-15) first looked marginal at
  60 tokens: n=4 → 11.31, n=5 → 11.40, n=6 → 10.21 (reproducible dip),
  n=8 → 11.67 — best is +3.2%. But the 60-token bench was warmup-
  contaminated. At -n 500 (sustained-throughput regime) with 3 runs each
  for variance:
    - n=4 @ 500: 9.447 / 9.485 / 9.469 t/s → mean 9.467, spread 0.4%
    - n=8 @ 500: 10.257 / 10.357 / 10.280 t/s → mean 10.298, spread 1.0%
    - Delta: **+8.8%**, well outside ~1.4% combined noise. Worst-case
      n=8 (10.257) beats best-case n=4 (9.485) by +8.1%.
  So n=8 IS the right setting for 12B service workloads. The "marginal"
  framing from the 60-token sweep was wrong — short benches over-credit
  n=4 because warmup effects matter more relative to a 60-token decode
  than a 500-token one. The reproducible n=6 dip seen at 60 tokens is
  not retested at 500 — likely still real (kernel-tile artifact) but
  irrelevant since n=8 is the recommendation. Updated 2026-06-15 from
  "+356% at n=4 (60-tok bench)" to the +315% n=8 figure at 500-tok
  sustained (10.298 vs 2.48 t/s non-MTP).

Updates to bench.sh and follow-on rows: stop pinning `--spec-draft-n-max 4`
by default. For new MTP rows, either omit the flag entirely (default = 3) or
pick the per-arch override above.

## Patch 10 (DONE 2026-06-15) — Gemma 12B `n_max` follow-up sweep

Swept `n_max ∈ {5, 6, 8}` on `unsloth-gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
with `mtp-gemma-4-12B-it.gguf` drafter, same flags as patch-9 12B run
(`-fa 1 --temp 0 -t 8 --no-mmap --no-spec-draft-backend-sampling -n 60`,
raw prompt "Write a short poem about a robot.").

Results:

| n_max | tg t/s | accept | drafted | accepted | per-cycle wall-clock |
|-------|--------|--------|---------|----------|---------------------|
| 4 (baseline rerun) | 11.31  | 96.2% | 52 | 50 | ~0.45 s |
| 5     | 11.40  | 94.5%  | 55      | 52       | ~0.50 s |
| 6     | 10.21  | 93.3%  | 60      | 56       | ~0.65 s ¹ |
| 6 (rerun) | 10.21 | 93.3% | 60      | 56       | confirmed |
| 8     | **11.67** | 90.6% | 64    | 58       | ~0.71 s |

¹ Reproducible dip at n=6. First run 10.189 t/s, rerun 10.213 t/s — not
transient memory pressure. The per-cycle wall-clock jumps from ~0.50 s
at n=5 to ~0.65 s at n=6, then back down (relatively) to ~0.71 s at
n=8. Looks like a kernel-tile / batch-alignment artifact at the
n_max+1=7 verify-batch boundary — possibly the FA SRAM tile size or
A100 IME2 matmul tile lines up cleanly for 5/8 but not for 6/7.
Worth a probe with `GGML_SPACEMIT_DISPATCH_LOG` to verify, but not on
the critical path.

**Initial conclusion (revised below)**: looked like n=8 was a marginal
+3.2% over n=4 — within ~3% of "run-to-run noise". Recommended n=8 but
flagged it as borderline.

**Variance check — revised conclusion (DONE 2026-06-15)**: the +3% felt
suspect (user instinct: "feels like this is actually a big change"),
so reran 3x n=4 and 3x n=8 at `-n 500` (steady-state regime, not
warmup-dominated).

| Run    | n=4 @ 500 | n=8 @ 500 |
|--------|-----------|-----------|
| 1      | 9.447 t/s | 10.257 t/s |
| 2      | 9.485 t/s | 10.357 t/s |
| 3      | 9.469 t/s | 10.280 t/s |
| **mean** | **9.467** | **10.298** |
| spread | 0.038 (0.4%) | 0.100 (1.0%) |

**Delta: +8.8%, far outside ~1.4% combined noise.** Worst-case n=8
(10.257) still beats best-case n=4 (9.485) by +8.1%. Rock-solid win,
not marginal at all.

Why the 60-token bench underestimated the win: short runs spend a
proportionally larger fraction in warmup / cold-cache state, where
the per-cycle MTP graph cost penalty is masked. At sustained
500-token decode the per-cycle cost difference between n=4's many
short cycles and n=8's fewer long cycles plays out properly —
n=8's higher n_drafted (456 vs 404 per 503 tokens) and only-slightly-
lower accept (98.7% vs 99.5%) gives 1+8·0.987 = 8.90 accepted/cycle
vs 1+4·0.995 = 4.98 accepted/cycle. n=8 needs ~57 cycles vs n=4's
~101 cycles to produce 500 tokens — so even though each n=8 cycle is
heavier, fewer total cycles wins.

Also note: absolute 500-tok tg (9.47 / 10.30) is LOWER than 60-tok tg
(11.31 / 11.67) for both. That's KV-cache-growth attention cost — the
12B model attending over 500+ tokens is materially slower per token
than at 60 tokens. The 500-token number is the realistic
service-throughput figure.

**Findings**:
- **n=8 IS the right setting for 12B service workloads at +8.8%
  over n=4.** Strong recommendation, not marginal.
- The reproducible n=6 dip at 60 tokens (10.21 vs 11.4 / 11.67) is
  not retested at 500 tokens — likely still a real kernel-tile
  artifact, but irrelevant given n=8 is the winner.
- RAM was not the cap — `common_params_fit_impl` reported 5653 MiB
  projected vs 15971 MiB total. Could push higher n_max if there
  were reason to expect more wins (probably tapped out around n=8
  given the per-cycle cost curve, but unmeasured).
- Updated main sweep table: 12B winner is n=8, vs-non-MTP figure
  revised from "+371% (60-tok)" to "+315% (500-tok sustained)".
- **Methodology lesson for future K3 benches**: 60-token benches
  underestimate MTP win on slow models. Use -n 500 (or higher) for
  any service-throughput claim. Reserve 60-token for fast iteration
  during code-change validation only.

No code change needed for the recommendation — `bench.sh` doesn't pin
12B-specific flags, so the user just passes `--spec-draft-n-max 8`
when running 12B.

## Patch queue (reorganized 2026-06-15)

The original §Patch 11 (E2B `eh_proj + hnorm` fusion) was dropped: its
premise was wrong on two counts. (a) `eh_proj` and `hnorm` are
Qwen 3.5 / GLM 4 / Exaone 4 / BailingMoE2 MTP tensors, not Gemma 4 —
the Gemma 4 Assistant graph (`src/models/gemma4-assistant.cpp`) uses
`nextn_proj_pre` + transformer block + `nextn_proj_post`, no
eh_proj/hnorm pair exists in that path. (b) E2B is no longer
net-negative once the n_max sweep flipped it to `n_max=2` at +6.9%
(see §`--spec-draft-n-max` tuning sweep table). The original entry
confused Qwen MTP structure with Gemma and described a problem that
already had a solution.

Upstream scan confirmed: `354ebac8c..upstream/master` (160 commits)
has no MTP-block fusion commit. Any fusion work below is original
investigation, not a cherry-pick.

Priority order is "expected perf delta, decreasing". Items below
the patch-15 refactor are deferred (X100) or out of scope (rebase,
tcm_sync_mem driver).

## Patch 11 (DONE, 2026-06-15) — `f_attention_scale` verified, comment added

Closed without sweep. Pre-sweep source inspection of
`src/models/gemma4.cpp:11` revealed a load-bearing comment on the
trunk Gemma 4 arch:

```cpp
hparams.f_attention_scale = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
```

i.e. the Gemma 4 family was *deliberately* trained with attention
scaling disabled — `1.0f` is not an unset-defaulted reference value,
it's the design value. The Assistant variant inherits the same
architecture (trunk + nextn predict heads), so the 1.0 in
`gemma4-assistant.cpp:12` is also correct.

Burning 4 builds × 3 runs at -n 500 (~1 hr) to confirm "the trunk
comment is right" is poor budget. Resolved by cherry-picking the
trunk comment forward so future readers don't redo the same
investigation:

```cpp
hparams.f_attention_scale = 1.0f; // see gemma4.cpp: Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
```

The remaining hypothesis ("Assistant head specifically *should* have
scaling even though trunk doesn't, because it's a separate trained
predictor") is not supported by anything in the architecture or in
the gguf metadata — the Assistant arch reuses the trunk attention
block verbatim and adds nextn_proj layers on top.

Status: cherry-picked; original §K3 improvements note about
f_attention_scale removed downstream when those notes get refactored.

## Patch 12 (DONE, 2026-06-15) — premise resolved, no perf benefit; flag dropped

Original premise: the bench flag `--no-spec-draft-backend-sampling`
was mandatory because enabling backend sampling on the draft would
trigger "backend sampling requires at most one output token per
sequence" assertion failures (bench.sh:58 comment). Goal was to fix
the underlying scheduler/sampler constraint and reclaim the ~94 ms
sampler cost.

Both halves of the premise broke under empirical testing:

**1. The assertion no longer fires.** Quick probe on Gemma 12B with
the flag REMOVED:

```
$BIN -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --model-draft mtp-gemma-4-12B-it.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -t 8 --no-mmap -fa 1 --temp 0 -n 30 \
  -p "Hello, "
```

Ran to completion. The check at `src/llama-context.cpp:1729-1750`
guards `has_samplers && batch_inp.logits`, and the MTP draft loop
in `common_speculative_impl_draft_mtp::draft()` (common/speculative.cpp:
702, 709, 783) only ever flags 1 output per seq per `llama_decode`
call. Somewhere along the recent upstream merges (likely the
c9a10c1d8 cherry-pick or earlier) the dft-side constraint stopped
triggering on the MTP path. The flag is a stale workaround.

**2. Enabling backend sampling is slightly net-NEGATIVE on K3.**
A/B at Gemma 12B n_max=4, -n 60:

| run | tg | accept | sampling time | total |
|---|---|---|---|---|
| backend sampling **OFF** | 11.321 t/s | 96.154% | 166.98 ms | 12,198 ms |
| backend sampling **ON**  | 10.855 t/s | 96.154% | 163.30 ms | 12,450 ms |

Bit-identical sampling decisions (accept rate identical, sampler
wall-clock identical within 2%). The +250 ms total cost comes from
the additional sampling tensors in the draft graph (~3-4 ms per
decode × 50+ decodes). The CPU sampler at `common/speculative.cpp:729`
still runs to populate candidates for `p_min` checking — so
attaching the backend chain doesn't elide CPU work, it just adds
parallel work.

**Achievable upper-bound win** if the CPU sampler call in draft()
were rewired to consume candidates from `sampling.probs` /
`sampling.candidates` instead: ~165 ms over 12.2 s decode = **+1.4%
wall-clock**, exactly at the patch-10 noise floor (1.4% combined).
Below the threshold worth burning code-change effort, especially
versus the 5-15% expected wins on patches 13/14.

Resolution:
- Drop `--no-spec-draft-backend-sampling` from `bench.sh:58` (now
  redundant — it's the default value for the param). Update the
  comment to reflect empirical state, not the stale premise.
- Leave the flag itself in `common/arg.cpp:3620-3627` (it's an
  upstream-provided control, not ours to remove).
- No code change to `common_speculative_impl_draft_mtp`.

Status: closed. K3 doesn't currently benefit from draft-side backend
sampling; revisit if backend sampler becomes cheaper-per-decode in
a future ggml release.

## Patch 13 (DONE 2026-06-15) — Gemma 4 RMS_NORM + MUL + ADD fusion, dismissed empirically

Path explored: inspection of `gemma4-assistant.cpp:161-164` identified
`attn_post_norm + residual add` (and three sibling pairs per layer) as
a clean RMSNorm+MUL → ADD triple. The pre-existing
`ggml_cpu_try_fuse_ops` already fuses RMS_NORM+MUL; extending that to
RMS_NORM+MUL+ADD as a third FUSE_OP enum variant was straightforward.

Implementation done and built (see git stash dropped during validation):
- `enum ggml_rms_norm_fuse_op` extended with `GGML_RMS_NORM_FUSE_OP_MUL_ADD`
- `ggml_compute_forward_rms_norm_mul_add_fused()` wrapper added
- 3-op detect inserted ahead of the 2-op detect in `ggml_cpu_try_fuse_ops`
- Inner loop: `y[i] = x[i] * scale * w[i] + r[i]` (single pass)

Validation — `-n 500 × 3` A/B (stash → rebuild → bench → pop → rebuild):

| Model | n_max | BEFORE tg | AFTER tg | Δ tg    | Trajectory |
|-------|-------|-----------|----------|---------|------------|
| E2B   | 2     | 12.426    | 11.167   | **−10.1%** | drifted: accept 31.5%→23.0%, n_drafted 92→74 |
| E4B   | 3     | 9.465     | 10.565   | **+11.6%** | drifted: accept 27.0%→34.2%, n_drafted 126→117 |
| 12B   | 4     | 9.171     | 9.178    | **+0.08%** | bit-identical: same accept %, n_drafted, n_accept |

Verdict: dismissed. 12B is the clean kernel A/B because its logit
margins are wide enough that the fused kernel's 1-ulp FMA folding
(`a*scale*w + r` collapses into one FMA, ≠ separate mul-then-add) does
not flip any argmax, so the trajectory is bit-stable. On that clean
A/B the kernel saves **+0.08%** — within session σ (~0.01-0.03 t/s),
indistinguishable from noise.

The E4B/E2B numbers were driven by FMA-induced trajectory drift, not
by kernel speedup: at temp=0 a 1-ulp logit shift can flip the greedy
argmax on close-margin tokens, cascading into a totally different
generation path. E4B got "lucky" trajectory (higher accept, faster);
E2B got "unlucky" (lower accept, slower). Real-world tg shifts but
isn't a function of the kernel — and the direction isn't predictable
per prompt/model. Not a robust win.

Why the kernel save is negligible on K3:
- The pre-existing 2-op RMS_NORM+MUL fusion already collapses the
  dominant cost (rms-scan + weight multiply in one pass).
- The marginal save from folding the residual ADD is one load+store
  per row of n_embd floats — bounded by memory bandwidth, which the
  2-op kernel was likely already saturating.
- ggml-sched dispatch overhead per node is sub-millisecond on this
  graph; folding two nodes into one saves the dispatch but the
  absolute saving is too small to register in -n 500 × 3 bench σ.

Action taken: code reverted (no commit to ggml/src/ggml-cpu/). TODO
section and `results.log` carry the empirical A/B rows for the
permanent record. Pattern matches patches 11 and 12: implementation
ready, bench data killed the premise, dismissed without committing.

Don't re-explore RMS_NORM+ADD or similar small-tensor residual
fusions on K3 unless a profiler-grade per-op timing tool shows the
residual ADD pass costing >5% of per-iter time — empirically it does
not (this exercise was that profile, by elimination).

## Patch 14 (DONE 2026-06-15) — `GGML_OP_TIMING` probe instrumentation; eh_proj+hnorm fusion target dismissed

Original target was a `RMSNorm + concat + matmul` fusion at
`qwen35.cpp:556-566` (the MTP block's `mtp_hnorm` / `mtp_enorm` ->
`mtp_concat` -> `mtp_eh_proj` chain). Patch 13's dismissal made the
right move clear: profile first, code second.

Pivoted to building the profiler itself — an env-gated, thread-0
per-named-tensor wall-clock recorder (`GGML_OP_TIMING=1` in
`ggml/src/ggml-cpu/ggml-cpu.c`). Mirrors the `GGML_SPACEMIT_DISPATCH_LOG`
pattern. Filters to `mtp_*` / `h_*` names by default (the MTP-block
region); `GGML_OP_TIMING_ALL=1` records every named tensor. atexit()
prints a sorted table.

Probe result — Qwen 3.5-4B MTP `-n 200 --temp 0`, n_max=3, 319 MTP
graph evaluations, 25.761 s decode:

| name              | op       | total_us | % MTP block | % decode |
|-------------------|----------|---------:|------------:|---------:|
| mtp_ffn_out       | MUL_MAT  | 286,255  | 40.16%      | 1.11%    |
| mtp_eh_proj       | MUL_MAT  | 184,456  | 25.88%      | 0.72%    |
| mtp_Qcur_full     | MUL_MAT  | 149,565  | 20.98%      | 0.58%    |
| mtp_attn_out      | MUL_MAT  |  77,343  | 10.85%      | 0.30%    |
| mtp_tok_embd      | GET_ROWS |   6,089  |  0.85%      | 0.024%   |
| **mtp_concat**    | CONCAT   |   3,769  |  0.53%      | 0.015%   |
| h_pre_norm        | ADD      |   2,095  |  0.29%      | 0.008%   |
| mtp_gate          | CONT     |   1,831  |  0.26%      | 0.007%   |
| mtp_attn_residual | ADD      |   1,355  |  0.19%      | 0.005%   |

MTP block total: 712,758 us = **2.77 % of decode wall-clock**.
The four IME2-accelerated matmuls account for 97.87 % of that block;
everything else (concat, residual adds, hnorm/enorm hidden inside the
already-fused 2-op RMS_NORM+MUL kernels) is below the noise floor.

Conclusion: dismissing the original `eh_proj+hnorm` fusion target.
Best-case kernel save (fold concat + materialise hnorm into eh_proj)
is bounded above by the sum of `mtp_concat` + h_pre_norm + small adds
= **<0.1 % of decode wall-clock**. Two orders of magnitude below the
1.4 % patch-10 noise floor; no commit-worthy delta achievable.

What stays: the `GGML_OP_TIMING` probe itself. Off by default, zero
overhead unless env is set. From now on, any future SpacemiT-fork
fusion patch must run this probe first and show ≥ 2 % of decode
wall-clock in the target region before coding.

Usage:
```
GGML_OP_TIMING=1 llama-speculative-simple ...  2>&1 | grep -A100 GGML_OP_TIMING
GGML_OP_TIMING=1 GGML_OP_TIMING_ALL=1 ...      # record every named tensor (large)
```

## Patch 15 (DEFERRED 2026-06-15) — unify `embd_pre_norm` / `embd_nextn` + fold `mtp_on_hybrid_qwen35`

Closed without implementation. Scope conflicts with preserving the
parallel pre_norm / nextn tap infrastructure for upcoming features and
external consumers of this fork.

Investigation surfaced that `llama_model_mtp_uses_nextn` currently
returns `true` for every arch (including `default:`), which means the
`embd_pre_norm` buffer, `cparams.embeddings_pre_norm{,_masked}` flags,
`llama_set_embeddings_pre_norm` setter, `get_embeddings_pre_norm{,_ith}`
getters, `t_h_pre_norm` plumbing, `common_speculative_need_embd_pre_norm`
wrapper, and the server-context `need_embd_pre_norm()` hook are all
reachable only behind a switch that never flips at runtime today.

Two refactor variants were considered:
  1. Unify both taps into a single `embd_drafter_h` buffer + single
     setter/getter pair (original TODO plan, ~150–300 LOC across 7
     files).
  2. Delete the pre_norm tap as dead code (smaller, subtractive).

Both were rejected: the parallel `_pre_norm` / `_nextn` plumbing is
scaffolding for upcoming MTP arches that will tap pre-output-norm and
for other consumers of this fork that may already rely on the C API
pair. Collapsing or deleting it forces those callers to re-plumb the
seam later, which is the opposite of what a refactor patch should do.

The third sub-item — folding the local `const bool mtp_on_hybrid_qwen35`
in `src/llama-model.cpp` into a per-arch `hparams` flag — was also
dropped. It's used twice in one function, cosmetic-only, net-zero LOC,
no perf or correctness payoff.

Outcome: the parallel pre_norm/nextn plumbing stays in place. If a
future arch flips `llama_model_mtp_uses_nextn` to `false`, the pre_norm
path is ready without further work. Revisit only if a concrete consumer
(internal or external) demonstrates that the dual API has a real cost.

## Patch 16 (DISMISSED 2026-06-15) — X100 sampling threadpool, fails its own probe gate

Closed without implementation after a zero-code data dip into existing
bench logs.

The patch entry's own pre-implementation gate was: probe shows ≥5%
tg uplift from moving sampling off the main coordinator thread onto
a 2-thread X100 pool. The hypothesis (~100 ms / ~5% upside) came from
absolute sampler cost, never normalized against decode wall-clock.

Data (results.log row 43, Gemma 12B MTP, n=60, --no-spec-draft-backend-sampling):
- `sampling time = 166.98 ms (104.69 ms samplers)`
- `total = 12197.90 ms`
- → sampling = **1.37% of decode** (broad), **0.86% of decode** (sampler chain only)

That's already at or below the patch-10 1.4% noise floor — *before*
subtracting any X100/A100 LPDDR contention overhead, scheduler
overhead, or imperfect overlap with the next-decode setup window.

The TODO's other X100 candidates (RoPE, norms, KV shuffle, embedding
lookup) are weaker still — smaller absolute time, same shared LPDDR
bottleneck. Sampling was the strongest candidate and it's already
below detectable headroom.

Conclusion: X100 cores are not a perf opportunity for MTP decode on
the K3 under the current A100 + IME2 + TCM configuration. The shipped
A100-only `cpu_mask: ff00` configuration is the right one. Closing
the `GGML_CPU_RISCV64_SPACEMIT_X100=ON` compile-flag idea entirely —
no plumbing to add, no flag to introduce.

Re-open only if a future workload makes sampling a meaningfully
larger fraction of decode (e.g., extremely small models where matmul
cost collapses, or sampler chains substantially heavier than greedy
+ top-k).

## X100 core utilization analysis (background for patch 16)

User decision (2026-06-15): not touching X100 cores until the patch
11-15 line ships. Any X100 work MUST be behind a compile flag —
non-default since the K3 backend has shipped as A100-only and the
A100-only path is the validated configuration. Flag name:
`GGML_CPU_RISCV64_SPACEMIT_X100=ON`, off by default. Patch 16 above
is the concrete deferred patch entry; this section is the supporting
analysis.

The user's prior intuition was "X100s probably aren't useful". This
section documents what could plausibly help, so the deferral is an
informed choice not a guess.

K3 layout reminder: cores 0–7 are X100 general-purpose RISC-V
application cores (RVV but **no IME2 matrix instructions**); cores
8–15 are A100 AI cores with IME2 + TCM. The SpacemiT backend pins all
8 compute workers to 8–15 today (`cpu_mask: ff00`,
`perfer_core_arch_id: a064`); only the unpinned main coordinator
thread floats on 0–7.

What X100 could plausibly pick up:

1. **Sampling** — softmax / top-k / top-p / repetition penalty are
   pure scalar+vector logits processing with **no matmul**. IME2
   gives X100 nothing here. Today the CPU sampler already runs on the
   unpinned main thread (which happens to land on X100), so on the
   non-backend-sampling path this is already X100 work — just on one
   core. A 2–4 way X100-only threadpool for sampling could overlap
   with the A100 next-decode setup. **Probably the highest-upside
   X100 candidate** because (a) zero IME2 dependency, (b) sampling is
   already known to be a non-trivial cost (~100 ms total on the patch
   6c bench), (c) it sits at the end of one decode and the start of
   the next, so it's a natural pipelining point.
2. **RoPE** — elementwise rotation, no matmul, no IME2 use. Could
   run on X100 if it were on the critical path; but it's typically
   fused with the attention matmul on the A100 side and isn't a
   measured bottleneck on the K3 profile. Low confidence in headroom.
3. **Norms (RMSNorm / LayerNorm)** — reduction + elementwise scale,
   again no matmul / no IME2 win. Same critical-path issue as RoPE.
4. **KV cache memory shuffling** — pure DRAM bandwidth, doesn't
   benefit from A100 compute. Could run on X100 while A100 does the
   next matmul. But: both core complexes share the same LPDDR
   controller, so the bandwidth is the bottleneck not the core. This
   is the bucket where intuition probably matches reality —
   parallelising doesn't help if the resource is bandwidth-bound.
5. **Token embedding / lookup** — gather op, memory-bound, same
   contention story as #4.

The structural blocker for any of this: ggml's CPU backend has **one
threadpool**, and parallelism is across threads of that pool for a
single op. To put op X on X100 cores and op Y on A100 cores in
parallel, you'd need either:
- (a) Two CPU sub-backends with their own threadpools and ggml-sched
  routing nodes to one or the other (large refactor, touches the
  scheduler).
- (b) A hybrid threadpool that can route specific ops to specific
  core subsets at submission time (moderate, but ggml's scheduler
  doesn't currently express "this op should use threads [0..n)").
- (c) Manual op-level pthread spawn for one or two specific ops
  (light but ugly, easy to regress).

Conclusion: the only X100 candidate likely to net a measured win is
**(1) sampling on a small X100 threadpool, overlapped with the next
decode's first ops**, and even that needs a careful pipelining
implementation in `tools/server` or `common/speculative` to overlap
correctly. The other candidates are either bandwidth-bound (4, 5)
or not on the critical path (2, 3). This is what patch 16 above
formalizes — see the patch entry for the probe-then-commit plan.

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

## Patch 17 (DISMISSED 2026-06-16) — trunk-graph probe; ROPE-RVV and Q4_1-HP-unlock both fail the decode gate

First-ever full trunk-graph profile (`GGML_OP_TIMING=1 GGML_OP_TIMING_ALL=1`),
captured on branch `platima-ime2-trunk-probe`. Prior probe work (patch 14)
only ever profiled the MTP block; the 97%-of-decode trunk graph had never
been measured. Model: Qwen3.5-2B-Q4_1, `llama-completion -t 8 --no-mmap
-fa 1 -n 200 --temp 0` (14.31 tok/s decode under the probe).

Decode wall-clock breakdown (aggregated by op, TOTAL_US 5,361,665):
- **MUL_MAT 76.5%** — already 100% on the IME2 fast path (every weight
  tensor logs `ime` under `GGML_SPACEMIT_DISPATCH_LOG=1`).
- GET_ROWS 4.9%, RMS_NORM 4.2% (RVV), UNARY 3.1% (SILU, RVV),
  GLU 1.6% (SwiGLU, RVV), FLASH_ATTN_EXT 1.0% (RVV).
- **ROPE 0.22%**, ADD 0.37%, MUL 0.19%.

Two candidate patches were gated on this data and both fail:

1. **ROPE RVV vectorization** (`rotate_pairs`, scalar in `ops.cpp`): at
   0.22% of decode it is ~9× under the 2% gate. Dismissed without coding.

2. **Q4_1 32×256 HP dispatch unlock** (the `// TODO` at `ime.cpp` ~1301;
   blocked by the `quant_b_zp != NULL` `GGML_ABORT` in
   `gemm_kernel_i8i4_hp_m1`). Decisive same-arch A/B: minted a Q4_0 by
   requantizing the Q4_1 (lossy quality, valid for a speed test), so the
   FFN/qkv tensors route to `q4_0_32x256_q8_0` (HP) vs the Q4_1
   `q4_1_32x32_q8_0`. Same size, same q8_0 activations.
   - Decode: **15.83 (HP) vs 15.98 (non-HP) tok/s** → 0%, within noise.
   - Prefill: **pp512 119.46 (HP) vs 106.67 (non-HP) t/s** → **+12%**.
   The HP tile only helps prefill (weight reuse across M≥4 columns); decode
   is M=1, weight-bandwidth-bound, so wider-K tiling reads each weight once
   either way and buys nothing. Getting the +12% prefill would require
   hand-writing zp support into the tuned m1 asm (fragile, silent-corruption
   risk) for zero decode benefit, on a quant type that isn't even in the
   README perf table. Fails the decode-focused gate. Dismissed.

Conclusion: the trunk graph holds no ≥2%-of-decode region that isn't already
IME2/RVV-optimal. Decode is bandwidth-bound GEMV (~17.6 GB/s on 1.1 GB
weights); no kernel/tile/dispatch lever moves it — only MTP (shipped) or
smaller weights do. The patch-14 probe earned its keep here by killing two
speculative patches before any asm was written. Branch carries this writeup
only, no code change.

Re-open the HP unlock only if a prefill-bound Q4_1 workload becomes important
*and* someone is willing to own the m1 zp asm.

## K3 A100 / X100 improvements observed during the merge

- **X100 cores are unused.** The current SpacemiT backend (`ggml-cpu/spacemit/`)
  targets the A100 AI cores only (`perfer_core_arch_id: a064`, `cpu_mask: ff00`,
  i.e. cores 8–15). The K3 also has X100 application cores (0–7) which are
  currently only used for the main thread. **→ patch 16** (deferred, compile-
  flag gated).
- **Pre-norm + nextn dual hidden-state extraction.** Two parallel buffers
  (`embd_pre_norm` sized `n_embd` for Qwen3.5, `embd_nextn` sized
  `n_embd_out` for Gemma4 via `nextn_proj_post`) with parallel flags /
  setters / getters. **→ patch 15 deferred** — preserved as scaffolding
  for upcoming MTP arches and external consumers of this fork.
- **`mtp_on_hybrid_qwen35` is fork-only.** Qwen3.5 MTP wiring lives in
  `src/llama-model.cpp` and isn't in upstream. **→ patch 15 deferred**
  (cosmetic refactor only, no perf payoff).
- **`deepstack_mapping_arr` is missing.** Granite4 Vision (upstream commit
  `64086f2b2`) added this field; a log-print referencing it was removed from
  `llama-model.cpp` during the cherry-pick. Tracking-only — restore at rebase
  time. No standalone patch (zero functional impact until a Granite4 Vision
  gguf is loaded).
- **`f_attention_scale` hard-coded to 1.0 for Gemma4 assistant.** Inherited
  from upstream's reference impl. We have Gemma 4 Assistant ggufs in
  `~/models` (the `mtp-gemma-4-{E2B,E4B,12B}-it.gguf` files) and have been
  running them since patch 7. **→ patch 11** (sweep alternative scale values
  for accept-rate impact, FIRST in the queue).
- **Cherry-pick base drift.** `version:` currently shows `9481 (161be67d6)`
  but `161be67d6` is a *local* cherry-pick tip, not an upstream commit. The
  custom `--version` line now also shows the upstream base
  (`354ebac8c`) and the SpacemiT release tag separately so this is no longer
  misleading. Resolved by version-stamp patch.

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

## Functional test (DONE 2026-06-15 — task #18)

End-to-end smoke test on K3 after the Gemma4 MTP + E2B/E4B-assistants
cherry-pick (`c4bbdafed` + `161be67d6`) settled into the SpacemiT branch.
All runs with `-fa 1 --temp 0 -t 8 --no-mmap`, `--spec-draft-n-max 3
--no-spec-draft-backend-sampling` on the MTP runs.

| Modality | Tool | Result |
|----------|------|--------|
| Gemma 4 E4B MTP (text) | `llama-speculative-simple` | tg 10.16 t/s, accept 26.7% (28/105), coherent poem |
| Qwen 3.5 4B MTP (text, chat-templated) | `llama-speculative-simple` | tg 9.09 t/s, accept 59.4% (41/69), coherent thinking-mode |
| Image (Gemma 4 E4B vision) | `llama-mtmd-cli --jinja --image` | coherent multimodal analysis of input image |
| Audio (Gemma 4 E4B mmproj) | `llama-mtmd-cli --jinja --audio` | coherent audio summary (input was `tools/mtmd/test-2.mp3`) |

Pass criteria all met:
- No `malloc(): invalid size (unsorted)` heap corruption on any run
  (the previous-attempt regression mode that motivated this checklist).
- `use_ime2: 1` and `tcm is available, blk_size: 393216, blk_num: 8`
  printed at startup on every run — A100 backend intact.
- `/dev/tcm_sync_mem` fallback log fired as expected and is harmless.
- MTP accept rates non-trivial on both archs (Gemma E4B 26.7%,
  Qwen 4B 59.4%) — the dispatch wiring from patches 4/5 is still alive.
- MTMD vision and audio projector paths both run end-to-end (Gemma
  needed `--jinja` for its chat template; this is a known mtmd-cli
  template-compatibility quirk, not a backend issue).

Tap-mismatch note for future me: I tried the Gemma image test without
`--jinja` first and got `this custom template is not supported, try
using --jinja` — that's the `common_chat_templates_apply` path
throwing for Gemma's custom template. Fix is the flag, not the model.
