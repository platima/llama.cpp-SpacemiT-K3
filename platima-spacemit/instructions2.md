## Current state

The Gemma 4 MTP merge built successfully on the SpacemiT fork. Confirmed working:
`llama-completion` loads, runs on the A100 cores (`use_ime2: 1`), no heap crash, and the
0.8B baseline (FA on, no MTP) measured tg 26.79 t/s / pp 111.20 t/s. MTP itself
(`--spec-type draft-mtp`) has NOT yet been verified to engage. That is the first job below.

## Benchmark harness

There is a benchmark script at `~/llama-smt/bench.sh` (the edited `mtp_fa_smoketest.sh`). It
runs a 4-way FA/MTP matrix against the tiny Qwen3.5-0.8B model for fast build-test-eval cycles.

Key facts about the files (already verified, do not re-derive):

- Base model under test: `~/qwen/Qwen3.5-0.8B-Q4_K_M-MTP.gguf` (525M). This is the MTP-headed
  base; the MTP draft head is baked into the file, so it self-drafts. There is NO separate
  small drafter file and NO plain non-MTP 0.8B base in `~/qwen/`.
- For the MTP configs, try `--spec-type draft-mtp --spec-draft-n-max 4` WITHOUT `--model-draft`
  first (self-contained head). Only if the build demands a separate draft model, point
  `--model-draft` at the same file.
- Binary: `~/llama-smt/build/bin/llama-completion`
- Libs: `export LD_LIBRARY_PATH=~/llama-smt/build/bin`

## What to do

1. Verify MTP actually engages. Run the MTP configs in `bench.sh` to completion and confirm:
   - no `invalid argument: --spec-type` error
   - a draft acceptance percentage appears in the output
   - generation t/s beats the 26.79 baseline
   If `--spec-type draft-mtp` errors, investigate whether the flag is wired into
   `llama-completion` in this build or only into another binary, and fix the invocation.

2. Use `bench.sh` as the standing smoke test after every build. After each run, append one
   pipe-delimited line per config to `~/llama-smt/results.log` using the schema documented at
   the bottom of the script:

   ```
   datestamp | buildnumber | modelpath | tg128results | pg128results | testfeatures | commandline | notes
   ```

   - `datestamp`  YYYY-MM-DD:HH:MM
   - `buildnumber`  `git describe` plus any `+patchN` you have applied, so every result is
     traceable to an exact code state
   - read tg off the `eval time` perf line, pp off the `prompt eval time` line
   - record the draft acceptance % in `notes` for MTP runs
   - keep the existing worked-example rows in the script header as reference; do not overwrite them

3. Seed `results.log` with tonight's confirmed baseline as the reference row:
   ```
   2026-06-13:23:50|<git describe of merged build>|/home/owner/qwen/Qwen3.5-0.8B-Q4_K_M-MTP.gguf|26.79t/s|111.20t/s|IME2-A100-FA|<full command from bench.sh config 1>|Post-merge baseline, FA on no MTP, no heap crash, use_ime2=1 confirmed
   ```

## Add a TODO.md

Create or append to `~/llama-smt/TODO.md` capturing the next phase: systematically testing
potential performance improvements and further upstream merges against this benchmark, logging
every attempt in `results.log` so both regressions and gains are visible. Candidate items:

- Confirm and tune MTP: sweep `--spec-draft-n-max` (try 2, 4, 6, 8) for best acceptance vs overhead
- Test whether FA-on + MTP works together (the old README said MTP required FA off; the newer
  merged code may lift that restriction). If both work, that is the best case.
- Run the full model matrix with MTP: Gemma 4 E2B, E4B, and Qwen3.5-4B, each against their
  MTP-headed bases, logged against the existing non-MTP baselines in the script header
- Additional upstream MTP-related commits not yet cherry-picked (e.g. e95dae18d D2D copy fix,
  6c4cbdc70 MTP kv-cache ctk fix) - evaluate relevance to CPU/A100 and merge if they help
- ggml kernel / IME2 tuning opportunities
- TCM `init_barrier` via a real `/dev/tcm_sync_mem` if/when SpacemiT ships the driver (currently
  falls back to heap; this is expected and must remain safe, do not "fix" the errno=2 log)

Keep the post-merge baseline as the reference row so every later change is measured against it.

## Guardrails (unchanged from the merge brief)

- Do not break the SpacemiT A100 / IME2 / TCM backend or the existing Qwen3.5 MTP code.
- The `/dev/tcm_sync_mem errno=2 ... falling back to heap` log is expected on current hardware
  and must not be "fixed". Keep the heap fallback safe.
- Quant types that accelerate on A100: Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0.
  IQ* and UD-IQ* (imatrix) types do NOT accelerate; do not benchmark them chasing speed.
- Commit working states so there are checkpoints to roll back to.
