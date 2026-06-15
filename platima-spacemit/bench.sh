#!/usr/bin/env bash
#
# mtp_fa_smoketest.sh
# -------------------
# Fast iteration benchmark for the SpacemiT K3 (RISC-V, A100 cores w/ IME2).
#
# Purpose:
#   Tiny ~550MB model used as a quick build-test-eval loop while merging and
#   tuning the SpacemiT llama.cpp fork. Runs a 4-way matrix to isolate the
#   contribution of Flash Attention (FA) and Multi-Token Prediction (MTP):
#     1. FA on,  no MTP   (baseline)
#     2. FA off, no MTP   (isolates the cost of dropping FA, which MTP historically required)
#     3. FA off, MTP      (the MTP option as documented)
#     4. FA on,  MTP      (does the newly merged code allow both? best case if so)
#
#   The mmproj (vision/audio projector) is intentionally NOT loaded here: it has
#   no effect on text throughput. It only matters for actual image/audio input.
#
# Watch for in configs 3 and 4:
#   - "eval time" perf line -> generation tokens/sec (tg)
#   - a draft acceptance % -> confirms MTP is engaging, not silently falling back.
#     0% acceptance or a flag error means MTP is not wired up correctly.
#
# A100 vs X100 core verification:
#   Each run is wrapped in a background poller that samples per-thread last-CPU
#   ('psr' column from ps -L) twice per second for the first ~3 seconds of
#   active compute, then prints a histogram. K3 layout: cores 0-7 are X100
#   (general), cores 8-15 are A100 (AI). With THREADS=8 we expect every
#   worker thread on cores 8-15. Any sample on 0-7 means a thread leaked onto
#   X100 and IME2 acceleration was wasted there - a silent perf bug.

set -uo pipefail

# ---- Variables (adjust paths to match your layout) -------------------------
# IMPORTANT: llama-completion does NOT drive the speculative loop - it parses
# --spec-type but never calls common_speculative_draft/_accept. With
# --spec-type draft-mtp it silently sets cparams.n_rs_seq=4 which widens the
# DeltaNet recurrent state 5x and tanks throughput for zero benefit. The
# correct tool for an end-to-end MTP run is llama-speculative-simple (which
# loads both target + drafter and runs the spec verify/accept loop). The
# server (tools/server) is the other tool that drives spec.
BIN="$HOME/llama-smt/build/bin/llama-completion"          # configs 1 + 2 (no MTP)
BIN_SPEC="$HOME/llama-smt/build/bin/llama-speculative-simple"  # configs 3 + 4 (MTP)
export LD_LIBRARY_PATH="$HOME/llama-smt/build/bin"
MODEL="$HOME/models/unsloth-Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"   # MTP-headed base, self-drafts
THREADS=8
NPRED=200
PROMPT="Explain how RISC-V vector extensions speed up matrix multiplication."

# MTP flags. --spec-type draft-mtp is the post-2026-05-13 flag name (renamed from 'mtp').
# No --model-draft: the MTP head is part of the target model. patch 2 teaches
# llama-speculative-simple to create the MTP draft context against the target,
# avoiding a duplicate model load that on a 4B model blew past the K3's 16 GB RAM
# during warmup (~15 GB observed). Mirror of tools/server/server-context.cpp.
# Backend sampling is left at default (off). The original 'mandatory --no-' rationale
# was stale — patch 12 (2026-06-15) found the assertion no longer fires AND that
# enabling backend sampling on draft is net-negative on K3 (-4.1% tg, +250 ms total)
# because the CPU sampler in common_speculative_impl_draft_mtp::draft() still runs
# to populate candidates for p_min. Re-add --no-spec-draft-backend-sampling only if
# you see the "backend sampling requires at most one output per sequence" error.
MTP="--spec-type draft-mtp --spec-draft-n-max 4"

# ---- A100 core verification helper -----------------------------------------
# Samples per-thread last-CPU for $pid every 0.5s for ~3s, then prints a
# histogram of where time was spent. Exits cleanly when the parent finishes.
#
# Layout reminder (K3): 0-7 = X100, 8-15 = A100. We want EVERY worker on 8-15.
poll_affinity() {
    local pid="$1"
    local samples_file
    samples_file="$(mktemp)"
    local i=0
    while [[ $i -lt 6 ]] && kill -0 "$pid" 2>/dev/null; do
        ps -L -o psr= -p "$pid" 2>/dev/null >> "$samples_file"
        sleep 0.5
        i=$((i + 1))
    done
    if [[ -s "$samples_file" ]]; then
        echo "--- per-thread CPU histogram (A100 = 8-15, X100 = 0-7) ---"
        # SpacemiT backend pins 8 compute workers to cores 8-15 (A100). The main
        # coordinator thread is intentionally NOT pinned and floats on X100.
        # An X100 *core* with <= ceil(samples/threads) hits is the coordinator
        # and is fine. An X100 core with sustained hits would be a leaked
        # compute thread (real perf bug).
        awk -v threads=8 '
            { gsub(/[[:space:]]+/, ""); if ($0 != "") cpu[$0]++; total++ }
            END {
                if (total == 0) { print "no samples"; exit }
                # samples per thread per run = total / (threads + 1 coordinator)
                per_thread = total / (threads + 1)
                a100 = 0; x100_total = 0
                for (c in cpu) {
                    if (c+0 >= 8) a100 += cpu[c]
                    else x100_total += cpu[c]
                }
                for (c = 0; c < 16; c++) {
                    if (c in cpu) {
                        if (c >= 8) tag = "A100 worker"
                        else tag = "X100"
                        printf "  cpu %2d (%-12s): %d samples\n", c, tag, cpu[c]
                    }
                }
                # one floating coordinator thread = ~per_thread total X100
                # samples across however many X100 cores it visited. Allow
                # 1.5x slack. Anything beyond that = a compute worker leaked.
                threshold = per_thread * 1.5
                printf "  ---> A100 workers: %d   X100 total: %d   (one coordinator ~= %d samples)\n", \
                    a100, x100_total, int(per_thread + 0.5)
                if (x100_total > threshold)
                    printf "  WARNING: X100 total %d > %d - a compute worker may have leaked off A100\n", \
                        x100_total, int(threshold + 0.5)
                else
                    printf "  OK: all compute on A100, only the main coordinator on X100\n"
            }
        ' "$samples_file"
    fi
    rm -f "$samples_file"
}

run_with_check() {
    local label="$1"; shift
    echo "=== $label ==="
    # start the bench in background so we can poll /proc on it
    ( "$@" ) &
    local bench_pid=$!
    # let the model load before sampling - load is single-threaded and dominates
    # the first ~10s; sampling earlier misses the actual compute phase
    sleep 12 &
    local sleep_pid=$!
    wait -n "$bench_pid" "$sleep_pid" 2>/dev/null || true
    if kill -0 "$bench_pid" 2>/dev/null; then
        poll_affinity "$bench_pid"
    fi
    wait "$bench_pid" 2>/dev/null
}

# ---- Runs ------------------------------------------------------------------
START_TS="$(date +%H:%M:%S)"
echo ">>> bench start $START_TS  (THREADS=$THREADS, NPRED=$NPRED)"
echo ">>> binary: $BIN"
echo ">>> LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo

run_with_check "1. FA on, no MTP (baseline)" \
    "$BIN" -m "$MODEL" -t "$THREADS" --no-mmap -fa 1 -n "$NPRED" --temp 0 -no-cnv -p "$PROMPT"

run_with_check "2. FA off, no MTP (FA cost isolation)" \
    "$BIN" -m "$MODEL" -t "$THREADS" --no-mmap -fa 0 -n "$NPRED" --temp 0 -no-cnv -p "$PROMPT"

# shellcheck disable=SC2086
run_with_check "3. FA off, MTP" \
    "$BIN_SPEC" -m "$MODEL" $MTP -t "$THREADS" --no-mmap -fa 0 -n "$NPRED" --temp 0 -p "$PROMPT"

# shellcheck disable=SC2086
run_with_check "4. FA on, MTP (does the new build allow both?)" \
    "$BIN_SPEC" -m "$MODEL" $MTP -t "$THREADS" --no-mmap -fa 1 -n "$NPRED" --temp 0 -p "$PROMPT"

# ----------------------------------------------------------------------------
# RESULTS LOG TEMPLATE
# ----------------------------------------------------------------------------
# Append one pipe-delimited line per run to results.log. Fields:
#   datestamp | buildnumber | modelpath | tg128results | pg128results | testfeatures | commandline | notes
#
# Schema notes:
#   datestamp     YYYY-MM-DD:HH:MM
#   buildnumber   git describe / commit hash, plus any +patchN applied
#   modelpath     absolute path to the GGUF under test
#   tg128results  generation t/s (token generation)
#   pg128results  prefill t/s (prompt processing)
#   testfeatures  which capabilities were active, e.g. MTP+IME2+FA, MMPROJ, TCM
#   commandline   the exact invocation used (for reproducibility)
#   notes         what was being tested / changed vs the previous run
#
# ----------------------------------------------------------------------------
# WORKED EXAMPLES (real results from the 2026-06 bring-up session)
# These double as a knowledge base. They record what this hardware actually
# does, which quant types accelerate, and which configs fail. Format matches
# the schema above. tg/pg are llama-bench numbers unless noted.
# ----------------------------------------------------------------------------
#
# --- Baseline / template row ---
# 2026-06-13:23:24|commit123blah+patch1|/home/owner/testfiles/model.gguf|4t/s|8t/s|MTP+IME2|LD_LIBRARY_PATH=/home/owner/llama-smt/build/bin /home/owner/llama-smt/build/bin/llama-bench -m /home/owner/testfiles/model.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Base run, no potential improvements attempted
#
# --- The library-path lesson (single most important finding) ---
# The Debian system libggml shadows the SpacemiT one. If LD_LIBRARY_PATH does
# not point at the SpacemiT libs first, the backend init messages still print
# but compute silently falls back to the X100 general cores at ~3x lower tg and
# up to ~17x lower pg. Always confirm via a known model's numbers.
# 2026-06-07:10:00|v0.1.1(X100 fallback)|Qwen3-0.6B-Q4_0.gguf|17.16t/s|28.75t/s|RVV-only-X100|llama-bench -m Qwen3-0.6B-Q4_0.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|WRONG libggml loaded, fell back to X100. Note the dismal pg.
# 2026-06-07:10:05|v0.1.1(A100)|Qwen3-0.6B-Q4_0.gguf|53.23t/s|499.34t/s|IME2-A100|LD_LIBRARY_PATH=<spacemit>/lib llama-bench -m Qwen3-0.6B-Q4_0.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Correct libs. Same binary, ~3x tg / ~17x pg vs X100 fallback.
#
# --- Dense Gemma 4 scaling on A100 (image+audio capable) ---
# 2026-06-07:11:00|v0.1.3|gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf|12.89t/s|125.47t/s|IME2-A100-QAT|llama-bench -m gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Fastest multimodal option, lowest quality of the Gemma line.
# 2026-06-07:11:05|v0.1.3|gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf|7.67t/s|58.37t/s|IME2-A100-QAT|llama-bench -m gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Best speed/quality balance for multimodal. QAT beats plain Q4_0 on size+speed.
# 2026-06-07:11:10|v0.1.3|gemma-4-12B-it-qat-UD-Q4_K_XL.gguf|3.53t/s|25.62t/s|IME2-A100-QAT|llama-bench -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Good quality, slow. Fits as a service in 16GB.
# 2026-06-07:11:15|v0.1.3|gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf|8.52t/s|54.11t/s|IME2-A100-QAT-MoE|llama-bench -m gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Benchmarks well BUT 14GB weights OOM as a llama-server (KV cache + prompt cache push past 16GB). Bench != service.
#
# --- Qwen3 MoE (text only) ---
# 2026-06-07:14:00|v0.1.3|Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf|13.15t/s|46.37t/s|IME2-A100-MoE|llama-bench -m Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Best text-only speed/quality, but 13.7GB leaves <2GB headroom; OOMs as a service unless -c capped low and --cache-ram 0.
#
# --- Qwen3.5 hybrid (Gated DeltaNet) - the architecture trap ---
# DeltaNet linear-attention layers do not get full A100 acceleration; tg barely
# scales with size and stays low regardless. Quality-per-param is excellent but
# this hardware+backend underserves it (pending DeltaNet kernels from SpacemiT).
# 2026-06-13:16:00|v0.1.3|Qwen3.5-4B-Q4_K_M.gguf|7.03t/s|37.14t/s|IME2-A100-DeltaNet|llama-bench -m Qwen3.5-4B-Q4_K_M.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Haiku-class quality, full multimodal incl video, but DeltaNet caps speed.
# 2026-06-13:16:05|v0.1.3|Qwen3.5-9B-Q4_K_M.gguf|4.25t/s|23.88t/s|IME2-A100-DeltaNet|llama-bench -m Qwen3.5-9B-Q4_K_M.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Higher quality, slower. DeltaNet bottleneck again.
#
# --- DOCUMENTED FAILURES (do not waste time re-testing these) ---
# UD-XL mixed-precision + DeltaNet = catastrophic. The per-layer precision
# switching defeats the A100 kernels entirely.
# 2026-06-13:16:30|v0.1.3|Qwen3.5-4B-UD-Q4_K_XL.gguf|4.43t/s|9.24t/s|IME2-A100-DeltaNet-UDXL|llama-bench -m Qwen3.5-4B-UD-Q4_K_XL.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|pg collapses to 9 vs 37 for plain Q4_K_M. Always use standard quant types for Qwen3.5.
# 2026-06-07:12:00|v0.1.3|Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf|1.09t/s|1.21t/s|A100-UDXL-FAIL|llama-bench -m Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|Q2_K_XL mixed precision is incompatible with the A100 path. Slower on A100 than X100. Abandoned.
# 2026-06-13:18:00|self-built b9616 (raw mainline + SMT flags)|Qwen3.5-4B-Q4_K_M.gguf|CRASH|35.72t/s|IME2-A100-HEAPBUG|llama-bench -m Qwen3.5-4B-Q4_K_M.gguf -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128|pg128 completed then "malloc(): invalid size (unsorted)". Newer mainline decode path conflicts with SMT heap-fallback barrier (tcm_sync_mem missing). Use the fork base, not raw mainline.
#
# --- Quant types that DO accelerate on A100 (per SpacemiT build doc) ---
#   Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0
# IQ* and UD-IQ* (imatrix) types are NOT A100-accelerated -> avoid them.
#
# --- TODO targets to log against this baseline (see TODO.md) ---
#   - Gemma 4 E2B/E4B MTP once the merge builds (configs 3 and 4 above)
#   - Qwen3.5-4B MTP via Qwen3.5-4B-Q4_K_M-MTP.gguf drafter
#   - FA-on + MTP together (config 4) if the merged code lifts the FA-off rule
#   - DeltaNet A100 kernels if/when SpacemiT ships them
#   - TCM init_barrier via real /dev/tcm_sync_mem once the driver lands
# ----------------------------------------------------------------------------
