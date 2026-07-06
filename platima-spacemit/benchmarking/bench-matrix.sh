#!/usr/bin/env bash
#
# bench_matrix.sh  -  four-quadrant FA x MTP benchmark, SpacemiT K3 (A100/IME2)
# ----------------------------------------------------------------------------
# Quadrants:
#   1. no MTP, no FA   -> Phase 1 llama-bench, fa=0
#   2. no MTP, FA      -> Phase 1 llama-bench, fa=1
#   3. MTP,    no FA   -> Phase 2 llama-cli, fa=0
#   4. MTP,    FA      -> Phase 2 llama-cli, fa=1
#
# llama-bench cannot do MTP, so MTP quadrants come from llama-cli (5x avg).
# Compare an MTP cell against the SAME model's no-MTP llama-bench cell at the same fa.
#
# Outputs (timestamped dir):
#   results_bench.tsv  build model fa pp_tps tg_tps command
#   results_mtp.tsv    build model fa mtp_mean mtp_min mtp_max accept command
#   table_tg_fa0.md / table_tg_fa1.md      no-MTP generation
#   table_pp_fa0.md / table_pp_fa1.md      no-MTP prefill
#   table_mtp_fa0.md / table_mtp_fa1.md    MTP generation (mean)
#
set -uo pipefail

# ---- Tunables --------------------------------------------------------------
PP=128; NG=128; THREADS=8; UB=128; MMP=0     # llama-bench shape
FA_STATES=(0 1)                               # quadrant FA dimension
MTP_NPRED=256; MTP_RUNS=5; MTP_NMAX=4
SLEEP_RUN=3        # settle between individual runs (let pages free, OOM pressure ease)
SLEEP_MODEL=300    # cooldown between models (thermal headroom on sustained all-core load)
PROMPT="Explain in technical detail how RISC-V vector extensions accelerate matrix multiplication in transformer inference, covering register width, fused multiply-add, and memory bandwidth."

MODELS_ROOT="/models"   # USB SSD, HF org/repo layout (moved off root fs 2026-07-03)
OUTDIR="$HOME/bench-matrix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
BENCH_TSV="$OUTDIR/results_bench.tsv"
MTP_TSV="$OUTDIR/results_mtp.tsv"

# ---- Builds: "LABEL|DIR" ----
BUILDS=(
  # upstream-b9628 dropped: vanilla upstream + SMT flags heap-crashes ('malloc(): invalid
  # size') on the prefill->generation transition across architectures (Gemma + Qwen3.5).
  # Reproduced on b9628 and a fresh build. The SpacemiT fork carries an un-upstreamed fix.
  # "upstream-b9628|$HOME/llama.cpp-b9628/build/bin"
  "bianbu-system|/usr/bin"
  "spacemit-0.1.3|$HOME/spacemit-llama.cpp.riscv64.0.1.3"
  "platima-llama-smt|$HOME/llama-smt/build/bin"
)
COL_HEADERS=("Bianbu system" "SpacemiT 0.1.3" "Platima llama-smt")

bin_for () { local dir="$1" tool="$2"
  if [ "$dir" = "/usr/bin" ]; then echo "/usr/bin/$tool"
  elif [ "$dir" = "$HOME/spacemit-llama.cpp.riscv64.0.1.3" ]; then echo "$dir/bin/$tool"
  else echo "$dir/$tool"; fi; }
lib_for () { local dir="$1"
  if [ "$dir" = "/usr/bin" ]; then echo "/usr/lib"
  elif [ "$dir" = "$HOME/spacemit-llama.cpp.riscv64.0.1.3" ]; then echo "$dir/lib"
  else echo "$dir"; fi; }

# ---- Phase 1 models (base): "LABEL|RELPATH" ----
BENCH_MODELS=(
  "Qwen3.5-0.8B-Q4_K_M|unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
  "Qwen3.5-2B-Q4_1|unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_1.gguf"
  "Qwen3.5-4B-Q4_K_M|unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
  "Qwen3.5-4B-UD-Q4_K_XL|unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"
  "Qwen3.5-9B-Q4_K_M|unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
  "gemma-4-E2B-q4_0-google|google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf"
  "gemma-4-E2B-qat-UD-Q4_K_XL|unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
  "gemma-4-E4B-q4_0-google|google/gemma-4-E4B-it-qat-q4_0-gguf/gemma-4-E4B_q4_0-it.gguf"
  "gemma-4-E4B-qat-UD-Q4_K_XL|unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
  "gemma-4-12b-Q4_K_M|unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf"
  "gemma-4-12B-qat-UD-Q4_K_XL|unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
)

# ---- Phase 2 MTP models: "LABEL|BASE_REL|DRAFTER_REL" (SELF = self-draft) ----
MTP_MODELS=(
  "Qwen3.5-2B-Q4_1-MTP|unsloth/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-Q4_1.gguf|SELF"
  "Qwen3.5-4B-Q4_K_M-MTP|unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-Q4_K_M.gguf|SELF"
  "Qwen3.5-9B-Q4_K_M-MTP|unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf|SELF"
  "gemma-4-E2B-qat-MTP|unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf|unsloth/gemma-4-E2B-it-qat-GGUF/mtp-gemma-4-E2B-it.gguf"
  "gemma-4-E4B-qat-MTP|unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf|unsloth/gemma-4-E4B-it-qat-GGUF/mtp-gemma-4-E4B-it.gguf"
  "gemma-4-12b-MTP|unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf|unsloth/gemma-4-12b-it-GGUF/mtp-gemma-4-12b-it.gguf"
  "gemma-4-12B-qat-MTP|unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf|unsloth/gemma-4-12B-it-qat-GGUF/mtp-gemma-4-12B-it.gguf"
  # Qwen3.6 = hybrid Gated-DeltaNet + MoE arch (self-MTP, PR #22673); 27B Q3_K_S ~12.5 GB, fits 16 GB with mmap
  "Qwen3.6-27B-Q3_K_S-MTP|unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q3_K_S.gguf|SELF"
)

# ============================ PHASE 1 (no-MTP, fa 0 and 1) ============================
printf 'build\tmodel\tfa\tpp_tps\ttg_tps\tcommand\n' > "$BENCH_TSV"
bench_extract () { local f="$1" kind="$2" pat
  [ "$kind" = "pp" ] && pat="pp${PP}" || pat="tg${NG}"
  grep -E "\| *${pat} *\|" "$f" | grep -oE '[0-9]+\.[0-9]+ . ' | grep -oE '^[0-9]+\.[0-9]+' | tail -1; }

echo "######## PHASE 1: llama-bench, no MTP, fa 0 and fa 1 ########"
for b in "${BUILDS[@]}"; do
  IFS='|' read -r blabel bdir <<< "$b"
  bin=$(bin_for "$bdir" llama-bench); lib=$(lib_for "$bdir")
  [ -x "$bin" ] || { echo "!! skip $blabel (no llama-bench)"; continue; }
  echo "===== $blabel ====="
  for m in "${BENCH_MODELS[@]}"; do
    IFS='|' read -r mlabel mrel <<< "$m"; mpath="$MODELS_ROOT/$mrel"
    if [ ! -f "$mpath" ]; then
      for fa in "${FA_STATES[@]}"; do printf '%s\t%s\t%s\tMISSING\tMISSING\t-\n' "$blabel" "$mlabel" "$fa" >> "$BENCH_TSV"; done
      echo "  !! missing $mpath"; continue
    fi
    for fa in "${FA_STATES[@]}"; do
      log="$OUTDIR/p1_${blabel}__${mlabel}__fa${fa}.log"
      cmd="LD_LIBRARY_PATH=$lib $bin -m $mpath -t $THREADS -p $PP -n $NG -mmp $MMP -fa $fa -ub $UB"
      echo "  >>> $mlabel fa=$fa"
      LD_LIBRARY_PATH="$lib" "$bin" -m "$mpath" -t "$THREADS" -p "$PP" -n "$NG" -mmp "$MMP" -fa "$fa" -ub "$UB" > "$log" 2>&1
      if [ $? -ne 0 ]; then pp="FAIL"; tg="FAIL"; echo "      FAILED - see $log"
      else pp=$(bench_extract "$log" pp); tg=$(bench_extract "$log" tg); [ -z "$pp" ] && pp="?"; [ -z "$tg" ] && tg="?"; echo "      pp=$pp tg=$tg"; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$blabel" "$mlabel" "$fa" "$pp" "$tg" "$cmd" >> "$BENCH_TSV"
      sleep "$SLEEP_RUN"
    done
    echo "  ... cooldown ${SLEEP_MODEL}s"; sleep "$SLEEP_MODEL"
  done
done

# ============================ PHASE 2 (MTP, fa 0 and 1) ============================
printf 'build\tmodel\tfa\tmtp_mean_tps\tmtp_min\tmtp_max\taccept_pct\tcommand\n' > "$MTP_TSV"

echo "######## PHASE 2: llama-cli + MTP, fa 0 and fa 1, ${MTP_RUNS}x${MTP_NPRED} tok ########"
for b in "${BUILDS[@]}"; do
  IFS='|' read -r blabel bdir <<< "$b"
  bin=$(bin_for "$bdir" llama-cli); lib=$(lib_for "$bdir")
  [ -x "$bin" ] || { echo "!! skip $blabel (no llama-cli)"; continue; }
  echo "===== $blabel ====="
  for m in "${MTP_MODELS[@]}"; do
    IFS='|' read -r mlabel mrel drel <<< "$m"; mpath="$MODELS_ROOT/$mrel"
    if [ ! -f "$mpath" ]; then
      for fa in "${FA_STATES[@]}"; do printf '%s\t%s\t%s\tMISSING\t-\t-\t-\t-\n' "$blabel" "$mlabel" "$fa" >> "$MTP_TSV"; done
      echo "  !! missing base $mpath"; continue
    fi
    if [ "$drel" = "SELF" ]; then draft_args="--spec-type draft-mtp --spec-draft-n-max $MTP_NMAX"
    else
      dpath="$MODELS_ROOT/$drel"
      if [ ! -f "$dpath" ]; then
        for fa in "${FA_STATES[@]}"; do printf '%s\t%s\t%s\tMISSING\t-\t-\t-\t-\n' "$blabel" "$mlabel" "$fa" >> "$MTP_TSV"; done
        echo "  !! missing drafter $dpath"; continue
      fi
      draft_args="--model-draft $dpath --spec-type draft-mtp --spec-draft-n-max $MTP_NMAX"
    fi
    for fa in "${FA_STATES[@]}"; do
      cmd="LD_LIBRARY_PATH=$lib $bin -m $mpath $draft_args -t $THREADS --no-mmap -fa $fa -n $MTP_NPRED --temp 0 -st --simple-io -p \"...\""
      echo "  >>> $mlabel fa=$fa ($MTP_RUNS passes)"
      rates=(); acc=""; fail=0
      for i in $(seq 1 "$MTP_RUNS"); do
        log="$OUTDIR/p2_${blabel}__${mlabel}__fa${fa}__run${i}.log"
        # shellcheck disable=SC2086
        LD_LIBRARY_PATH="$lib" "$bin" -m "$mpath" $draft_args -t "$THREADS" --no-mmap -fa "$fa" \
            -n "$MTP_NPRED" --temp 0 -st --simple-io -p "$PROMPT" > "$log" 2>&1
        if [ $? -ne 0 ]; then echo "      run $i FAILED - see $log"; fail=1; break; fi
        # tg t/s: prefer the perf "eval time" line; fall back to the interactive
        # "[ ... Generation: X t/s ]" summary that llama-cli prints by default.
        r=$(grep -E 'eval time =' "$log" | grep -vi 'prompt' | grep -oE '[0-9]+\.[0-9]+ tokens per second' | grep -oE '^[0-9]+\.[0-9]+' | tail -1)
        [ -z "$r" ] && r=$(grep -oE 'Generation:[[:space:]]*[0-9]+\.[0-9]+' "$log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        [ -n "$r" ] && rates+=("$r")
        # acceptance (only present under -v): match a percent OR a 0..1 ratio
        a=$(grep -iE 'accept' "$log" | grep -oE '[0-9]+\.[0-9]+%|[01]\.[0-9]+' | tail -1); [ -n "$a" ] && acc="$a"
        sleep "$SLEEP_RUN"
      done
      if [ $fail -eq 1 ] || [ ${#rates[@]} -eq 0 ]; then
        printf '%s\t%s\t%s\tFAIL\t-\t-\t%s\t%s\n' "$blabel" "$mlabel" "$fa" "${acc:--}" "$cmd" >> "$MTP_TSV"; echo "      mean=FAIL"
      else
        read -r mean mn mx <<< "$(printf '%s\n' "${rates[@]}" | awk '{s+=$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1} END{printf "%.2f %.2f %.2f", s/NR, mn, mx}')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$blabel" "$mlabel" "$fa" "$mean" "$mn" "$mx" "${acc:--}" "$cmd" >> "$MTP_TSV"
        echo "      mean=$mean (min=$mn max=$mx) accept=${acc:--}"
      fi
    done
    echo "  ... cooldown ${SLEEP_MODEL}s"; sleep "$SLEEP_MODEL"
  done
done

# ============================ PIVOT TABLES ============================
pivot () { # $1=tsv $2=fa $3=value_col $4=models_array $5=outfile $6=title
  local tsv="$1" fa="$2" col="$3" arrname="$4" out="$5" title="$6"
  local -n marr="$arrname"
  {
    echo "## $title"; echo
    printf '| Model |'; for h in "${COL_HEADERS[@]}"; do printf ' %s |' "$h"; done
    printf '\n|---|'; for _ in "${COL_HEADERS[@]}"; do printf '---|'; done; printf '\n'
    for m in "${marr[@]}"; do
      IFS='|' read -r mlabel _ <<< "$m"; printf '| %s |' "$mlabel"
      for b in "${BUILDS[@]}"; do
        IFS='|' read -r blabel _ <<< "$b"
        v=$(awk -F'\t' -v b="$blabel" -v m="$mlabel" -v f="$fa" -v c="$col" '$1==b && $2==m && $3==f {print $c}' "$tsv" | tail -1)
        [ -z "$v" ] && v="-"; printf ' %s |' "$v"
      done; printf '\n'
    done
  } > "$out"
}

pivot "$BENCH_TSV" 0 4 BENCH_MODELS "$OUTDIR/table_pp_fa0.md" "Prefill pp${PP}, no MTP, FA off (t/s)"
pivot "$BENCH_TSV" 1 4 BENCH_MODELS "$OUTDIR/table_pp_fa1.md" "Prefill pp${PP}, no MTP, FA on (t/s)"
pivot "$BENCH_TSV" 0 5 BENCH_MODELS "$OUTDIR/table_tg_fa0.md" "Generation tg${NG}, no MTP, FA off (t/s)"
pivot "$BENCH_TSV" 1 5 BENCH_MODELS "$OUTDIR/table_tg_fa1.md" "Generation tg${NG}, no MTP, FA on (t/s)"
pivot "$MTP_TSV"   0 4 MTP_MODELS   "$OUTDIR/table_mtp_fa0.md" "MTP gen mean (${MTP_RUNS}x${MTP_NPRED}), FA off (t/s)"
pivot "$MTP_TSV"   1 4 MTP_MODELS   "$OUTDIR/table_mtp_fa1.md" "MTP gen mean (${MTP_RUNS}x${MTP_NPRED}), FA on (t/s)"

echo; echo "=== Done. Output dir: $OUTDIR ==="
for t in table_tg_fa1 table_tg_fa0 table_mtp_fa1 table_mtp_fa0 table_pp_fa1 table_pp_fa0; do
  echo; echo "----- $t.md -----"; cat "$OUTDIR/$t.md"
done
echo; echo "Raw TSVs (full commands in last column): $BENCH_TSV  and  $MTP_TSV"