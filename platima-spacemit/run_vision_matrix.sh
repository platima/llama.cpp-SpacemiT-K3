#!/usr/bin/env bash
# Vision auto-gating test matrix. Runs llama-mtmd-cli with NO tier env vars so the
# auto-detect path decides (Tier-2/IME2 on K3 for bf16 mmproj; no-op for f16 Qwen).
set -u
cd /home/owner/llama-smt
export LD_LIBRARY_PATH="$PWD/build/bin"
BIN="$PWD/build/bin/llama-mtmd-cli"
OUT=/tmp/vmatrix
mkdir -p "$OUT"

run() {
  local tag="$1" model="$2" mmproj="$3" img="$4"; shift 4
  local log="$OUT/$tag.log"
  echo "### $tag"
  GGML_SPACEMIT_DISPATCH_LOG=1 "$BIN" \
    -m "$model" --mmproj "$mmproj" --image "$img" \
    -t 8 --threads-batch 8 --jinja -n 400 --temp 0 -v \
    -p "Describe this image." "$@" > "$log" 2>&1
  local enc vime
  enc=$(grep -oiE 'image slice encoded in [0-9]+ ms' "$log" | tail -1)
  vime=$(grep -cE 'SPM_DISPATCH\|v\.blk.*\|ime' "$log")
  echo "  encode: ${enc:-?}   vision-tensors-on-ime: $vime"
  grep -iE 'converting 2D bf16|q8_0 for IME2|buffer type not found' "$log" | sort -u | sed 's/.*load_tensors: /  gate: /'
}

E2B=/models/google/gemma-4-E2B-it-qat-q4_0-gguf
E4B=/models/google/gemma-4-E4B-it-qat-q4_0-gguf
QW=/models/unsloth/Qwen3.5-0.8B-GGUF

case "${1:-all}" in
  e2b1) run "e2b_test"  $E2B/gemma-4-E2B_q4_0-it.gguf $E2B/gemma-4-E2B-it-mmproj.gguf Test.png ;;
  e2b3) run "e2b_test3" $E2B/gemma-4-E2B_q4_0-it.gguf $E2B/gemma-4-E2B-it-mmproj.gguf Test3.jpg ;;
  e4b1) run "e4b_test"  $E4B/gemma-4-E4B_q4_0-it.gguf $E4B/gemma-4-E4B-it-mmproj.gguf Test.png ;;
  e4b3) run "e4b_test3" $E4B/gemma-4-E4B_q4_0-it.gguf $E4B/gemma-4-E4B-it-mmproj.gguf Test3.jpg ;;
  qw1)  run "qw_test"   $QW/Qwen3.5-0.8B-Q4_K_M.gguf $QW/mmproj-F16.gguf Test.png ;;
  qw3)  run "qw_test3"  $QW/Qwen3.5-0.8B-Q4_K_M.gguf $QW/mmproj-F16.gguf Test3.jpg ;;
esac
