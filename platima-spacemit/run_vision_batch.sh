#!/usr/bin/env bash
# Full bf16-vs-f16 vision matrix on bigger Qwen/Gemma. Auto-gating (no tier env vars):
# bf16 mmproj -> Tier-2 IME2 q8_0; f16 mmproj -> no-op f16 path. Compares both images.
set -u
cd /home/owner/llama-smt
export LD_LIBRARY_PATH="$PWD/build/bin"
BIN="$PWD/build/bin/llama-mtmd-cli"
OUT=/tmp/vmatrix
SUM="$OUT/batch_summary.txt"
mkdir -p "$OUT"; : > "$SUM"

MQW9=/home/owner/models/unsloth-Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
MQW4=/home/owner/models/unsloth-Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf
MGE4=/home/owner/models/unsloth-gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf

run() {
  local tag="$1" model="$2" mmproj="$3" img="$4" jinja="$5"
  local log="$OUT/$tag.log"
  local jflag=""; [ "$jinja" = "jinja" ] && jflag="--jinja"
  GGML_SPACEMIT_DISPATCH_LOG=1 timeout 1200 "$BIN" \
    -m "$model" --mmproj "$mmproj" --image "$img" \
    -t 8 --threads-batch 8 $jflag -n 400 --temp 0 -v \
    -p "Describe this image in detail." > "$log" 2>&1
  local rc=$? enc vime gate
  enc=$(grep -oiE 'image slice encoded in [0-9]+ ms' "$log" | tail -1)
  vime=$(grep -cE 'SPM_DISPATCH\|v\.blk.*\|ime' "$log")
  gate=$(grep -ciE 'q8_0 for IME2' "$log")
  echo "$tag | rc=$rc | ${enc:-noenc} | vision_on_ime=$vime | tier2_gate=$gate" | tee -a "$SUM"
}

for img in Test.png Test3.jpg; do
  s=$([ "$img" = Test.png ] && echo t1 || echo t3)
  run "qw4b_f16_$s"  $MQW4 /home/owner/models/unsloth-Qwen3.5-4B-GGUF/mmproj-F16.gguf       $img no
  run "qw9b_bf16_$s" $MQW9 /home/owner/models/unsloth-Qwen3.5-9B-GGUF/mmproj-BF16.gguf      $img no
  run "qw9b_f16_$s"  $MQW9 /home/owner/models/unsloth-Qwen3.5-9B-GGUF/mmproj-F16.gguf       $img no
  run "ge4b_f16_$s"  $MGE4 /home/owner/models/unsloth-gemma-4-E4B-it-qat-GGUF/mmproj-F16.gguf $img jinja
done
echo "ALL DONE" | tee -a "$SUM"
