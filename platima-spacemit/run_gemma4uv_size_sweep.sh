#!/usr/bin/env bash
# Patch-24 prerequisite: find the minimum (bicubic-upscaled) Test.png size at which the
# 12B gemma4uv model reads "Hi" reliably on K3. Runs original + 256/384/512 squares.
set -u
cd /home/owner/llama-smt
export LD_LIBRARY_PATH="$PWD/build/bin"
BIN="$PWD/build/bin/llama-mtmd-cli"
MODEL=/home/owner/models/unsloth-gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
MMPROJ=/home/owner/models/unsloth-gemma-4-12B-it-qat-GGUF/mmproj-BF16.gguf
OUT=/tmp/vsweep
PROMPT="What letters or text appear in this image? Answer with only the text you see."

run() {
  local tag="$1" img="$2"
  local log="$OUT/$tag.log"
  rm -f /dev/shm/tcm_sync_standalone 2>/dev/null || true
  "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --image "$img" \
    -t 8 --threads-batch 8 --jinja -n 96 --temp 0 -v \
    -p "$PROMPT" > "$log" 2>&1
  local rc=$?
  local enc tok
  enc=$(grep -oiE 'image (slice )?encoded in [0-9]+ ms' "$log" | tail -1)
  tok=$(grep -oiE 'n_tokens[ =:]+[0-9]+|image tokens[ =:]+[0-9]+|[0-9]+ tokens' "$log" | tail -1)
  echo "### $tag  (rc=$rc)"
  echo "  encode: ${enc:-?}   tokens: ${tok:-?}"
  echo "  --- model answer ---"
  awk '/<\|channel\|>final|assistant/{f=1} f{print}' "$log" | tail -8 | sed 's/^/    /'
  echo
}

for s in orig 256 384 512; do
  run "size_$s" "$OUT/Test_$s.png"
done
echo "DONE_SWEEP"
