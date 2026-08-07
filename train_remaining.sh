#!/usr/bin/env bash
# Train the wake words that hey_claude did not cover, one at a time.
#
# Sequential on purpose: phase 3 alone peaked near 7G resident on the 28G box,
# so two of these at once is how the OOM kills started. Each word logs to
# train_<name>.log and a failure does not stop the ones behind it.
set -uo pipefail
cd "$(dirname "$0")"

words=(
  "hey gemini:hey_gemini"
  "hey codex:hey_codex"
  "hey copilot:hey_copilot"
  "hey grok:hey_grok"
  "hey genie:hey_genie"
)

started=$(date +%s)
echo "[batch $(date +%H:%M:%S)] training ${#words[@]} words sequentially"

for entry in "${words[@]}"; do
  phrase="${entry%%:*}"
  name="${entry##*:}"
  echo "[batch $(date +%H:%M:%S)] === $name ('$phrase') ==="
  if ./train_word.sh "$phrase" "$name" > "train_$name.log" 2>&1; then
    echo "[batch $(date +%H:%M:%S)] $name OK -> models/$name.onnx"
  else
    echo "[batch $(date +%H:%M:%S)] $name FAILED (exit $?) - see train_$name.log" >&2
  fi
done

echo "[batch $(date +%H:%M:%S)] done in $(( ($(date +%s) - started) / 60 )) min"
ls -lh models/ 2>/dev/null
