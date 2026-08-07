#!/usr/bin/env bash
# Retrain the wake words that failed the first batch pass. Waits for
# train_remaining.sh to finish so the two do not compete for memory.
#
# Both failures were "codex"/"grok" missing from CMUdict, which sends
# adversarial-negative generation down the DeepPhonemizer path: first a missing
# `dp` package, then torch>=2.6 refusing to unpickle its checkpoint. setup.sh
# now installs deep-phonemizer and allowlists the three dp classes.
set -uo pipefail
cd "$(dirname "$0")"

words=("hey codex:hey_codex" "hey grok:hey_grok")

while pgrep -f "[t]rain_remaining" >/dev/null; do sleep 20; done
echo "[retry $(date +%H:%M:%S)] batch finished, retraining ${#words[@]} words"

for entry in "${words[@]}"; do
  phrase="${entry%%:*}"
  name="${entry##*:}"
  rm -rf "work/$name" "work/$name.yml"
  echo "[retry $(date +%H:%M:%S)] === $name ('$phrase') ==="
  if ./train_word.sh "$phrase" "$name" > "train_$name.log" 2>&1; then
    echo "[retry $(date +%H:%M:%S)] $name OK -> models/$name.onnx"
  else
    rc=$?
    echo "[retry $(date +%H:%M:%S)] $name FAILED (exit $rc) - see train_$name.log" >&2
  fi
done

echo "[retry $(date +%H:%M:%S)] retries done"
ls -lh models/
