#!/usr/bin/env bash
# Train one custom OpenWakeWord model from a phrase, entirely from synthetic
# TTS positives (plus auto-generated adversarial negatives). Usage:
#
#   wake-training/train_word.sh "hey claude" hey_claude
#
# Produces models/<name>.onnx; copy/symlink it into rust/edotd/models and add
# it to wake-models in voice-node.toml. Uses CUDA when torch sees a GPU (TTS,
# feature extraction and the classifier all move over); on CPU sample
# generation is the slow part (~1-2h per word).
set -euo pipefail
cd "$(dirname "$0")"

phrase="${1:?usage: train_word.sh \"<phrase>\" <model_name>}"
name="${2:?usage: train_word.sh \"<phrase>\" <model_name>}"
py=.venv/bin/python
# upstream's reference count; 3000 was a CPU-time compromise, and on a GPU the
# extra clips cost minutes (see README "Tuning")
n_samples="${N_SAMPLES:-10000}"
n_val="${N_VAL:-500}"
# GPU defaults; on CPU drop these back to 32 / 16. batch_n_per_class is the
# memory lever (see README "Memory"), not a speed one.
tts_batch="${TTS_BATCH:-64}"
aug_batch="${AUG_BATCH:-32}"
neg_batch="${NEG_BATCH:-512}"
steps="${STEPS:-20000}"
# Final negative-loss weight of the first sequence's ramp. upstream doubles it
# for each later sequence that misses target_false_positives_per_hour, so the
# effective cap is up to 4x this - lower it if recall collapses.
max_neg_weight="${MAX_NEG_WEIGHT:-1500}"

log() { echo "[$name $(date +%H:%M:%S)] $*"; }

log "device: $($py -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (no CUDA)")')"
# Phase 3 peaks near 7G resident. An OOM kill there leaves no traceback, so
# say up front when the box is too tight rather than an hour in.
avail_gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
if (( avail_gb < 9 )); then
  log "WARNING: only ${avail_gb}G RAM available; phase 3 needs ~7G. Consider NEG_BATCH=256." >&2
fi

# Real background audio (MUSAN, built by setup.sh) when present, else the
# synthetic noise beds. Upstream lists each dir non-recursively, so every
# entry must be a flat dir of wavs.
bg_dirs=()
for d in data/background/{music,noise,speech}; do
  [[ -d "$d" ]] && bg_dirs+=("$PWD/$d")
done
(( ${#bg_dirs[@]} )) || bg_dirs=("$PWD/data/noise")
bg_yaml=$(printf '  - "%s"\n' "${bg_dirs[@]}")
bg_rate=$(printf '  - 1\n%.0s' "${bg_dirs[@]}")

# Per-word confusables in negatives/<name>.txt, one phrase per line. Upstream
# puts custom phrases in front of ~n_samples generated ones and the TTS cycles
# the list once, so a dozen phrases would be ~0.1% of the negatives. Repeat
# them to NEG_SHARE (default 20%) of the adversarial clips instead.
neg_yaml="[]"
if [[ -f "negatives/$name.txt" ]]; then
  mapfile -t negs < <(grep -v '^\s*#' "negatives/$name.txt" | sed '/^\s*$/d')
  if (( ${#negs[@]} )); then
    reps=$(( (n_samples * ${NEG_SHARE_PCT:-20} / 100 + ${#negs[@]} - 1) / ${#negs[@]} ))
    neg_yaml=$(for ((i = 0; i < reps; i++)); do printf '  - "%s"\n' "${negs[@]}"; done)
    neg_yaml=$'\n'"$neg_yaml"
    log "custom negatives: ${#negs[@]} phrases x $reps from negatives/$name.txt"
  fi
fi
log "background audio: ${bg_dirs[*]#$PWD/}"

mkdir -p work models
cat > "work/$name.yml" <<YAML
model_name: "$name"
target_phrase:
  - "$phrase"
custom_negative_phrases: $neg_yaml
n_samples: $n_samples
n_samples_val: $n_val
tts_batch_size: $tts_batch
augmentation_batch_size: $aug_batch
augmentation_rounds: 1
piper_sample_generator_path: "$PWD/piper-sample-generator"
output_dir: "$PWD/work/$name"
rir_paths:
  - "$PWD/data/mit_rirs"
background_paths:
$bg_yaml
background_paths_duplication_rate:
$bg_rate
false_positive_validation_data_path: "$PWD/data/validation_set_features.npy"
feature_data_files:
  "ACAV100M_sample": "$PWD/data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
batch_n_per_class:
  "ACAV100M_sample": $neg_batch
  "adversarial_negative": 50
  "positive": 50
model_type: "dnn"
layer_size: 32
steps: $steps
max_negative_weight: $max_neg_weight
target_false_positives_per_hour: 0.2
YAML

log "phase 1/3: generating clips (positives + adversarial negatives) - the slow part"
$py openWakeWord/openwakeword/train.py --training_config "work/$name.yml" --generate_clips
# Before spending ~15 minutes training on them, check the clips actually say the
# phrase. hey_copilot trained fine on audio that said "hey cope islet", scored
# 73.6% on its own held-out set, and never once fired for a human voice. Set
# SKIP_VERIFY=1 to bypass, or WHISPER_HOST/WHISPER_PORT to point elsewhere; a
# missing Whisper skips the check rather than failing the run.
if [[ "${SKIP_VERIFY:-0}" != "1" ]]; then
  log "verifying the TTS is saying '$phrase'"
  set +e
  $py verify_clips.py "$phrase" "$name" \
    --host "${WHISPER_HOST:-127.0.0.1}" --port "${WHISPER_PORT:-10300}"
  verify_rc=$?
  set -e
  if [[ $verify_rc -eq 1 ]]; then
    log "aborting: clips do not say '$phrase' - training them would waste the run" >&2
    exit 1
  fi
fi

log "phase 2/3: augmenting clips and computing features"
# train.py skips this phase if positive_features_train.npy exists, but that
# file is the first of four memmaps it writes - a crash partway through leaves
# it behind, and every re-run then skips straight to a FileNotFoundError in
# phase 3. Treat anything short of the full set as stale.
feat="work/$name/$name"
n_feat=0
for f in "$feat"/{positive,negative}_features_{train,test}.npy; do
  [[ -f "$f" ]] && n_feat=$((n_feat + 1))
done
if (( n_feat > 0 && n_feat < 4 )); then
  log "discarding partial features from an earlier run ($n_feat/4 files)"
  rm -f "$feat"/{positive,negative}_features_{train,test}.npy
fi
$py openWakeWord/openwakeword/train.py --training_config "work/$name.yml" --augment_clips
log "phase 3/3: training classifier"
# train.py writes the onnx and *then* tries a tflite conversion that needs
# onnx_tf (pulls in tensorflow). edotd loads onnx via tract, so tflite is dead
# weight - let that step fail and gate success on the onnx existing below.
$py openWakeWord/openwakeword/train.py --training_config "work/$name.yml" --train_model || true

found=$(find "work/$name" -name "*.onnx" | head -1)
if [[ -n "$found" ]]; then
  cp "$found" "models/$name.onnx"
  log "DONE -> models/$name.onnx"
else
  log "training finished but no onnx found under work/$name" >&2
  exit 1
fi
