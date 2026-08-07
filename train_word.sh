#!/usr/bin/env bash
# Train one custom OpenWakeWord model from a phrase, entirely from synthetic
# TTS positives (plus auto-generated adversarial negatives). Usage:
#
#   wake-training/train_word.sh "hey claude" hey_claude
#
# Produces models/<name>.onnx; copy/symlink it into rust/edotd/models and add
# it to wake-models in voice-node.toml. CPU-only: sample generation is the
# slow part (~1-2h per word); training the classifier head is minutes.
set -euo pipefail
cd "$(dirname "$0")"

phrase="${1:?usage: train_word.sh \"<phrase>\" <model_name>}"
name="${2:?usage: train_word.sh \"<phrase>\" <model_name>}"
py=.venv/bin/python
n_samples="${N_SAMPLES:-3000}"
n_val="${N_VAL:-500}"

log() { echo "[$name $(date +%H:%M:%S)] $*"; }

mkdir -p work models
cat > "work/$name.yml" <<YAML
model_name: "$name"
target_phrase:
  - "$phrase"
custom_negative_phrases: []
n_samples: $n_samples
n_samples_val: $n_val
tts_batch_size: 32
augmentation_batch_size: 16
augmentation_rounds: 1
piper_sample_generator_path: "$PWD/piper-sample-generator"
output_dir: "$PWD/work/$name"
rir_paths:
  - "$PWD/data/mit_rirs"
background_paths:
  - "$PWD/data/noise"
background_paths_duplication_rate:
  - 1
false_positive_validation_data_path: "$PWD/data/validation_set_features.npy"
feature_data_files:
  "ACAV100M_sample": "$PWD/data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
batch_n_per_class:
  "ACAV100M_sample": 1024
  "adversarial_negative": 50
  "positive": 50
model_type: "dnn"
layer_size: 32
steps: 20000
max_negative_weight: 1500
target_false_positives_per_hour: 0.2
YAML

log "phase 1/3: generating clips (positives + adversarial negatives) - the slow part"
$py openWakeWord/openwakeword/train.py --training_config "work/$name.yml" --generate_clips
log "phase 2/3: augmenting clips and computing features"
$py openWakeWord/openwakeword/train.py --training_config "work/$name.yml" --augment_clips
log "phase 3/3: training classifier"
$py openWakeWord/openwakeword/train.py --training_config "work/$name.yml" --train_model

found=$(find "work/$name" -name "*.onnx" | head -1)
if [[ -n "$found" ]]; then
  cp "$found" "models/$name.onnx"
  log "DONE -> models/$name.onnx"
else
  log "training finished but no onnx found under work/$name" >&2
  exit 1
fi
