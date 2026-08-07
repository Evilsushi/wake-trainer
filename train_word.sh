#!/usr/bin/env bash
# Train one custom OpenWakeWord model from a phrase, entirely from synthetic
# TTS positives. Usage:
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

workdir="work/$name"
mkdir -p "$workdir/positive_train" "$workdir/positive_test" models

if [[ ! -f "$workdir/.samples_done" ]]; then
  log "generating $n_samples train + $n_val test positives for '$phrase'"
  $py piper-sample-generator/generate_samples.py "$phrase" \
    --model piper-sample-generator/models/en_US-libritts_r-medium.pt \
    --max-samples "$n_samples" --batch-size 32 \
    --output-dir "$workdir/positive_train"
  $py piper-sample-generator/generate_samples.py "$phrase" \
    --model piper-sample-generator/models/en_US-libritts_r-medium.pt \
    --max-samples "$n_val" --batch-size 32 \
    --output-dir "$workdir/positive_test"
  touch "$workdir/.samples_done"
fi

log "writing training config"
cat > "$workdir/config.yml" <<YAML
model_name: $name
target_phrase: ["$phrase"]
n_samples: $n_samples
n_samples_val: $n_val
output_dir: $PWD/$workdir/output
piper_sample_generator_path: $PWD/piper-sample-generator
rir_paths: [$PWD/data/mit_rirs]
background_paths: [$PWD/data/noise]
background_paths_duplication_rate: [1]
feature_data_files:
  ACAV100M_sample: $PWD/data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
false_positive_validation_data_path: $PWD/data/validation_set_features.npy
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
steps: 20000
max_negative_weight: 1500
target_accuracy: 0.7
target_recall: 0.5
target_false_positives_per_hour: 0.2
model_type: dnn
layer_size: 32
augmentation_rounds: 1
augmentation_batch_size: 16
tts_batch_size: 32
custom_negative_phrases: []
YAML

log "augmenting clips + computing features"
$py openWakeWord/openwakeword/train.py --training_config "$workdir/config.yml" --augment_clips
log "training classifier"
$py openWakeWord/openwakeword/train.py --training_config "$workdir/config.yml" --train_model

found=$(find "$workdir/output" -name "$name.onnx" | head -1)
if [[ -n "$found" ]]; then
  cp "$found" "models/$name.onnx"
  log "DONE -> models/$name.onnx"
else
  log "training finished but no onnx found under $workdir/output" >&2
  exit 1
fi
