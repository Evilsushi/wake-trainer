#!/usr/bin/env bash
# One-time setup for custom OpenWakeWord training (CPU).
# Creates a dedicated venv, clones the training repos, and downloads the
# shared training data (~6-8 GB). Idempotent: safe to re-run after failures.
set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

if [[ ! -d .venv ]]; then
  log "creating venv"
  python3 -m venv .venv
fi
pip=.venv/bin/pip
py=.venv/bin/python

log "installing python deps (torch cpu)"
$pip -q install --upgrade pip
$pip -q install torch --index-url https://download.pytorch.org/whl/cpu
$pip -q install torchmetrics speechbrain audiomentations torch-audiomentations \
  acoustics mutagen datasets soundfile scipy onnx onnxruntime tqdm pyyaml

if [[ ! -d openWakeWord ]]; then
  log "cloning openWakeWord"
  git clone -q https://github.com/dscripka/openWakeWord
fi
# --no-deps: openwakeword pins speexdsp-ns (inference-only noise suppression)
# which has no wheel for this python; training does not need it.
$pip -q install --no-deps -e ./openWakeWord
$pip -q install scikit-learn requests scipy tqdm

if [[ ! -d piper-sample-generator ]]; then
  log "cloning piper-sample-generator"
  git clone -q https://github.com/rhasspy/piper-sample-generator
fi
# v3 is a proper package (piper-tts, webrtcvad, audiomentations deps)
$pip -q install -e ./piper-sample-generator
# openWakeWord's train.py imports the old v1/v2 generate_samples module
cp generate_samples_shim.py piper-sample-generator/generate_samples.py

mkdir -p data models
if [[ ! -f piper-sample-generator/models/en_US-libritts_r-medium.pt ]]; then
  log "downloading piper TTS checkpoint (multi-speaker LibriTTS-R)"
  mkdir -p piper-sample-generator/models
  curl -fL --progress-bar -o piper-sample-generator/models/en_US-libritts_r-medium.pt \
    https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
fi

if [[ ! -f data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy ]]; then
  log "downloading precomputed negative features (~5 GB, the big one)"
  curl -fL --progress-bar -o data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
fi
if [[ ! -f data/validation_set_features.npy ]]; then
  log "downloading validation features"
  curl -fL --progress-bar -o data/validation_set_features.npy \
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
fi

if [[ ! -d data/mit_rirs ]]; then
  log "fetching MIT room impulse responses"
  $py - <<'PYEOF'
import datasets, soundfile, pathlib
out = pathlib.Path("data/mit_rirs"); out.mkdir(parents=True, exist_ok=True)
ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
n = 0
for row in ds:
    audio = row["audio"]
    soundfile.write(out / f"rir_{n:04d}.wav", audio["array"], audio["sampling_rate"])
    n += 1
print("wrote", n, "RIRs")
PYEOF
fi

if [[ ! -d data/noise ]]; then
  log "generating synthetic background noise beds (placeholder for audioset/fma)"
  $py - <<'PYEOF'
import numpy as np, soundfile, pathlib
rng = np.random.default_rng(0)
out = pathlib.Path("data/noise"); out.mkdir(parents=True, exist_ok=True)
sr = 16000
for i in range(200):
    kind = i % 3
    n = rng.normal(0, 1, sr * 10)
    if kind == 1:  # pink-ish
        n = np.cumsum(n); n -= n.mean(); n /= max(abs(n).max(), 1e-9)
    elif kind == 2:  # band-limited hum + noise
        t = np.arange(sr * 10) / sr
        n = 0.3 * np.sin(2 * np.pi * (60 + 10 * (i % 7)) * t) + 0.4 * n / max(abs(n).max(), 1e-9)
    else:
        n /= max(abs(n).max(), 1e-9)
    soundfile.write(out / f"noise_{i:03d}.wav", (n * 0.3).astype(np.float32), sr)
print("wrote 200 noise beds")
PYEOF
fi

log "setup complete"
