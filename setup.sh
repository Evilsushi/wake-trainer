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

log "post-install fixes"
# train.py needs these but no package declares them
# onnxscript: torch>=2.9's torch.onnx.export imports it even on the legacy path
# deep-phonemizer: adversarial-negative generation falls back to it for any
# word outside CMUdict, so "codex"/"grok" need it but "gemini"/"claude" do not.
# Easy to miss - the first few wake words train fine without it.
$pip -q install torchinfo pronouncing onnxscript deep-phonemizer
# piper-sample-generator's torchaudio dep resolves to the CUDA build on pypi;
# re-pin the torch stack to CPU wheels (must run after every package install
# above). torchcodec is torchaudio's I/O backend now and has the same trap.
$pip -q install --force-reinstall torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu
# acoustics uses scipy.special.sph_harm, removed in scipy 1.15
$py - <<'PYEOF'
from pathlib import Path
p = next(Path(".venv/lib").glob("python*/site-packages/acoustics/directivity.py"))
src = p.read_text()
old = "from scipy.special import sph_harm  # pylint: disable=no-name-in-module"
new = """try:
    from scipy.special import sph_harm  # pylint: disable=no-name-in-module
except ImportError:  # scipy >= 1.15 removed sph_harm
    from scipy.special import sph_harm_y

    def sph_harm(m, n, theta, phi):
        return sph_harm_y(n, m, phi, theta)"""
if "sph_harm_y" not in src:
    p.write_text(src.replace(old, new))
    print("patched acoustics")
PYEOF

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
  # plain 16k wavs in the repo; direct download avoids the datasets/torchcodec
  # audio-decoding machinery entirely
  mkdir -p data/mit_rirs
  curl -s "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz" \
    | $py -c "import json,sys; [print(f['path']) for f in json.load(sys.stdin)]" \
    | xargs -P 8 -I{} curl -fsSL -o "data/mit_rirs/{}" --create-dirs \
        "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/main/{}"
  mv data/mit_rirs/16khz/*.wav data/mit_rirs/ && rmdir data/mit_rirs/16khz
  log "downloaded $(ls data/mit_rirs | wc -l) RIRs"
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

# torch_audiomentations calls torchaudio.info, removed in torchaudio's
# torchcodec era; provide it via soundfile through sitecustomize
cat > "$(ls -d .venv/lib/python*/site-packages)/sitecustomize.py" <<'PYEOF'
try:
    import torchaudio
except Exception:
    pass
else:
    if not hasattr(torchaudio, "info"):
        import collections

        _AudioMetaData = collections.namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )

        def _info(filepath, *args, **kwargs):
            import soundfile

            meta = soundfile.info(str(filepath))
            return _AudioMetaData(meta.samplerate, meta.frames, meta.channels, 16, "PCM_S")

        torchaudio.info = _info

# Python 3.14 defaults POSIX multiprocessing to forkserver, which requires
# picklable worker args; openWakeWord's DataLoader uses lambdas. Use fork.
import multiprocessing

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

# torch>=2.9 routes torch.onnx.export through the dynamo exporter, which emits
# opset 18 and only *logs* the failure to down-convert to the requested opset.
# edotd runs tract-onnx 0.23.4 and the stock oww models are opset 13, so an
# opset 18 export trains fine and then fails to load. Pin the legacy exporter,
# which honours opset_version for real.
try:
    import torch

    _export = torch.onnx.export

    def _export_legacy(*a, **kw):
        kw.setdefault("dynamo", False)
        return _export(*a, **kw)

    torch.onnx.export = _export_legacy
except Exception:
    pass
PYEOF

# feature extraction needs the shared oww frontend models in the package tree
mkdir -p openWakeWord/openwakeword/resources/models
cp ../../../EchoDot2Liberator/rust/edotd/models/melspectrogram.onnx \
   ../../../EchoDot2Liberator/rust/edotd/models/embedding_model.onnx \
   openWakeWord/openwakeword/resources/models/ 2>/dev/null \
  || .venv/bin/python -c "import openwakeword.utils as u; u.download_models()"

log "setup complete"
