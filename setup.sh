#!/usr/bin/env bash
# One-time setup for custom OpenWakeWord training.
# Creates a dedicated venv, clones the training repos, and downloads the
# shared training data (~29 GB with MUSAN). Idempotent: safe to re-run after failures.
#
# Installs the CUDA torch stack by default (TORCH_INDEX=cu130: torch >= 2.9 is
# the first with sm_120 kernels, which Blackwell / RTX 50xx needs). Set
# TORCH_INDEX=cpu for a CPU-only box.
set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[setup $(date +%H:%M:%S)] $*"; }

torch_index="https://download.pytorch.org/whl/${TORCH_INDEX:-cu130}"
if [[ "${TORCH_INDEX:-cu130}" == cpu ]]; then ort=onnxruntime; else ort=onnxruntime-gpu; fi

# Prefer uv: it needs no ensurepip (absent on stock Debian/Ubuntu pythons
# without python3-venv) and resolves the torch stack far faster than pip.
if command -v uv >/dev/null; then
  [[ -d .venv ]] || { log "creating venv (uv)"; uv venv -q --python 3.14 .venv; }
  pip() { uv pip "$1" -q --python .venv/bin/python "${@:2}"; }
else
  [[ -d .venv ]] || { log "creating venv"; python3 -m venv .venv; }
  pip() { .venv/bin/pip "$1" -q "${@:2}"; }
  pip install --upgrade pip
fi
py=.venv/bin/python

log "installing python deps (torch from $torch_index)"
pip install torch torchaudio torchcodec --index-url "$torch_index"
pip install torchmetrics speechbrain audiomentations torch-audiomentations \
  acoustics mutagen datasets soundfile scipy onnx "$ort" tqdm pyyaml

if [[ ! -d openWakeWord ]]; then
  log "cloning openWakeWord"
  git clone -q https://github.com/dscripka/openWakeWord
fi
# --no-deps: openwakeword pins speexdsp-ns (inference-only noise suppression)
# which has no wheel for this python; training does not need it.
pip install --no-deps -e ./openWakeWord
pip install scikit-learn requests scipy tqdm

if [[ ! -d piper-sample-generator ]]; then
  log "cloning piper-sample-generator"
  git clone -q https://github.com/rhasspy/piper-sample-generator
fi
# v3 is a proper package (piper-tts, webrtcvad, audiomentations deps)
pip install -e ./piper-sample-generator
# openWakeWord's train.py imports the old v1/v2 generate_samples module
cp generate_samples_shim.py piper-sample-generator/generate_samples.py

log "post-install fixes"
# train.py needs these but no package declares them
# onnxscript: torch>=2.9's torch.onnx.export imports it even on the legacy path
# deep-phonemizer: adversarial-negative generation falls back to it for any
# word outside CMUdict, so "codex"/"grok" need it but "gemini"/"claude" do not.
# Easy to miss - the first few wake words train fine without it.
# transformers: verify_clips.py's in-process Whisper when no Wyoming server answers
pip install torchinfo pronouncing onnxscript deep-phonemizer transformers
# Later installs can drag torch back to whatever pypi resolves (a different
# CUDA build, or CPU); re-pin the torch stack to $torch_index (must run after
# every package install above). torchcodec is torchaudio's I/O backend now and
# has the same trap. Same for onnxruntime: anything depending on the plain CPU
# package installs it alongside onnxruntime-gpu, and whichever wrote last owns
# the shared module directory, silently dropping CUDAExecutionProvider.
pip install --reinstall torch torchaudio torchcodec --index-url "$torch_index"
if [[ "$ort" == onnxruntime-gpu ]]; then
  pip uninstall onnxruntime 2>/dev/null || true
  pip install --reinstall onnxruntime-gpu
fi
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

# Real background audio. The noise beds above are synthetic, and mixing
# positives over real music, noise and babble is what teaches the model to stay
# quiet through a TV. MUSAN (openslr 17, ~11 GB, 16 kHz wav, so no FFmpeg)
# covers all three. Flattened into data/background/{music,noise,speech}
# because openWakeWord lists background dirs non-recursively. MUSAN=0 skips.
if [[ ! -d data/background && "${MUSAN:-1}" == 1 ]]; then
  if [[ ! -d data/musan_dl/musan ]]; then
    log "downloading MUSAN background audio (~11 GB)"
    mkdir -p data/musan_dl
    curl -fL -C - --progress-bar -o data/musan_dl/musan.tar.gz \
      https://www.openslr.org/resources/17/musan.tar.gz
    tar -xzf data/musan_dl/musan.tar.gz -C data/musan_dl
    rm data/musan_dl/musan.tar.gz
  fi
  rm -rf data/background.tmp
  for kind in music noise speech; do
    mkdir -p "data/background.tmp/$kind"
    find "data/musan_dl/musan/$kind" -name '*.wav' -exec mv -t "data/background.tmp/$kind" {} +
  done
  mv data/background.tmp data/background
  rm -rf data/musan_dl
  log "background audio: $(find data/background -name '*.wav' | wc -l) MUSAN clips"
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

    # torchaudio.load goes through torchcodec too, which dlopens the system
    # FFmpeg (libavutil.so.5x) and fails outright on a box without it. Every
    # file this pipeline reads is a wav, so fall back to soundfile - sticky
    # after the first failure, since each torchcodec attempt probes five
    # FFmpeg majors before giving up.
    _load = torchaudio.load
    _use_soundfile = []

    def _load_soundfile(uri, frame_offset=0, num_frames=-1, *args, **kwargs):
        import soundfile
        import torch

        stop = None if num_frames is None or num_frames < 0 else frame_offset + num_frames
        data, sr = soundfile.read(str(uri), start=frame_offset, stop=stop,
                                  dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def _load_with_fallback(uri, *args, **kwargs):
        if not _use_soundfile:
            try:
                return _load(uri, *args, **kwargs)
            except (RuntimeError, OSError, ImportError):
                _use_soundfile.append(True)
        return _load_soundfile(uri, *args, **kwargs)

    torchaudio.load = _load_with_fallback

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

# Words outside CMUdict send adversarial-negative generation to DeepPhonemizer,
# whose checkpoint URL (public-asai-dl-models S3) now answers 403 - and
# openWakeWord saves that error page as the .pt without checking. Merge the
# repo's pronunciations.txt into pronouncing's dictionary instead, so known
# words never reach that path. The venv sits in the repo, hence sys.prefix.
try:
    import io
    import sys
    from pathlib import Path

    import pronouncing

    _extra = Path(sys.prefix).parent / "pronunciations.txt"
    _init_cmu = pronouncing.init_cmu

    def _init_cmu_with_extra(filehandle=None):
        if filehandle is None and pronouncing.pronunciations is None and _extra.exists():
            import cmudict

            with cmudict.dict_stream() as base:
                # parse_cmu chokes on blank lines, so drop them
                lines = (base.read() + b"\n" + _extra.read_bytes()).splitlines()
                filehandle = io.BytesIO(b"\n".join(l for l in lines if l.strip()))
        return _init_cmu(filehandle)

    pronouncing.init_cmu = _init_cmu_with_extra
except Exception:
    pass

# DeepPhonemizer (adversarial negatives for words outside CMUdict) ships a
# pickled checkpoint, and torch>=2.6 defaults torch.load to weights_only=True.
# Allowlist just the three dp classes it needs rather than turning the safety
# check off wholesale.
try:
    import torch
    from dp.preprocessing.text import (
        LanguageTokenizer,
        Preprocessor,
        SequenceTokenizer,
    )

    torch.serialization.add_safe_globals(
        [Preprocessor, LanguageTokenizer, SequenceTokenizer]
    )
except Exception:
    pass
PYEOF

# Feature extraction needs the frozen openWakeWord frontend (melspectrogram +
# embedding) in the package tree. These are the same two files any openWakeWord
# runtime loads, so if you already have them locally point OWW_FRONTEND_DIR at
# that directory to skip the download; otherwise they come from upstream.
mkdir -p openWakeWord/openwakeword/resources/models
if [[ -n "${OWW_FRONTEND_DIR:-}" \
      && -f "$OWW_FRONTEND_DIR/melspectrogram.onnx" \
      && -f "$OWW_FRONTEND_DIR/embedding_model.onnx" ]]; then
  log "using frontend models from $OWW_FRONTEND_DIR"
  cp "$OWW_FRONTEND_DIR/melspectrogram.onnx" \
     "$OWW_FRONTEND_DIR/embedding_model.onnx" \
     openWakeWord/openwakeword/resources/models/
else
  log "downloading openWakeWord frontend models"
  .venv/bin/python -c "import openwakeword.utils as u; u.download_models()"
fi

log "setup complete"
