"""Compat shim: openWakeWord's train.py does `from generate_samples import
generate_samples` against piper-sample-generator's old v1/v2 layout. The v3
repo moved it into the piper_sample_generator package and made the TTS model
path a required argument. This file gets copied into the piper-sample-generator
checkout as generate_samples.py (setup.sh does this); it forwards the old call
shape and injects the LibriTTS-R checkpoint. v3's **kwargs swallows retired
arguments like auto_reduce_batch_size.
"""

from pathlib import Path

from piper_sample_generator.__main__ import generate_samples as _generate_samples

_MODEL = Path(__file__).parent / "models" / "en_US-libritts_r-medium.pt"


def generate_samples(*args, **kwargs):
    kwargs.setdefault("model", str(_MODEL))
    return _generate_samples(*args, **kwargs)
