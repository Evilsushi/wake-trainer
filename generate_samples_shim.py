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
TARGET_SR = 16_000


def resample_dir_to_16k(output_dir) -> int:
    """openWakeWord requires 16k clips; v3 writes at the TTS model's native
    rate (22050). Resample in place, returning the number converted."""
    import wave

    import numpy as np
    from scipy.signal import resample_poly

    converted = 0
    for path in Path(output_dir).glob("*.wav"):
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if rate == TARGET_SR:
                continue
            frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        resampled = resample_poly(frames.astype(np.float32), TARGET_SR, rate)
        clipped = np.clip(resampled, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(TARGET_SR)
            wav.writeframes(clipped.tobytes())
        converted += 1
    return converted


def generate_samples(*args, **kwargs):
    kwargs.setdefault("model", str(_MODEL))
    result = _generate_samples(*args, **kwargs)
    output_dir = kwargs.get("output_dir")
    if output_dir:
        resample_dir_to_16k(output_dir)
    return result
