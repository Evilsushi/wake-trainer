#!/usr/bin/env python3
"""Score trained classifiers on one shared held-out set, so models trained
with different data or settings can be compared like for like.

train.py's own "Final Model" numbers are measured on each run's own test
clips, which change whenever the augmentation or negatives do; they rank
checkpoints within a run, not models across runs.

    ./eval_models.py okay_gizmo models/okay_gizmo_v1.onnx models/okay_gizmo.onnx

The first argument names the work dir whose test features are used
(work/<name>/<name>/{positive,negative}_features_test.npy). False fires per
hour are counted on data/validation_set_features.npy (~11 h of mixed real
audio) as debounced events — a run of firing windows counts once, the way a
runtime with a refractory period would report it — alongside the raw window
count upstream reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

FRAME_S = 0.08  # one openWakeWord embedding frame
WINDOW = 16


def load(path: str) -> ort.InferenceSession:
    """The exported graph pins batch to 1 (that is the runtime contract), so
    relax it on an in-memory copy to score ~500k windows in a few calls."""
    import onnx

    m = onnx.load(path)
    for t in (*m.graph.input, *m.graph.output):
        t.type.tensor_type.shape.dim[0].dim_param = "batch"
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


def score(sess: ort.InferenceSession, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    """x: (N, 16, 96) -> (N,) scores."""
    name = sess.get_inputs()[0].name
    out = [sess.run(None, {name: x[i:i + batch].astype(np.float32)})[0].reshape(-1)
           for i in range(0, len(x), batch)]
    return np.concatenate(out)


def windows(frames: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(frames, (WINDOW, frames.shape[1]))[:, 0]


def events(scores: np.ndarray, thr: float, refractory_s: float) -> int:
    hot = np.flatnonzero(scores >= thr)
    gap = int(refractory_s / FRAME_S)
    n, last = 0, -gap - 1
    for i in hot:
        if i - last > gap:
            n += 1
        last = i
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.35])
    ap.add_argument("--refractory", type=float, default=2.0)
    args = ap.parse_args()

    feat = Path("work") / args.name / args.name
    pos = np.load(feat / "positive_features_test.npy")
    neg = np.load(feat / "negative_features_test.npy")
    val = windows(np.load("data/validation_set_features.npy"))
    hours = (len(val) + WINDOW) * FRAME_S / 3600

    print(f"test set: {len(pos)} positive, {len(neg)} adversarial negative clips; "
          f"{hours:.1f} h false-fire audio")
    print(f"{'model':32s} {'thr':>5s} {'recall':>7s} {'adv-neg fire':>13s} "
          f"{'fires/h':>8s} {'windows/h':>10s}")
    for path in args.models:
        sess = load(path)
        # a clip counts as detected if any 16-frame window inside it fires
        sp = np.array([score(sess, windows(c)).max() for c in pos])
        sn = np.array([score(sess, windows(c)).max() for c in neg])
        sv = score(sess, val)
        for thr in args.threshold:
            print(f"{Path(path).name:32s} {thr:5.2f} {np.mean(sp >= thr):7.1%} "
                  f"{np.mean(sn >= thr):13.1%} {events(sv, thr, args.refractory) / hours:8.2f} "
                  f"{np.sum(sv >= thr) / hours:10.1f}")


if __name__ == "__main__":
    main()
