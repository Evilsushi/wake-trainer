# wake-trainer

Trains custom wake-word classifiers from a phrase — no recorded audio needed.
Positives are synthesised with a 904-speaker TTS model, negatives come from a
2000-hour precomputed feature bank, and the output is an ONNX classifier that
drops into any openWakeWord-compatible runtime.

Built to produce the wake words for [VoiceBox](https://github.com/Evilsushi/VoiceBox),
but it has no dependency on it — the only thing crossing the boundary is the
model contract below.

## Output contract

Every model this produces is:

| Property | Value |
|---|---|
| Format | ONNX, **opset 13** |
| Input | `float32[1, 16, 96]` — 16 frames of 96-dim openWakeWord embeddings |
| Output | `float32[1, 1]` — sigmoid score in `[0, 1]` |
| Size | ~200 KB (50,401 parameters) |

Opset 13 is deliberate, not incidental. The stock openWakeWord models are
opset 13 and [tract](https://github.com/sonos/tract) — what a Rust runtime is
likely to use — is happiest there. See "Gotchas" for how easily this regresses.

A model is only the classifier head. At inference you also need the two frozen
frontend models (`melspectrogram.onnx`, `embedding_model.onnx`) that turn audio
into those 96-dim embeddings; `setup.sh` fetches them.

## Requirements

- Linux, Python 3.12+ (tested on 3.14); `uv` is used if present
- **~35 GB disk** — the negative feature bank is 17 GB, MUSAN background
  audio ~12 GB extracted (`MUSAN=0 ./setup.sh` skips it)
- **~8 GB free RAM** during training (see "Memory")
- NVIDIA GPU optional but used when present. `setup.sh` installs the cu130
  torch stack (sm_120 kernels, so RTX 50xx / Blackwell works) plus
  onnxruntime-gpu; TTS, feature extraction and the classifier all move to
  CUDA. `TORCH_INDEX=cpu ./setup.sh` builds the old CPU-only environment.
  Tested on an RTX 5080 (16 GB) under WSL2.

## Quickstart

```bash
./setup.sh                              # one-time: venv, clones, ~29G download
./train_word.sh "hey jarvis" hey_jarvis # -> models/hey_jarvis.onnx
```

First run takes a while — the 17 GB feature bank dominates. Every download is
existence-guarded, so re-running `setup.sh` is cheap and safe.

If you already have the openWakeWord frontend models locally:

```bash
OWW_FRONTEND_DIR=/path/to/models ./setup.sh   # skips that download
```

Training several words:

```bash
./train_remaining.sh    # edit the `words` array first
```

It runs strictly one at a time — see "Memory" for why — and one word failing
does not stop the rest. `retry_failed.sh` re-runs a subset afterwards.

## Tuning

Everything lives in the heredoc config inside `train_word.sh`:

| Setting | Here | Upstream reference | Effect |
|---|---|---|---|
| `n_samples` | 10000 | 10000 | Positive clips. **The main quality lever.** (3000 on the old CPU box) |
| `steps` | 20000 | 50000 | Training steps |
| `batch_n_per_class` | 512 | 1024 | Batch size — raise only if RAM allows |
| `layer_size` | 32 | 32 | Classifier width |

Most of these are env vars on `train_word.sh`: `N_SAMPLES`, `N_VAL`, `STEPS`,
`NEG_BATCH` (`batch_n_per_class`), and `TTS_BATCH` / `AUG_BATCH` (default
64 / 32, sized for a GPU — use 32 / 16 on CPU). `MAX_NEG_WEIGHT` (default
1500) caps the negative-loss ramp; upstream doubles it for each later training
sequence that misses the false-positive target, so with real background audio
and custom negatives it can overshoot into a model that never fires.
`okay_gizmo` went from 36.6% to 57.2% recall at 0.5 when lowered to 500, at
0.37 false fires/hour. Compare runs with `./eval_models.py <name> <model>...`,
which scores every model on the same held-out set.

Background audio for augmentation is MUSAN music, noise and babble
(`data/background/`), falling back to synthetic noise beds when absent.

Per-word confusables go in `negatives/<name>.txt`, one phrase per line. They
are repeated to `NEG_SHARE_PCT` (default 20) percent of the adversarial
negatives — upstream would otherwise use each one roughly once among ~10,000
generated phrases. Seed it from what Whisper mishears the phrase as in the
verify step.

Words missing from CMUdict need a line in `pronunciations.txt` (CMUdict
format). Upstream's fallback, DeepPhonemizer, downloads its checkpoint from an
S3 bucket that now returns 403, and openWakeWord saves the error page as the
checkpoint.

`verify_clips.py` checks the TTS says the phrase before training. It talks to
a Wyoming Whisper server if one answers on `WHISPER_HOST:WHISPER_PORT`, and
otherwise runs `openai/whisper-large-v3-turbo` in-process through
transformers (first use downloads ~1.6 GB).

Reference results from the six words trained on 2026-08-07 at `n_samples: 3000`,
scored on 500 held-out synthetic clips at threshold 0.35:

| Model | Detect | False-fire |
|---|---|---|
| hey_codex | 76.0% | 0.60% |
| hey_genie | 73.8% | 0.60% |
| hey_grok | 73.8% | 0.80% |
| hey_copilot | 73.6% | 0.80% |
| hey_claude | 71.6% | 1.20% |
| hey_gemini | 61.0% | 0.40% |

Negatives are rejected easily while positives are soft, which is what a 3000 vs
5.6-million class imbalance looks like. Raise `n_samples` toward upstream's
10000 if a word under-triggers. Note these are *per-window* scores on synthetic
speech — a real runtime slides a window and takes the peak across an utterance,
so this is a lower bound, not a field miss rate.

## Memory

Training peaks near **7 GB resident**, on top of memory-mapping the 17 GB
feature bank. On a 28 GB machine with a browser and two language servers
running, that was enough to get the trainer OOM-killed at ~75% of the first
sequence — twice, reproducibly.

If it dies without a traceback, that is the kernel, not a bug. Check:

```bash
journalctl -k --since "1 hour ago" | grep -i "killed process"
```

On the GPU build (RTX 5080, WSL2 with 15 GB visible) a full default run of
`hey_claude` peaked at 8.7 GB RSS — mapped feature-bank pages included — and
finished in under 9 minutes end to end: ~35 s of TTS, ~25 s of features, the
rest classifier training.

Lower `batch_n_per_class` before anything else; it cuts both the working set
and the page-cache churn. Never run two words concurrently.

## Gotchas

Four upstream-compatibility problems are patched here, all the same shape — a
modern-torch default breaking an older library. They are handled in
`setup.sh`'s generated `sitecustomize.py`, documented inline, and listed here
because each one costs a full train-and-fail cycle to rediscover:

1. **`torchaudio.info` removed** in the torchcodec era — shimmed via soundfile.
2. **Python 3.14 defaults to `forkserver`**, which needs picklable worker args;
   openWakeWord's DataLoader uses lambdas. Forced back to `fork`.
3. **`torch.onnx.export` silently emits opset 18.** torch ≥2.9 routes through
   the dynamo exporter, which *logs* rather than raises when it cannot
   down-convert to the requested opset. The legacy exporter is pinned via
   `dynamo=False`. Without this you get a model that trains fine and then fails
   to load in tract.
4. **DeepPhonemizer checkpoints will not unpickle** under torch ≥2.6's
   `weights_only=True` default. Three `dp` classes are allowlisted explicitly
   rather than disabling the safety check.

Two more worth knowing:

- **Adversarial negatives need CMUdict.** Words outside it (e.g. *codex*,
  *grok*) fall back to DeepPhonemizer, so gotcha 4 only bites on some words —
  the first few train fine and hide it.
- **tflite conversion always fails** unless `onnx_tf` (and TensorFlow) is
  installed. This is non-fatal by design: the ONNX is written first, and
  success is gated on that file existing.

## Layout

```
setup.sh              one-time environment build (idempotent)
train_word.sh         train one word: <phrase> <model_name>
train_remaining.sh    batch driver, strictly sequential
retry_failed.sh       re-run a subset after a partial batch
verify_clips.py       Whisper check that the TTS says the phrase
eval_models.py        score models on one shared held-out set
negatives/<name>.txt  per-word confusable phrases
pronunciations.txt    CMUdict entries for words it lacks
generate_samples_shim.py
                      adapts piper-sample-generator v3 to the v1/v2 API
                      openWakeWord expects, and resamples output to 16 kHz
HANDOFF.md            original project notes
```

Not in git: `data/` (corpora), `work/` (per-word intermediates), `models/`
(trained artifacts), `.venv/`, and the two vendored upstream clones.

## Credits

Stands on [openWakeWord](https://github.com/dscripka/openWakeWord) by David
Scripka and [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
by Rhasspy. This repo is the training pipeline and the environment fixes around
them; the embedding model, feature bank, and TTS are theirs.
