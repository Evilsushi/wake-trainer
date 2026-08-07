# Wake-word training — handoff

Recovered 2026-08-07 after the driving agent session crashed mid-training.
Source of truth for the reconstruction: `~/.claude/projects/-dev/cd324a8c-fa1f-4526-8331-832e3ab1becb.jsonl`
(session ran with cwd `/dev`, which is why it is filed under the `-dev` project folder rather than an
EchoDot one — worth knowing if this needs digging up again).

## Goal

Replace the stock openWakeWord models with one custom wake word per hosted mind:

| Wake phrase   | Model name    | Routes to        |
|---------------|---------------|------------------|
| Hey Claude    | `hey_claude`  | claude           |
| Hey Gemini    | `hey_gemini`  | gemini           |
| Hey Codex     | `hey_codex`   | codex            |
| Hey Copilot   | `hey_copilot` | copilot          |
| Hey Grok      | `hey_grok`    | grok             |
| Hey Genie     | `hey_genie`   | local 8B persona |

`hey_claude` is the pilot: prove the pipeline end to end, then batch the other five.

**"Alexa" is banned** — there are real Echo devices in the house and they kept triggering. Avoid
"Echo" too (Amazon alternate wake word). "Computer" is fine; it is the liberated Dot's own wake word.
`hey_rhasspy` is only a stopgap for the general persona until `hey_genie` exists.

Persona routing, the transcript prefix matcher, and the Whisper `initial-prompt` bias are already
wired for all six names, so each `.onnx` becomes live the moment it lands in
`rust/edotd/models/` and is added to `wake-models` in `voice-node.toml`.

## Pipeline

CPU-only, no recordings of Brian needed — everything is synthetic.

- `setup.sh` — one-time, idempotent. Builds `.venv` (Python 3.14), clones `dscripka/openWakeWord`
  and `rhasspy/piper-sample-generator`, fetches the LibriTTS-R piper checkpoint, the precomputed
  negative/validation features, MIT room impulse responses, and generates noise beds.
- `train_word.sh "<phrase>" <name>` — 3 phases into `train_<name>.log`:
  1. generate clips (3000 positive_train / 500 positive_test / 3000 negative_train / 500 negative_test)
  2. augment (reverb + noise) and compute features
  3. train the classifier head, 20000 steps → `models/<name>.onnx`
- `generate_samples_shim.py` — copied over `piper-sample-generator/generate_samples.py`; bridges
  openWakeWord's old `from generate_samples import generate_samples` call to the restructured v3
  package, injects the TTS checkpoint, and resamples output 22050 Hz → 16 kHz.

Phases 1 and 2 are skipped automatically when their outputs already exist, so a re-run after a
failed phase 3 costs seconds.

## Environment fixes (all already applied — do not re-derive)

Python 3.14 + current torch broke this repeatedly. Each fix is folded back into `setup.sh`:

- `speexdsp-ns` has no 3.14 wheel — install openwakeword with `--no-deps` and list real deps explicitly.
- piper-sample-generator is v3 now (package layout, no `requirements.txt`) — `pip install -e`.
- `torchinfo` and `pronouncing` are imported but undeclared — install explicitly.
- PyPI pulls the **CUDA** torchaudio; this box is an AMD Phoenix APU. Force the CPU index:
  `pip install --force-reinstall torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu`
- `acoustics/directivity.py` imports `scipy.special.sph_harm`, removed upstream — patched in place.
- HF `datasets` routes audio through torchcodec even with `decode=False` — MIT RIRs are downloaded
  as plain WAVs over HTTP instead (270 files).
- Feature extraction needs `melspectrogram.onnx` + `embedding_model.onnx` inside the package tree —
  copied from `~/EchoDot2Liberator/rust/edotd/models/`.
- `torchaudio.info` was removed in the torchcodec era but `torch_audiomentations` calls it —
  shimmed via `soundfile` in `.venv/lib/python3.14/site-packages/sitecustomize.py`.
- **Python 3.14 defaults multiprocessing to `forkserver`**, which pickles DataLoader worker args;
  openWakeWord uses lambdas → `PicklingError`. Same `sitecustomize.py` forces `fork`. This fix is
  confirmed working — it is what let phase 3 finally run.

Gotcha: if phase 3 dies, delete `work/<name>/<name>/*.npy` before re-running **only if** the crash
happened during phase 2. A partial feature file makes phase 2 report "already exist, skipping" and
then phase 3 fails on the missing companion file.

## State at the crash

Phase 3 reached **16569 / 20000 steps (83%)** at ~150 it/s and was killed at 01:09:25 when its
parent session died — roughly 22 seconds short of finishing. Nothing was wrong with the pipeline.
Features on disk are complete and valid, so resuming is just re-running `train_word.sh`.

## Next steps

1. Confirm `models/hey_claude.onnx` exists and `train_hey_claude.log` ends in `DONE ->`.
2. Copy it into `~/EchoDot2Liberator-streaming/rust/edotd/models/`, add `hey_claude` to
   `wake-models` in `voice-node.toml`.
3. Live-test against Brian's real voice. Synthetic-trained models often need a per-speaker
   threshold — use `--wake-calibrate`.
4. If it holds up, batch the other five overnight:
   `for w in "hey genie:hey_genie" "hey gemini:hey_gemini" "hey codex:hey_codex" "hey copilot:hey_copilot" "hey grok:hey_grok"; do ...`
5. Commit the `setup.sh` fork fix (uncommitted at the time of the crash).

## Unrelated open threads from the same session

- `sudo timedatectl set-ntp true` — this PC's clock is ~53 s fast and NTP is off. Needs Brian's
  password; it matters for multi-node wake arbitration.
- Speaker enrollment never run: `speaker_id.py enroll brian --seconds 15`.
- `[hub]` in `voice-node.toml` is still commented out, waiting on the other agent's pipeline.
- Watch for the Piper synthesis stall (one 13 s hang observed, cause unknown, now bounded to an
  8 s timeout + one retry; the new `tts_sentence` JSONL event will catch a recurrence).
- Noise beds are synthetic placeholders; real AudioSet/FMA negatives would be an upgrade.
