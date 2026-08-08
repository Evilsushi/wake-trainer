#!/usr/bin/env python3
"""Check that generated positives actually say the wake phrase.

Synthesised clips are trusted blindly by the rest of the pipeline, and when the
TTS mispronounces the target the failure is *silent*: the model trains happily,
scores well on its own held-out set (which contains the same wrong audio), and
then never fires for a human. That is exactly what happened to hey_copilot,
whose clips say "hey cope islet" — it scored 73.6% synthetic and 0.001 against a
real voice.

The root cause is that words outside CMUdict fall through to a grapheme-to-
phoneme guess. `codex` and `grok` survived that path; `copilot` did not, and
nothing downstream noticed.

So: transcribe a sample of the clips with Whisper and compare against the phrase
we asked for. Speech-to-text is an independent oracle — the same trick that
makes the Rust ports verifiable, applied to the data instead of the code.

Speaks the Wyoming protocol directly over a socket (JSON header line, then the
data blob, then the payload) so the trainer gains no new dependency.

    ./verify_clips.py "hey claude" hey_claude
    ./verify_clips.py "hey claude" hey_claude --samples 25 --host 10.0.0.5

Exit codes: 0 pass, 1 below threshold, 2 could not check (no clips, no Whisper).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import socket
import statistics
import sys
import wave
from pathlib import Path

WYOMING_VERSION = "1.10.0"


def _send(sock: socket.socket, etype: str, data: dict | None = None,
          payload: bytes | None = None) -> None:
    header: dict = {"type": etype, "version": WYOMING_VERSION}
    data_bytes = None
    if data:
        data_bytes = json.dumps(data, ensure_ascii=False).encode()
        header["data_length"] = len(data_bytes)
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall(json.dumps(header, ensure_ascii=False).encode() + b"\n")
    if data_bytes:
        sock.sendall(data_bytes)
    if payload:
        sock.sendall(payload)


def _recv(reader) -> tuple[str, dict] | None:
    line = reader.readline()
    if not line:
        return None
    header = json.loads(line)
    data = {}
    if n := header.get("data_length"):
        data = json.loads(reader.read(n))
    if n := header.get("payload_length"):
        reader.read(n)
    return header.get("type", ""), data


def transcribe(path: Path, host: str, port: int, timeout: float) -> str:
    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        pcm = w.readframes(w.getnframes())

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        reader = sock.makefile("rb")
        _send(sock, "transcribe")
        meta = {"rate": rate, "width": width, "channels": channels, "timestamp": None}
        _send(sock, "audio-start", meta)
        for i in range(0, len(pcm), 8192):
            _send(sock, "audio-chunk", meta, pcm[i:i + 8192])
        _send(sock, "audio-stop", {"timestamp": None})
        while True:
            ev = _recv(reader)
            if ev is None:
                return ""
            if ev[0] == "transcript":
                return (ev[1].get("text") or "").strip()


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def similarity(target: str, heard: str) -> float:
    """Whole-phrase string similarity. Reported for context, not decided on —
    see `word_count` for why."""
    t, h = normalise(target), normalise(heard)
    if not h:
        return 0.0
    return difflib.SequenceMatcher(None, t, h).ratio()


def word_count(heard: str) -> int:
    return len(normalise(heard).split())


# Why word count rather than similarity:
#
# Whisper mangles any rare word, correct audio or not. "hey grok" transcribes as
# Grock / Grof / Gronk / Groh and scores 0.75 similarity - *worse* than broken
# hey_copilot's 0.73 - yet grok is the best-performing model in real use (0.954).
# Similarity cannot separate "unusual word" from "wrong audio".
#
# The failure mode can. A correctly pronounced "hey X" transcribes as two words
# however badly X is spelled. A mispronounced one splits: hey_copilot came out
# as "Hey, Cope, Islet" / "Hey cop, I lit" / "Hey Coop, I looked" - three and
# four words, because the TTS really did insert a boundary that is not in the
# word. That extra token is the G2P failure showing through, and it survives
# Whisper's spelling problems.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phrase")
    ap.add_argument("name")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10300)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-extra-words", type=float, default=0.5,
                    help="fail if the median transcript exceeds the phrase's own "
                         "word count by more than this")
    args = ap.parse_args()

    clips = sorted((Path("work") / args.name / args.name / "positive_train").glob("*.wav"))
    if not clips:
        print(f"[verify] no clips under work/{args.name}/{args.name}/positive_train", file=sys.stderr)
        return 2
    # Spread the sample across the run rather than taking the first N; the
    # generator varies speaker and prosody as it goes.
    step = max(1, len(clips) // args.samples)
    sample = clips[::step][:args.samples]

    rows = []
    for c in sample:
        try:
            heard = transcribe(c, args.host, args.port, args.timeout)
        except (OSError, socket.timeout) as e:
            print(f"[verify] cannot reach Whisper at {args.host}:{args.port} ({e}); skipping check",
                  file=sys.stderr)
            return 2
        rows.append((word_count(heard), similarity(args.phrase, heard), heard))

    expected = len(normalise(args.phrase).split())
    median_words = statistics.median(r[0] for r in rows)
    extra = median_words - expected
    median_sim = statistics.median(r[1] for r in rows)

    print(f"[verify] {args.name}: asked for {args.phrase!r}, checked {len(rows)} clips")
    for wc, sim, heard in sorted(rows, key=lambda r: -r[0])[:5]:
        print(f"[verify]   {wc} words  sim {sim:.2f}  {heard!r}")
    print(f"[verify] median {median_words:g} words vs {expected} expected "
          f"(+{extra:g}, allowed +{args.max_extra_words:g}); median similarity {median_sim:.2f}")

    if extra > args.max_extra_words:
        print(f"[verify] FAIL — the TTS is splitting {args.phrase!r} into extra words,",
              file=sys.stderr)
        print("[verify] which means it is mispronouncing it. A word outside CMUdict gets a",
              file=sys.stderr)
        print("[verify] grapheme-to-phoneme guess that can be badly wrong, and the model will",
              file=sys.stderr)
        print("[verify] train happily on audio nobody says. Pick a phrase that pronounces.",
              file=sys.stderr)
        return 1
    print("[verify] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
