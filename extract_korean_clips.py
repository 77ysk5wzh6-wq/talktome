#!/usr/bin/env python3
"""
Pull 200 clean 3-8s Korean utterances out of KsponSpeech_01 and save them
as 16kHz mono WAVs alongside cleaned transcripts.

Speaker diversity strategy: KsponSpeech doesn't expose speaker IDs in the
file paths, but each KsponSpeech_NNNN/ subfolder is a different recording
session (different speakers). We round-robin across the 124 subfolders so
no single session dominates -> ~ at least 100 distinct speakers represented.

Source layout:
  web/korean_audio/한국어_음성_분야/KsponSpeech_01/
    KsponSpeech_0001/
      KsponSpeech_000001.pcm  (16kHz 16-bit mono, headerless)
      KsponSpeech_000001.txt  (CP949, with KsponSpeech markup)
    ...

Output: web/korean_audio_clips/
  ko_000_<slug>.wav  ko_000_<slug>.txt
  manifest.tsv
"""
import re
import sys
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("web/korean_audio/한국어_음성_분야/KsponSpeech_01")
OUT = Path("web/korean_audio_clips")
COUNT = 200
SR = 16000              # KsponSpeech standard
MIN_DUR = 3.0
MAX_DUR = 8.0

# KsponSpeech transcript cleanup
DUAL = re.compile(r"\(([^()]*?)\)/\(([^()]*?)\)")    # (X)/(Y) -> Y (spoken form)
TAG = re.compile(r"\b[bonl]/")                       # b/ o/ n/ l/ markers
PAREN = re.compile(r"\([^()]*\)")                    # leftover parens
SLASH_PLUS = re.compile(r"[/+*@]")
WS = re.compile(r"\s+")

# A few characters / words we don't want in slugs
SLUG_DROP = re.compile(r"[\s.,!?\"'()/\\]+")


def read_text(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8", "cp949", "euc-kr", "utf-16"):
        try:
            return raw.decode(enc).strip()
        except Exception:
            pass
    return raw.decode("utf-8", errors="ignore").strip()


def clean(text: str) -> str:
    text = DUAL.sub(lambda m: m.group(2), text)
    text = TAG.sub("", text)
    text = PAREN.sub("", text)
    text = SLASH_PLUS.sub(" ", text)
    return WS.sub(" ", text).strip()


def slug(text: str, n: int = 16) -> str:
    s = SLUG_DROP.sub("", text)
    return s[:n] or "clip"


def pcm_duration(p: Path) -> float:
    return p.stat().st_size / (SR * 2)   # 16-bit = 2 bytes/sample


def main() -> int:
    if not ROOT.exists():
        print(f"missing source: {ROOT}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    sessions = sorted(ROOT.glob("KsponSpeech_*"))
    print(f"sessions: {len(sessions)}")

    # Build candidate list per session: pcm + cleaned text + duration
    by_session: dict[str, list] = {}
    for s in sessions:
        pcms = sorted(s.glob("*.pcm"))
        if not pcms:
            continue
        # Don't scan all 1000 in each folder — sample 50 per session
        random.seed(hash(s.name) & 0xFFFF)
        candidates = random.sample(pcms, min(50, len(pcms)))
        bucket = []
        for pcm in candidates:
            txt = pcm.with_suffix(".txt")
            if not txt.exists():
                continue
            try:
                dur = pcm_duration(pcm)
            except OSError:
                continue
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            try:
                raw = read_text(txt)
            except Exception:
                continue
            cleaned = clean(raw)
            if len(cleaned) < 4:
                continue
            bucket.append((pcm, cleaned, dur))
        if bucket:
            by_session[s.name] = bucket

    total = sum(len(v) for v in by_session.values())
    print(f"clean candidates: {total} across {len(by_session)} sessions")

    # Round-robin pick to maximise speaker diversity
    random.seed(42)
    pools = [list(v) for v in by_session.values()]
    for p in pools:
        random.shuffle(p)
    picked = []
    while pools and len(picked) < COUNT:
        for p in list(pools):
            if not p:
                pools.remove(p)
                continue
            picked.append(p.pop())
            if len(picked) >= COUNT:
                break

    print(f"sampling {len(picked)} clips")

    manifest = OUT / "manifest.tsv"
    kept = 0
    with manifest.open("w", encoding="utf-8") as mf:
        mf.write("file\ttext\tduration\trms\tsession\n")
        for pcm, text, dur in picked:
            try:
                raw = pcm.read_bytes()
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                continue
            if arr.size == 0:
                continue
            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            if rms < 0.005:
                continue

            stem = f"ko_{kept:03d}_{slug(text)}"
            wav_out = OUT / f"{stem}.wav"
            txt_out = OUT / f"{stem}.txt"
            sf.write(wav_out, arr, SR, subtype="PCM_16")
            txt_out.write_text(text + "\n", encoding="utf-8")
            mf.write(f"{wav_out.name}\t{text}\t{dur:.3f}\t{rms:.5f}\t{pcm.parent.name}\n")
            mf.flush()
            kept += 1
            if kept % 20 == 0:
                print(f"  kept {kept}/{COUNT}")

    print(f"\nDone. {kept} clips written to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
