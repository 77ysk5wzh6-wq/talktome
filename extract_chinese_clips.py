#!/usr/bin/env python3
"""
Slice 200 clean 3-8s utterances out of the MagicData Mandarin Chinese
Conversational Speech Corpus and save them as 16kHz mono WAVs.

Source: web/chinese_audio/Mandarin_Chinese_Conversational_Speech_Corpus/
  WAV/<session>.wav   long F2F session
  TXT/<session>.txt   UTF-16 transcript with [start,end]\tspk\tgender,lang\ttext

Output: web/chinese_audio_clips/
  zh_000_<slug>.wav  zh_000_<slug>.txt  ...
  manifest.tsv (file, text, duration, rms)
"""
import re
import sys
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

ROOT = Path("web/chinese_audio/Mandarin_Chinese_Conversational_Speech_Corpus")
OUT = Path("web/chinese_audio_clips")
COUNT = 200
TARGET_SR = 16000
MIN_DUR = 3.0
MAX_DUR = 8.0

NOISE = re.compile(r"\[(LAUGH|SONANT|ENS|MUSIC|UNK|SYSTEM|SIL|PII|\*)\]|\+|\$")
LINE = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+(\S+)\s+(.*)")

# Drop punctuation/whitespace for slug
SLUG_DROP = re.compile(r"\s+")


def read_txt(p: Path) -> list[str]:
    raw = p.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16").splitlines()
    try:
        return raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="ignore").splitlines()


def slug(text: str, n: int = 16) -> str:
    s = SLUG_DROP.sub("", text)
    return s[:n] or "clip"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    txt_dir = ROOT / "TXT"
    wav_dir = ROOT / "WAV"
    txts = sorted(txt_dir.glob("*.txt"))
    if not txts:
        print(f"no TXT files in {txt_dir}", file=sys.stderr)
        return 1

    # Build candidate list across all sessions, then sample.
    # Each candidate = (session_stem, start, end, text).
    candidates = []
    for txt in txts:
        stem = txt.stem
        wav_path = wav_dir / f"{stem}.wav"
        if not wav_path.exists():
            continue
        for line in read_txt(txt):
            line = line.strip()
            if not line.startswith("["):
                continue
            m = LINE.match(line)
            if not m:
                continue
            start, end, spk, meta, text = m.groups()
            text = text.strip()
            if not text or NOISE.search(text):
                continue
            dur = float(end) - float(start)
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            candidates.append((stem, float(start), float(end), text))

    print(f"found {len(candidates)} clean candidates across {len(txts)} sessions")
    if len(candidates) < COUNT:
        print(f"warning: only {len(candidates)} candidates available", file=sys.stderr)

    # Reproducible sampling, biased toward distributing across sessions.
    random.seed(42)
    by_session: dict[str, list] = {}
    for c in candidates:
        by_session.setdefault(c[0], []).append(c)
    # Round-robin pick to avoid single session dominating.
    pools = list(by_session.values())
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

    # Cache loaded session waveforms so we don't reload the same .wav for
    # multiple slices.
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    manifest = OUT / "manifest.tsv"
    kept = 0
    with manifest.open("w", encoding="utf-8") as mf:
        mf.write("file\ttext\tduration\trms\tsource_session\n")
        for stem, start, end, text in picked:
            if stem not in audio_cache:
                w, sr = sf.read(str(wav_dir / f"{stem}.wav"), dtype="float32")
                if w.ndim > 1:
                    w = w.mean(axis=1)
                if sr != TARGET_SR:
                    w = librosa.resample(w, orig_sr=sr, target_sr=TARGET_SR)
                    sr = TARGET_SR
                audio_cache[stem] = (w, sr)

            full, sr = audio_cache[stem]
            i0 = int(start * sr)
            i1 = int(end * sr)
            seg = full[i0:i1]
            if seg.size == 0:
                continue

            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
            if rms < 0.005:
                continue

            dur_actual = seg.size / sr
            stem_out = f"zh_{kept:03d}_{slug(text)}"
            wav_out = OUT / f"{stem_out}.wav"
            txt_out = OUT / f"{stem_out}.txt"
            sf.write(wav_out, seg, sr, subtype="PCM_16")
            txt_out.write_text(text + "\n", encoding="utf-8")
            mf.write(f"{wav_out.name}\t{text}\t{dur_actual:.3f}\t{rms:.5f}\t{stem}\n")
            mf.flush()

            kept += 1
            if kept % 20 == 0:
                print(f"  kept {kept}/{COUNT}")

    print(f"\nDone. {kept} clips written to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
