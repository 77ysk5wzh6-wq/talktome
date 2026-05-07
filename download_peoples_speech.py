#!/usr/bin/env python3
"""
Stream-download a small subset of MLCommons People's Speech (CC-BY-SA 4.0)
and save 3-8s English clips as 16kHz mono WAVs alongside their transcripts.

Usage:
    pip install datasets soundfile librosa numpy
    huggingface-cli login    # one-time, accept the People's Speech terms
    python download_peoples_speech.py --count 200 --out web/english_audio

Output:
    <out>/en_pls_000.wav, en_pls_000.txt, ...
    <out>/manifest.tsv  (file, text, duration, rms)
"""
import argparse
import os
import re
import sys
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset


SAFE_NAME = re.compile(r"[^a-z0-9]+")
TARGET_SR = 16000
MIN_DUR = 3.0
MAX_DUR = 8.0


def safe_slug(text: str, n: int = 24) -> str:
    s = SAFE_NAME.sub("_", text.lower()).strip("_")
    return s[:n] or "clip"


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=200, help="how many clips to keep")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--config", default="clean",
                   help="People's Speech config: clean | dirty | clean_sa | dirty_sa")
    p.add_argument("--split", default="train")
    p.add_argument("--start", type=int, default=0,
                   help="skip the first N stream items before sampling")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Streaming MLCommons/peoples_speech ({args.config}/{args.split})...")
    ds = load_dataset(
        "MLCommons/peoples_speech",
        args.config,
        split=args.split,
        streaming=True,
    )

    manifest_path = args.out / "manifest.tsv"
    kept = 0
    seen = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        mf.write("file\ttext\tduration\trms\n")
        for item in ds:
            seen += 1
            if seen <= args.start:
                continue

            audio = item.get("audio")
            text = (item.get("text") or "").strip()
            if not audio or not text:
                continue

            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            if arr.size == 0:
                continue

            dur = arr.size / sr
            if dur < MIN_DUR or dur > MAX_DUR:
                continue

            # Resample to 16kHz mono
            if sr != TARGET_SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            if arr.ndim > 1:
                arr = arr.mean(axis=1)

            # Skip near-silent clips
            r = rms(arr)
            if r < 0.005:
                continue

            slug = safe_slug(text)
            stem = f"en_pls_{kept:03d}_{slug}"
            wav_path = args.out / f"{stem}.wav"
            txt_path = args.out / f"{stem}.txt"

            sf.write(wav_path, arr, sr, subtype="PCM_16")
            txt_path.write_text(text + "\n", encoding="utf-8")

            mf.write(f"{wav_path.name}\t{text}\t{dur:.3f}\t{r:.5f}\n")
            mf.flush()

            kept += 1
            if kept % 10 == 0:
                print(f"  kept {kept}/{args.count} (scanned {seen})")
            if kept >= args.count:
                break

    print(f"\nDone. {kept} clips written to {args.out}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
