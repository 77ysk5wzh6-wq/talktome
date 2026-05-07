#!/usr/bin/env python3
"""
Pull a FLEURS subset for one language (Arabic / Norwegian by default) and
extract clean 3-8s clips alongside cleaned transcripts.

FLEURS is a multilingual parallel speech corpus (CC-BY 4.0) hosted on
Hugging Face. It ships per-language tar.gz audio + TSV transcripts. We
download the test split (~600-700 utterances per language) and slice 200
clips that pass a basic loudness floor.

Usage:
    HF_TOKEN=... python download_fleurs_clips.py ar_eg --out web/arabic_audio_clips
    HF_TOKEN=... python download_fleurs_clips.py nb_no --out web/norwegian_audio_clips
"""
import argparse
import io
import os
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import hf_hub_download

REPO = "google/fleurs"
COUNT = 200
TARGET_SR = 16000
MIN_DUR = 3.0
MAX_DUR = 8.0

# Drop ASCII punctuation + Arabic/Norwegian punctuation for slug.
SLUG_DROP = re.compile(r"[\s.,!?\"'()/\\،۔؟]+")

# Language code prefix used in saved filenames (different from FLEURS code).
LANG_PREFIX = {
    "ar_eg": "ar",
    "nb_no": "no",
}


def slug(text: str, n: int = 20) -> str:
    s = SLUG_DROP.sub("", text)
    return (s[:n] or "clip")


def load_tsv(path: Path) -> dict[str, str]:
    """FLEURS TSV: id\tfilename\traw_transcription\tnormalised\tnum_samples\tgender"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            _id, fname, raw, norm = cols[0], cols[1], cols[2], cols[3]
            # Use the normalised transcription (lowercased, no extra punct).
            text = (norm or raw).strip()
            if text and fname:
                out[fname] = text
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("lang", help="FLEURS language code, e.g. ar_eg, nb_no")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--split", default="train", help="dev | test | train")
    args = p.parse_args()

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("HF_TOKEN env var is required", file=sys.stderr)
        return 1
    if args.lang not in LANG_PREFIX:
        print(f"Add {args.lang} to LANG_PREFIX first", file=sys.stderr)
        return 1
    prefix = LANG_PREFIX[args.lang]
    args.out.mkdir(parents=True, exist_ok=True)

    tsv_path = f"data/{args.lang}/{args.split}.tsv"
    tar_path = f"data/{args.lang}/audio/{args.split}.tar.gz"

    print(f"Downloading transcripts: {tsv_path}")
    local_tsv = hf_hub_download(REPO, tsv_path, repo_type="dataset", token=tok)
    print(f"  -> {local_tsv}")
    print(f"Downloading audio archive: {tar_path}")
    local_tar = hf_hub_download(REPO, tar_path, repo_type="dataset", token=tok)
    print(f"  -> {local_tar}")

    transcripts = load_tsv(Path(local_tsv))
    print(f"transcripts available: {len(transcripts)}")

    manifest = args.out / "manifest.tsv"
    kept = 0
    scanned = 0

    with tarfile.open(local_tar, "r:gz") as tar, manifest.open("w", encoding="utf-8") as mf:
        mf.write("file\ttext\tduration\trms\n")
        for member in tar:
            if not member.isfile():
                continue
            fname = Path(member.name).name
            if not fname.endswith((".wav", ".flac", ".mp3")):
                continue
            scanned += 1

            text = transcripts.get(fname)
            if not text:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            try:
                arr, sr = sf.read(io.BytesIO(data), dtype="float32")
            except Exception:
                continue
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if arr.size == 0:
                continue

            dur = arr.size / sr
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            if sr != TARGET_SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR

            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            if rms < 0.005:
                continue

            stem = f"{prefix}_{kept:03d}_{slug(text)}"
            wav_out = args.out / f"{stem}.wav"
            txt_out = args.out / f"{stem}.txt"
            sf.write(wav_out, arr, sr, subtype="PCM_16")
            txt_out.write_text(text + "\n", encoding="utf-8")
            mf.write(f"{wav_out.name}\t{text}\t{dur:.3f}\t{rms:.5f}\n")
            mf.flush()

            kept += 1
            if kept % 20 == 0:
                print(f"  kept {kept}/{COUNT}  (scanned {scanned})")
            if kept >= COUNT:
                break

    print(f"\nDone. {kept} clips written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
