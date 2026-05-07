#!/usr/bin/env python3
"""
Download a single People's Speech parquet shard, then extract up to N clean
3-8s English clips into web/english_audio/ as 16kHz mono WAVs + transcripts.

Why this path: the streaming API stalls fetching the global index. Pulling
one ~100-300MB parquet directly is much faster and gives us more than 200
candidates inside it.
"""
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "MLCommons/peoples_speech"
SHARD = "clean/test-00000-of-00009.parquet"   # smaller than train shards
OUT_DIR = Path("web/english_audio")
COUNT = 200
TARGET_SR = 16000
MIN_DUR = 3.0
MAX_DUR = 8.0

SAFE = re.compile(r"[^a-z0-9]+")


def slug(text: str, n: int = 24) -> str:
    s = SAFE.sub("_", text.lower()).strip("_")
    return (s[:n] or "clip")


def main() -> int:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("HF_TOKEN env var is required", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading shard: {SHARD}")
    local_path = hf_hub_download(
        REPO, SHARD, repo_type="dataset", token=tok,
    )
    print(f"  -> {local_path}")

    # Stream rows from the parquet without loading the whole thing into RAM.
    pf = pq.ParquetFile(local_path)
    print(f"Shard rows: {pf.metadata.num_rows}, columns: {pf.schema_arrow.names}")

    manifest = OUT_DIR / "manifest.tsv"
    kept = 0
    scanned = 0
    with manifest.open("w", encoding="utf-8") as mf:
        mf.write("file\ttext\tduration\trms\n")
        for batch in pf.iter_batches(batch_size=64):
            df = batch.to_pydict()
            audios = df.get("audio", [])
            texts = df.get("text", df.get("transcript", []))
            for audio, text in zip(audios, texts):
                scanned += 1
                if not audio or not text:
                    continue
                text = (text or "").strip()
                if not text:
                    continue
                # audio is {"bytes": ..., "path": ...} or {"array":..., "sampling_rate":...}
                if "bytes" in audio and audio["bytes"]:
                    arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                elif "array" in audio:
                    arr = np.asarray(audio["array"], dtype=np.float32)
                    sr = int(audio["sampling_rate"])
                else:
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

                stem = f"en_pls_{kept:03d}_{slug(text)}"
                wav = OUT_DIR / f"{stem}.wav"
                txt = OUT_DIR / f"{stem}.txt"
                sf.write(wav, arr, sr, subtype="PCM_16")
                txt.write_text(text + "\n", encoding="utf-8")
                mf.write(f"{wav.name}\t{text}\t{dur:.3f}\t{rms:.5f}\n")
                mf.flush()

                kept += 1
                if kept % 10 == 0:
                    print(f"  kept {kept}/{COUNT}  (scanned {scanned})")
                if kept >= COUNT:
                    break
            if kept >= COUNT:
                break

    print(f"\nDone. {kept} clips written to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
