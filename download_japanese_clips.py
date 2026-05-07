#!/usr/bin/env python3
"""
Pull a single ReazonSpeech parquet shard (Japanese natural TV/podcast speech,
CDLA-Sharing-1.0) from a mirror that exposes parquet files directly, then
extract 200 clean 3-8s clips.

Mirror: japanese-asr/ja_asr.reazonspeech_test
  data/test-00000-of-00002.parquet  ~300MB
  features: audio (bytes), transcription (str)

Output: web/japanese_audio_clips/
  ja_000_<slug>.wav  ja_000_<slug>.txt
  manifest.tsv
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

REPO = "japanese-asr/ja_asr.reazonspeech_test"
SHARDS = [
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
]
OUT = Path("web/japanese_audio_clips")
COUNT = 200
TARGET_SR = 16000
MIN_DUR = 3.0
MAX_DUR = 8.0

# Built-in BGM filter so we don't need a separate audit pass.
# Tuned to ReazonSpeech (TV broadcast): clean speech sits at flatness ~0.20-0.30.
SNR_MIN_DB = 13.0
FLATNESS_MAX = 0.32
HF_MAX = 0.20
FRAME_MS = 30
HOP_MS = 15

# Drop whitespace + ASCII punctuation for filename slug; keep CJK chars.
SLUG_DROP = re.compile(r"[\s.,!?\"'()/\\、。「」]+")


def slug(text: str, n: int = 16) -> str:
    s = SLUG_DROP.sub("", text)
    return s[:n] or "clip"


def quality_metrics(arr: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (snr_db, flatness_voiced, hf_ratio) for BGM/quality screening."""
    n = int(sr * FRAME_MS / 1000)
    h = int(sr * HOP_MS / 1000)
    if arr.size < n:
        return 0.0, 1.0, 1.0
    num = 1 + (arr.size - n) // h
    blocks = np.lib.stride_tricks.as_strided(
        arr, shape=(num, n), strides=(arr.strides[0] * h, arr.strides[0]),
    )
    rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)
    gate = np.median(rms)
    voiced = rms > gate
    unvoiced = rms <= gate
    if voiced.sum() < 3 or unvoiced.sum() < 3:
        order = np.argsort(rms)
        unvoiced = np.zeros_like(rms, dtype=bool)
        voiced = np.zeros_like(rms, dtype=bool)
        unvoiced[order[: max(3, len(rms) // 4)]] = True
        voiced[order[-max(3, len(rms) // 4):]] = True
    snr_db = 20.0 * np.log10(
        float(rms[voiced].mean()) / (float(rms[unvoiced].mean()) + 1e-9)
    )
    spec = np.abs(np.fft.rfft(blocks[voiced], axis=1)) + 1e-10
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1)
    flat = float((geo / arith).mean())
    spec_pow = spec ** 2
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    hf = float((spec_pow[:, freqs >= 4000].sum(axis=1) /
                (spec_pow.sum(axis=1) + 1e-12)).mean())
    return snr_db, flat, hf


def main() -> int:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("HF_TOKEN env var is required", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    # Wipe any existing ja_*.wav from prior runs so numbering starts fresh.
    for old in list(OUT.glob("ja_*.wav")) + list(OUT.glob("ja_*.txt")):
        old.unlink()

    manifest = OUT / "manifest.tsv"
    rejected_log = OUT / "rejected.tsv"
    kept = 0
    scanned = 0
    rejected = 0
    rej_reasons: dict[str, int] = {}

    with manifest.open("w", encoding="utf-8") as mf, rejected_log.open("w", encoding="utf-8") as rf:
        mf.write("file\ttext\tduration\trms\tsnr_db\tflatness\thf\n")
        rf.write("scanned_idx\treason\tsnr_db\tflatness\thf\ttext\n")

        for shard in SHARDS:
            print(f"\nDownloading shard: {shard}")
            local = hf_hub_download(REPO, shard, repo_type="dataset", token=tok)
            print(f"  -> {local}")
            pf = pq.ParquetFile(local)
            print(f"  rows: {pf.metadata.num_rows}")

            for batch in pf.iter_batches(batch_size=64):
                df = batch.to_pydict()
                audios = df.get("audio", [])
                texts = df.get("transcription") or df.get("text") or []
                for audio, text in zip(audios, texts):
                    scanned += 1
                    if not audio or not text:
                        continue
                    text = (text or "").strip()
                    if not text:
                        continue
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

                    snr, flat, hf = quality_metrics(arr, sr)
                    reasons = []
                    if snr < SNR_MIN_DB: reasons.append(f"snr={snr:.1f}")
                    if flat > FLATNESS_MAX: reasons.append(f"flat={flat:.2f}")
                    if hf > HF_MAX: reasons.append(f"hf={hf:.2f}")
                    if reasons:
                        rejected += 1
                        key = ",".join(r.split("=")[0] for r in reasons)
                        rej_reasons[key] = rej_reasons.get(key, 0) + 1
                        rf.write(f"{scanned}\t{','.join(reasons)}\t{snr:.2f}\t{flat:.3f}\t{hf:.3f}\t{text[:80]}\n")
                        continue

                    stem = f"ja_{kept:03d}_{slug(text)}"
                    wav = OUT / f"{stem}.wav"
                    txt = OUT / f"{stem}.txt"
                    sf.write(wav, arr, sr, subtype="PCM_16")
                    txt.write_text(text + "\n", encoding="utf-8")
                    mf.write(f"{wav.name}\t{text}\t{dur:.3f}\t{rms:.5f}\t{snr:.2f}\t{flat:.3f}\t{hf:.3f}\n")
                    mf.flush()

                    kept += 1
                    if kept % 20 == 0:
                        print(f"  kept {kept}/{COUNT}  (scanned {scanned}, rejected {rejected})")
                    if kept >= COUNT:
                        break
                if kept >= COUNT:
                    break
            if kept >= COUNT:
                break

    print(f"\nDone. kept={kept}/{COUNT}, rejected={rejected}, scanned={scanned}")
    print("rejection reasons:", rej_reasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
