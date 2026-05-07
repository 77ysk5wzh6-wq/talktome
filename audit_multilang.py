#!/usr/bin/env python3
"""
Stricter audit for non-English clips where BGM is the dominant failure mode.
Spectral flatness is the most reliable BGM detector: clean speech sits at
~0.20-0.32, while speech mixed with music/noise creeps to 0.35+.

Usage:
    python audit_multilang.py --audio web/japanese_audio_clips --prefix ja_ [--apply]
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Tighter than the English audit because BGM is the issue, not just noise floor.
SNR_MIN = 13.0
FLATNESS_MAX = 0.32
HF_MAX = 0.20
CLIP_MAX = 0.005

FRAME_MS = 30
HOP_MS = 15


def frame(x: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    n = int(sr * frame_ms / 1000)
    h = int(sr * hop_ms / 1000)
    if x.size < n:
        return np.zeros((1, n), dtype=x.dtype)
    num = 1 + (x.size - n) // h
    return np.lib.stride_tricks.as_strided(
        x, shape=(num, n), strides=(x.strides[0] * h, x.strides[0]),
    )


def spectral_flatness(blocks: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(blocks, axis=1)) + 1e-10
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1)
    return geo / arith


def hf_ratio(blocks: np.ndarray, sr: int, cutoff: float = 4000) -> np.ndarray:
    spec = np.abs(np.fft.rfft(blocks, axis=1)) ** 2
    freqs = np.fft.rfftfreq(blocks.shape[1], d=1 / sr)
    hi = spec[:, freqs >= cutoff].sum(axis=1)
    total = spec.sum(axis=1) + 1e-12
    return hi / total


def analyze(p: Path) -> dict:
    arr, sr = sf.read(str(p), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if arr.size == 0:
        return {"file": p.name, "error": "empty"}

    clipping = float(np.mean(np.abs(arr) >= 0.999))
    blocks = frame(arr, sr, FRAME_MS, HOP_MS)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)
    if rms.size == 0:
        return {"file": p.name, "error": "no_frames"}

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
    flat = float(spectral_flatness(blocks[voiced]).mean())
    hf = float(hf_ratio(blocks[voiced], sr).mean())

    return {
        "file": p.name,
        "snr_db": snr_db,
        "flatness_voiced": flat,
        "hf_ratio": hf,
        "clipping_pct": clipping,
    }


def is_bad(m: dict):
    if "error" in m:
        return True, [m["error"]]
    reasons = []
    if m["snr_db"] < SNR_MIN:
        reasons.append(f"snr={m['snr_db']:.1f}<{SNR_MIN}")
    if m["flatness_voiced"] > FLATNESS_MAX:
        reasons.append(f"flat={m['flatness_voiced']:.2f}>{FLATNESS_MAX}")
    if m["hf_ratio"] > HF_MAX:
        reasons.append(f"hf={m['hf_ratio']:.2f}>{HF_MAX}")
    if m["clipping_pct"] > CLIP_MAX:
        reasons.append(f"clip={m['clipping_pct']*100:.2f}%")
    return len(reasons) > 0, reasons


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--reject-dir", type=Path, default=None,
                   help="default: <audio>/_rejected")
    p.add_argument("--prefix", default="")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    reject_dir = args.reject_dir or (args.audio / "_rejected")

    wavs = sorted(args.audio.glob(f"{args.prefix}*.wav"))
    print(f"Scoring {len(wavs)} clips in {args.audio}")
    print(f"Thresholds: SNR>={SNR_MIN} flat<={FLATNESS_MAX} hf<={HF_MAX}\n")

    results = []
    for w in wavs:
        try:
            m = analyze(w)
        except Exception as e:
            m = {"file": w.name, "error": f"exc:{e}"}
        results.append(m)

    rejected, kept = [], []
    for m in results:
        bad, reasons = is_bad(m)
        m["_reasons"] = reasons
        (rejected if bad else kept).append(m)

    print(f"=== Rejected: {len(rejected)} / {len(wavs)} ===")
    for m in sorted(rejected, key=lambda x: x.get("snr_db", -99)):
        if "error" in m:
            print(f"  [ERR] {m['file']:50s} {m['error']}")
        else:
            print(f"  [BAD] {m['file']:50s} SNR={m['snr_db']:5.1f} flat={m['flatness_voiced']:.2f} "
                  f"hf={m['hf_ratio']:.2f} -> {', '.join(m['_reasons'])}")

    print(f"\n=== Kept: {len(kept)} / {len(wavs)} ===")

    if args.apply and rejected:
        reject_dir.mkdir(parents=True, exist_ok=True)
        for m in rejected:
            src = args.audio / m["file"]
            if src.exists():
                shutil.move(str(src), str(reject_dir / m["file"]))
                txt = src.with_suffix(".txt")
                if txt.exists():
                    shutil.move(str(txt), str(reject_dir / txt.name))
        print(f"\nMoved {len(rejected)} files to {reject_dir}")
    elif rejected:
        print("\nDry run. Re-run with --apply to move rejected files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
