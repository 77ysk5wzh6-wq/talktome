#!/usr/bin/env python3
"""
Score every English clip on a few cheap quality metrics, print a sorted
report, and (with --apply) move the worst-rated files into web/audio_rejected/
so the runtime only sees clean takes.

Metrics (per clip):
    snr_db          : RMS of voiced frames / RMS of unvoiced frames, in dB.
                      Voiced/unvoiced is decided by a per-frame energy gate.
                      Higher is cleaner.
    flatness_voiced : geometric/arithmetic mean of the spectrum on voiced
                      frames. Speech sits low (~0.05-0.2). Hiss/static climbs.
    hf_ratio        : energy above 4 kHz / total energy. Above ~0.25 usually
                      means broadband noise rather than speech.
    clipping_pct    : fraction of samples at +/- full scale.

A clip is flagged "bad" if it fails any of:
    snr_db < SNR_MIN
    flatness_voiced > FLATNESS_MAX
    hf_ratio > HF_MAX
    clipping_pct > CLIP_MAX
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Thresholds tuned for People's Speech-style content (meetings, podcasts).
# Conservative: better to keep a borderline clip than reject too aggressively.
SNR_MIN = 11.0          # dB - accept slight room ambience
FLATNESS_MAX = 0.38     # speech with some treble still scores high here
HF_MAX = 0.30
CLIP_MAX = 0.005        # 0.5% of samples at full scale

FRAME_MS = 30
HOP_MS = 15


def frame(x: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    n = int(sr * frame_ms / 1000)
    h = int(sr * hop_ms / 1000)
    if x.size < n:
        return np.zeros((1, n), dtype=x.dtype)
    num = 1 + (x.size - n) // h
    out = np.lib.stride_tricks.as_strided(
        x, shape=(num, n), strides=(x.strides[0] * h, x.strides[0]),
    )
    return out


def spectral_flatness(frame_block: np.ndarray) -> np.ndarray:
    # frame_block: (num_frames, n)
    spec = np.abs(np.fft.rfft(frame_block, axis=1)) + 1e-10
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1)
    return geo / arith


def hf_energy_ratio(frame_block: np.ndarray, sr: int, cutoff_hz: float = 4000) -> np.ndarray:
    spec = np.abs(np.fft.rfft(frame_block, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_block.shape[1], d=1 / sr)
    hi = spec[:, freqs >= cutoff_hz].sum(axis=1)
    total = spec.sum(axis=1) + 1e-12
    return hi / total


def analyze(path: Path) -> dict:
    arr, sr = sf.read(str(path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if arr.size == 0:
        return {"file": path.name, "error": "empty"}

    duration = arr.size / sr
    clipping_pct = float(np.mean(np.abs(arr) >= 0.999))

    blocks = frame(arr, sr, FRAME_MS, HOP_MS)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)
    if rms.size == 0:
        return {"file": path.name, "error": "no_frames"}

    # Voiced/unvoiced gate: top half by energy = voiced candidates.
    gate = np.median(rms)
    voiced_mask = rms > gate
    unvoiced_mask = rms <= gate
    if voiced_mask.sum() < 3 or unvoiced_mask.sum() < 3:
        # Fall back to simple split
        sorted_idx = np.argsort(rms)
        unvoiced_mask = np.zeros_like(rms, dtype=bool)
        voiced_mask = np.zeros_like(rms, dtype=bool)
        unvoiced_mask[sorted_idx[: max(3, len(rms) // 4)]] = True
        voiced_mask[sorted_idx[-max(3, len(rms) // 4):]] = True

    voiced_rms = float(rms[voiced_mask].mean())
    unvoiced_rms = float(rms[unvoiced_mask].mean()) + 1e-9
    snr_db = 20.0 * np.log10(voiced_rms / unvoiced_rms)

    flat = spectral_flatness(blocks[voiced_mask])
    flatness_voiced = float(flat.mean())

    hf = hf_energy_ratio(blocks[voiced_mask], sr)
    hf_ratio = float(hf.mean())

    return {
        "file": path.name,
        "duration": duration,
        "snr_db": snr_db,
        "flatness_voiced": flatness_voiced,
        "hf_ratio": hf_ratio,
        "clipping_pct": clipping_pct,
    }


def is_bad(m: dict) -> tuple[bool, list[str]]:
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
    return (len(reasons) > 0, reasons)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path,
                   default=Path("web/audio"),
                   help="folder of .wav files to score")
    p.add_argument("--reject-dir", type=Path,
                   default=Path("web/audio_rejected"),
                   help="where to move rejected clips (with --apply)")
    p.add_argument("--apply", action="store_true",
                   help="actually move rejected files; default is dry-run")
    p.add_argument("--prefix", default="en_",
                   help="only consider files starting with this prefix")
    args = p.parse_args()

    wavs = sorted(args.audio.glob(f"{args.prefix}*.wav"))
    if not wavs:
        print(f"No files matching {args.prefix}*.wav in {args.audio}")
        return 1

    print(f"Scoring {len(wavs)} clips in {args.audio}\n")
    results = []
    for w in wavs:
        try:
            m = analyze(w)
        except Exception as e:
            m = {"file": w.name, "error": f"exc:{e}"}
        results.append(m)

    # Sort: rejected first (by reason count), then good clips by SNR descending.
    rejected, kept = [], []
    for m in results:
        bad, reasons = is_bad(m)
        m["_bad"] = bad
        m["_reasons"] = reasons
        (rejected if bad else kept).append(m)

    print(f"=== Rejected: {len(rejected)} / {len(wavs)} ===")
    for m in sorted(rejected, key=lambda x: x.get("snr_db", -99)):
        if "error" in m:
            print(f"  [ERR ] {m['file']:50s}  {m['error']}")
        else:
            print(
                f"  [BAD ] {m['file']:50s}  "
                f"SNR={m['snr_db']:5.1f}  flat={m['flatness_voiced']:.2f}  "
                f"hf={m['hf_ratio']:.2f}  clip={m['clipping_pct']*100:.2f}%  "
                f"-> {', '.join(m['_reasons'])}"
            )

    print(f"\n=== Kept: {len(kept)} / {len(wavs)} ===")
    for m in sorted(kept, key=lambda x: -x.get("snr_db", 0))[:20]:
        print(
            f"  [keep] {m['file']:50s}  "
            f"SNR={m['snr_db']:5.1f}  flat={m['flatness_voiced']:.2f}  "
            f"hf={m['hf_ratio']:.2f}"
        )
    if len(kept) > 20:
        print(f"  ... and {len(kept) - 20} more")

    if args.apply and rejected:
        args.reject_dir.mkdir(parents=True, exist_ok=True)
        for m in rejected:
            src = args.audio / m["file"]
            dst = args.reject_dir / m["file"]
            if src.exists():
                shutil.move(str(src), str(dst))
                # Move the matching .txt too if present
                txt_src = src.with_suffix(".txt")
                if txt_src.exists():
                    shutil.move(str(txt_src), str(args.reject_dir / txt_src.name))
        print(f"\nMoved {len(rejected)} files (and matching .txt) to {args.reject_dir}")
    elif rejected:
        print("\nDry run. Re-run with --apply to move rejected files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
