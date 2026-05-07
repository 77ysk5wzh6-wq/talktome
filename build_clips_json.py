#!/usr/bin/env python3
"""
Consolidate the 4 language audio folders into a single web/audio/ directory
and emit a unified web/clips.json. Runtime (index.html) loads files via
the relative path stored in `file`.

Sources:
  web/english_audio/      en_pls_*.wav   (180 clips, kept)
  web/audio/              en_*.wav       (existing 37 kept)
  web/chinese_audio_clips/ zh_*.wav      (198)
  web/korean_audio_clips/  ko_*.wav      (200)
  web/japanese_audio_clips/ ja_*.wav     (200)

Output:
  web/audio/<all wavs together>
  web/audio/<all txt together>
  web/clips.json (overwritten)
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

WEB = Path("web")
AUDIO = WEB / "audio"
CLIPS_JSON = WEB / "clips.json"

SOURCES = [
    # (folder, lang, wav_glob)
    (WEB / "english_audio",          "en", "en_pls_*.wav"),
    (WEB / "chinese_audio_clips",    "zh", "zh_*.wav"),
    (WEB / "korean_audio_clips",     "ko", "ko_*.wav"),
    (WEB / "japanese_audio_clips",   "ja", "ja_*.wav"),
    (WEB / "arabic_audio_clips",     "ar", "ar_*.wav"),
    (WEB / "norwegian_audio_clips",  "no", "no_*.wav"),
]


# Target loudness so all 4 languages sit around the same perceived volume.
# Picked from the loud half of ja/zh which already feels balanced; en+ko get
# pushed up to match. We re-write the wav in place to bake the gain in.
TARGET_RMS = 0.045


def measure(wav: Path) -> tuple[float, float]:
    arr, sr = sf.read(str(wav), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    duration = arr.size / sr
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    return duration, rms


def normalize_in_place(wav: Path) -> tuple[float, float]:
    """Boost a clip toward TARGET_RMS, but never amplify beyond +12 dB
    (so quiet clips stay quiet enough not to blow up the noise floor).
    Returns the new (duration, rms)."""
    arr, sr = sf.read(str(wav), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    duration = arr.size / sr
    cur = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    if cur < 1e-6:
        return duration, cur
    gain = TARGET_RMS / cur
    # Cap gain to avoid amplifying hiss in already-quiet clips.
    gain = min(gain, 4.0)   # +12 dB ceiling
    if gain <= 1.05:
        # Already loud enough — leave alone.
        return duration, cur
    # Headroom check: scale further down if gain would clip.
    peak = float(np.max(np.abs(arr)))
    if peak * gain > 0.97:
        gain = 0.97 / max(peak, 1e-6)
    if gain <= 1.0:
        return duration, cur
    new = (arr * gain).astype(np.float32)
    sf.write(wav, new, sr, subtype="PCM_16")
    new_rms = float(np.sqrt(np.mean(new.astype(np.float64) ** 2)))
    return duration, new_rms


def read_text(p: Path) -> str:
    if not p.exists():
        return ""
    raw = p.read_bytes()
    for enc in ("utf-8", "cp949", "utf-16"):
        try:
            return raw.decode(enc).strip()
        except Exception:
            pass
    return raw.decode("utf-8", errors="ignore").strip()


def main() -> int:
    AUDIO.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    # 1. Copy new clips into web/audio/ from each language folder.
    for src_dir, lang, glob in SOURCES:
        if not src_dir.exists():
            print(f"skip missing: {src_dir}")
            continue
        wavs = sorted(src_dir.glob(glob))
        print(f"{lang}: {len(wavs)} from {src_dir.name}")
        for wav in wavs:
            txt = wav.with_suffix(".txt")
            text = read_text(txt)
            if not text:
                continue
            # Copy if not already in destination (skip if same path).
            dst_wav = AUDIO / wav.name
            dst_txt = AUDIO / txt.name
            if wav.resolve() != dst_wav.resolve():
                shutil.copy2(wav, dst_wav)
                if txt.exists():
                    shutil.copy2(txt, dst_txt)
            try:
                duration, rms = normalize_in_place(dst_wav)
            except Exception as e:
                print(f"  measure fail: {dst_wav.name}: {e}")
                continue
            entries.append({
                "file": f"audio/{dst_wav.name}",
                "text": text,
                "duration": round(duration, 3),
                "rms": round(rms, 5),
                "lang": lang,
            })

    # 2. Add existing en_* clips that are still in web/audio/ (the 37 keepers).
    #    Skip files we just copied in (handled above).
    existing_en = sorted(AUDIO.glob("en_*.wav"))
    seen = {Path(e["file"]).name for e in entries}
    for wav in existing_en:
        if wav.name in seen:
            continue
        if wav.name.startswith("en_pls_"):
            continue   # those came from web/english_audio above
        txt = wav.with_suffix(".txt")
        text = read_text(txt)
        if not text:
            continue
        try:
            duration, rms = normalize_in_place(wav)
        except Exception as e:
            print(f"  measure fail: {wav.name}: {e}")
            continue
        entries.append({
            "file": f"audio/{wav.name}",
            "text": text,
            "duration": round(duration, 3),
            "rms": round(rms, 5),
            "lang": "en",
        })

    print(f"\nTotal clips in unified clips.json: {len(entries)}")
    from collections import Counter
    print("By language:", dict(Counter(e["lang"] for e in entries)))

    CLIPS_JSON.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {CLIPS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
