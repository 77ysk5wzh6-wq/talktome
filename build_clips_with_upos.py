#!/usr/bin/env python3
"""
Consolidate the 4 language audio folders into web/audio/, run Stanza on every
transcript to attach Universal POS tags, and emit web/clips.json.

Output schema per clip:
  {
    "file": "audio/zh_001_...",
    "text": "...",
    "duration": 3.42,
    "rms": 0.07,
    "lang": "zh",
    "tokens": [
      {"surf": "我",   "upos": "PRON", "cat": "ENTITY"},
      {"surf": "那天", "upos": "NOUN", "cat": "ENTITY"},
      ...
    ]
  }

The runtime can ignore tokens for backwards compat, or read `cat` directly to
draw cross-language wires.

Category mapping (5 buckets):
  ENTITY   = NOUN, PROPN, PRON
  ACTION   = VERB, AUX
  QUALITY  = ADJ, ADV
  RELATION = ADP, CCONJ, SCONJ, PART, DET
  OTHER    = NUM, INTJ, PUNCT, SYM, X (no wire)
"""
import json
import shutil
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import soundfile as sf
import stanza

WEB = Path("web")
AUDIO = WEB / "audio"
CLIPS_JSON = WEB / "clips.json"

SOURCES = [
    (WEB / "english_audio",        "en", "en_pls_*.wav"),
    (WEB / "chinese_audio_clips",  "zh", "zh_*.wav"),
    (WEB / "korean_audio_clips",   "ko", "ko_*.wav"),
    (WEB / "japanese_audio_clips", "ja", "ja_*.wav"),
]

UPOS_TO_CAT = {
    "NOUN": "ENTITY",  "PROPN": "ENTITY", "PRON": "ENTITY",
    "VERB": "ACTION",  "AUX":   "ACTION",
    "ADJ":  "QUALITY", "ADV":   "QUALITY",
    "ADP":  "RELATION","CCONJ": "RELATION","SCONJ":"RELATION",
    "PART": "RELATION","DET":   "RELATION",
}


def measure(wav: Path) -> tuple[float, float]:
    arr, sr = sf.read(str(wav), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    duration = arr.size / sr
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    return duration, rms


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


def load_pipelines(langs: list[str]) -> dict:
    print("Loading Stanza models (downloads on first run)...")
    pipes = {}
    for code in langs:
        # Stanza language codes match ours: en, zh, ko, ja
        try:
            stanza.download(code, processors="tokenize,pos", verbose=False)
        except Exception as e:
            print(f"  download warn {code}: {e}")
        pipes[code] = stanza.Pipeline(
            lang=code,
            processors="tokenize,pos",
            tokenize_no_ssplit=False,
            verbose=False,
            use_gpu=False,
        )
        print(f"  {code} ready")
    return pipes


def tag_tokens(text: str, pipe) -> list[dict]:
    doc = pipe(text)
    out = []
    for sent in doc.sentences:
        for w in sent.words:
            upos = w.upos or "X"
            out.append({
                "surf": w.text,
                "upos": upos,
                "cat": UPOS_TO_CAT.get(upos, "OTHER"),
            })
    return out


def collect_clips(pipes: dict) -> list[dict]:
    AUDIO.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for src_dir, lang, glob in SOURCES:
        if not src_dir.exists():
            print(f"skip missing: {src_dir}")
            continue
        wavs = sorted(src_dir.glob(glob))
        print(f"\n{lang}: {len(wavs)} from {src_dir.name}")
        pipe = pipes.get(lang)
        for i, wav in enumerate(wavs):
            txt_p = wav.with_suffix(".txt")
            text = read_text(txt_p)
            if not text:
                continue
            dst_wav = AUDIO / wav.name
            dst_txt = AUDIO / txt_p.name
            if wav.resolve() != dst_wav.resolve():
                shutil.copy2(wav, dst_wav)
                if txt_p.exists():
                    shutil.copy2(txt_p, dst_txt)
            try:
                duration, rms = measure(dst_wav)
            except Exception as e:
                print(f"  measure fail: {dst_wav.name}: {e}")
                continue
            try:
                tokens = tag_tokens(text, pipe) if pipe else []
            except Exception as e:
                print(f"  tag fail {wav.name}: {e}")
                tokens = []
            entries.append({
                "file": f"audio/{dst_wav.name}",
                "text": text,
                "duration": round(duration, 3),
                "rms": round(rms, 5),
                "lang": lang,
                "tokens": tokens,
            })
            if (i + 1) % 50 == 0:
                print(f"  ...processed {i+1}")

    # Existing en_* clips already in web/audio (not en_pls_*)
    seen = {Path(e["file"]).name for e in entries}
    en_pipe = pipes.get("en")
    for wav in sorted(AUDIO.glob("en_*.wav")):
        if wav.name in seen or wav.name.startswith("en_pls_"):
            continue
        text = read_text(wav.with_suffix(".txt"))
        if not text:
            continue
        try:
            duration, rms = measure(wav)
        except Exception as e:
            print(f"  measure fail: {wav.name}: {e}")
            continue
        tokens = tag_tokens(text, en_pipe) if en_pipe else []
        entries.append({
            "file": f"audio/{wav.name}",
            "text": text,
            "duration": round(duration, 3),
            "rms": round(rms, 5),
            "lang": "en",
            "tokens": tokens,
        })

    return entries


def main() -> int:
    pipes = load_pipelines(["en", "zh", "ko", "ja"])
    entries = collect_clips(pipes)

    print(f"\nTotal clips: {len(entries)}")
    print("By language:", dict(Counter(e["lang"] for e in entries)))
    cat_dist = Counter()
    for e in entries:
        for t in e["tokens"]:
            cat_dist[t["cat"]] += 1
    print("By UPOS category (all langs):", dict(cat_dist))

    CLIPS_JSON.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {CLIPS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
