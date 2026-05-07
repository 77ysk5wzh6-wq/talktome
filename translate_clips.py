#!/usr/bin/env python3
"""
Add a Korean gloss to every non-Korean ENTITY/ACTION/QUALITY/RELATION
token in web/clips.json.

We only translate unique (lang, surface) pairs and cache the result, so
repeated words across captions only cost one Google Translate hit each.
The gloss is stored on each token as `ko`.

Usage:
    python translate_clips.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

CLIPS = Path("web/clips.json")
TRANSLATABLE_CATS = {"ENTITY", "ACTION", "QUALITY", "RELATION"}
# Only translate words at least this many characters; single-char tokens are
# almost always grammar that doesn't need a Korean gloss.
MIN_SURF_LEN = 2

# deep_translator language codes.
LANG_TO_DT = {
    "en": "en",
    "zh": "zh-CN",
    "ja": "ja",
    "ar": "ar",
    "no": "no",
}


def main() -> int:
    if not CLIPS.exists():
        print(f"missing {CLIPS}", file=sys.stderr)
        return 1
    print(f"Loading {CLIPS}...")
    clips = json.loads(CLIPS.read_text(encoding="utf-8"))
    print(f"  {len(clips)} clips")

    # Collect unique (lang, surface) pairs across translatable categories.
    pairs: dict[tuple[str, str], str | None] = {}
    for clip in clips:
        lang = clip.get("lang")
        if lang == "ko" or lang not in LANG_TO_DT:
            continue
        for t in clip.get("tokens") or []:
            if t.get("cat") not in TRANSLATABLE_CATS:
                continue
            surf = (t.get("surf") or "").strip()
            if len(surf) < MIN_SURF_LEN:
                continue
            pairs[(lang, surf)] = None
    print(f"  {len(pairs)} unique (lang, surface) pairs to translate")

    # Translate. Group by source lang so we re-use translator instances.
    by_lang: dict[str, list[str]] = {}
    for (lang, surf) in pairs:
        by_lang.setdefault(lang, []).append(surf)

    cache: dict[tuple[str, str], str] = {}
    for lang, surfs in by_lang.items():
        dt_code = LANG_TO_DT[lang]
        translator = GoogleTranslator(source=dt_code, target="ko")
        print(f"\n[{lang} -> ko] {len(surfs)} words")
        for i, surf in enumerate(surfs, 1):
            try:
                ko = translator.translate(surf)
            except Exception as e:
                print(f"  fail {lang}/{surf!r}: {e}")
                ko = ""
                # Brief pause on errors to avoid hammering the rate limit.
                time.sleep(0.5)
            cache[(lang, surf)] = (ko or "").strip()
            if i % 50 == 0:
                print(f"  ...{i}/{len(surfs)}  e.g. {surf!r} -> {ko!r}")

    # Attach gloss to every translatable token in every clip.
    print("\nAttaching glosses to tokens...")
    annotated = 0
    for clip in clips:
        lang = clip.get("lang")
        if lang == "ko" or lang not in LANG_TO_DT:
            continue
        for t in clip.get("tokens") or []:
            if t.get("cat") not in TRANSLATABLE_CATS:
                continue
            surf = (t.get("surf") or "").strip()
            if len(surf) < MIN_SURF_LEN:
                continue
            ko = cache.get((lang, surf))
            if ko:
                t["ko"] = ko
                annotated += 1
    print(f"  annotated {annotated} tokens")

    # A few samples for sanity-check.
    print("\nSample translations:")
    sample_keys = list(cache.keys())[:15]
    for k in sample_keys:
        print(f"  [{k[0]}] {k[1]!r}  ->  {cache[k]!r}")

    print(f"\nWriting back to {CLIPS}...")
    CLIPS.write_text(
        json.dumps(clips, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
