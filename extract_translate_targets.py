#!/usr/bin/env python3
"""
Pull every unique (lang, surface) pair that needs a Korean gloss out of
web/clips.json and write them to translate_targets.json so Claude (in this
session) can translate them in chunks. Korean source clips are skipped.

Output shape:
    {
      "en": ["love", "person", ...],
      "zh": ["你好", "我", ...],
      "ja": [...],
      "ar": [...],
      "no": [...]
    }
"""
from __future__ import annotations

import json
from pathlib import Path

CLIPS = Path("web/clips.json")
OUT = Path("translate_targets.json")
TRANSLATABLE_CATS = {"ENTITY", "ACTION", "QUALITY", "RELATION"}
SOURCE_LANGS = {"en", "zh", "ja", "ar", "no"}
MIN_SURF_LEN = 2


def main() -> None:
    clips = json.loads(CLIPS.read_text(encoding="utf-8"))
    print(f"Loaded {len(clips)} clips from {CLIPS}")

    by_lang: dict[str, set[str]] = {l: set() for l in SOURCE_LANGS}
    for clip in clips:
        lang = clip.get("lang")
        if lang not in SOURCE_LANGS:
            continue
        for t in clip.get("tokens") or []:
            if t.get("cat") not in TRANSLATABLE_CATS:
                continue
            if t.get("ko"):                    # already translated, skip
                continue
            surf = (t.get("surf") or "").strip()
            if len(surf) < MIN_SURF_LEN:
                continue
            by_lang[lang].add(surf)

    out = {l: sorted(s) for l, s in by_lang.items()}
    total = sum(len(v) for v in out.values())
    print(f"Unique surfaces to translate: {total}")
    for l, v in out.items():
        print(f"  {l}: {len(v)}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
