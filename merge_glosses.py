#!/usr/bin/env python3
"""
Merge translate_glosses.json (a {lang: {surface: korean}} map) back into
web/clips.json by attaching `ko` to every matching token.
"""
from __future__ import annotations

import json
from pathlib import Path

CLIPS = Path("web/clips.json")
GLOSS = Path("translate_glosses.json")
TRANSLATABLE_CATS = {"ENTITY", "ACTION", "QUALITY", "RELATION"}


def main() -> None:
    clips = json.loads(CLIPS.read_text(encoding="utf-8"))
    glosses = json.loads(GLOSS.read_text(encoding="utf-8"))

    annotated = 0
    skipped = 0
    for clip in clips:
        lang = clip.get("lang")
        bucket = glosses.get(lang)
        if not bucket:
            continue
        for t in clip.get("tokens") or []:
            if t.get("cat") not in TRANSLATABLE_CATS:
                continue
            surf = (t.get("surf") or "").strip()
            ko = bucket.get(surf)
            if ko:
                t["ko"] = ko
                annotated += 1
            else:
                skipped += 1

    print(f"Annotated {annotated} tokens, skipped {skipped}")
    CLIPS.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {CLIPS}")


if __name__ == "__main__":
    main()
