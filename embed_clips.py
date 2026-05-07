#!/usr/bin/env python3
"""
Augment web/clips.json with cross-language semantic neighbours.

For every clip, every token whose UPOS category is meaningful (ENTITY /
ACTION / QUALITY) is embedded with paraphrase-multilingual-MiniLM-L12-v2.
We then compute, for each token, its top-N closest tokens *across all clips*
by cosine similarity, keep only neighbours with sim >= THRESHOLD, and
restrict to mutual neighbours (each must appear in the other's top-N).

The neighbours are stored on the token as `similar`, a list of:
    {"file": clip_file, "ti": token_index, "surf": surf, "sim": 0.83}

The runtime can read this and, for each visible token, find which other
visible tokens point back -> draw an arrow between them.

Usage:
    python embed_clips.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLIPS = Path("web/clips.json")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_CATS = {"ENTITY", "ACTION", "QUALITY"}
TOP_N = 3
THRESHOLD = 0.70
MIN_SURF_LEN = 2   # skip 1-char tokens (mostly grammar)


def main() -> int:
    if not CLIPS.exists():
        print(f"missing {CLIPS}", file=sys.stderr)
        return 1
    print(f"Loading {CLIPS}...")
    clips = json.loads(CLIPS.read_text(encoding="utf-8"))
    print(f"  {len(clips)} clips")

    # Flatten all semantic tokens with back-references so we can update them.
    flat = []   # list of (clip_idx, token_idx, surf)
    for ci, clip in enumerate(clips):
        tokens = clip.get("tokens") or []
        for ti, t in enumerate(tokens):
            if t.get("cat") not in SEMANTIC_CATS:
                continue
            surf = (t.get("surf") or "").strip()
            if len(surf) < MIN_SURF_LEN:
                continue
            flat.append((ci, ti, surf))
    print(f"  {len(flat)} semantic tokens to embed")

    print(f"Loading {MODEL} (first run downloads weights)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    surfaces = [s for _, _, s in flat]
    print(f"Encoding {len(surfaces)} surfaces (CPU)...")
    embs = model.encode(
        surfaces,
        batch_size=64,
        normalize_embeddings=True,   # cosine == dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  embeddings: shape={embs.shape}, dtype={embs.dtype}")

    # Cosine similarity matrix. ~12k x 12k floats32 = 576MB; fine for CPU.
    n = embs.shape[0]
    print(f"Computing {n}x{n} similarity matrix...")
    # Chunk the matmul to avoid one giant allocation if the dataset is huge.
    sim = embs @ embs.T          # already normalised -> cosine sim
    np.fill_diagonal(sim, -1)    # don't pair with self

    print(f"Picking top-{TOP_N} per token (threshold {THRESHOLD})...")
    # For each row, keep top-N indices where sim >= threshold.
    top_idx_per_row = []
    for i in range(n):
        row = sim[i]
        # argpartition for speed, then sort the small slice.
        if TOP_N >= n:
            cand = np.argsort(-row)
        else:
            cand_part = np.argpartition(-row, TOP_N)[:TOP_N]
            cand = cand_part[np.argsort(-row[cand_part])]
        # Filter by threshold and skip self (already -1).
        keep = [(int(j), float(row[j])) for j in cand if row[j] >= THRESHOLD]
        top_idx_per_row.append(keep)

    # Mutual filter: i->j only if j->i is also in j's top list.
    print("Filtering to mutual neighbours...")
    top_set = [set(j for j, _ in lst) for lst in top_idx_per_row]
    mutual = []
    total_pairs = 0
    for i, lst in enumerate(top_idx_per_row):
        kept = [(j, s) for (j, s) in lst if i in top_set[j]]
        mutual.append(kept)
        total_pairs += len(kept)
    print(f"  total mutual pair-edges: {total_pairs}")

    # Attach `similar` to each token in the original clips structure.
    for token_idx, neighbours in enumerate(mutual):
        ci, ti, _ = flat[token_idx]
        sim_list = []
        for j, score in neighbours:
            ncj, ntj, nsurf = flat[j]
            sim_list.append({
                "file": clips[ncj]["file"],
                "ti": ntj,
                "surf": nsurf,
                "sim": round(score, 3),
            })
        clips[ci]["tokens"][ti]["similar"] = sim_list

    # Quick stats
    edges_with_neighbours = sum(1 for m in mutual if m)
    print(f"\nTokens with at least one similar neighbour: {edges_with_neighbours}/{n}")
    if mutual:
        sample = next((m for m in mutual if m), [])
        if sample:
            i = mutual.index(sample)
            ci, ti, surf = flat[i]
            print(f"\nSample: token #{i} = '{surf}' (clip {ci}, lang={clips[ci]['lang']})")
            for j, score in sample[:5]:
                ncj, ntj, nsurf = flat[j]
                print(f"   ~ '{nsurf}' (lang={clips[ncj]['lang']}) sim={score:.3f}")

    print(f"\nWriting back to {CLIPS}...")
    CLIPS.write_text(
        json.dumps(clips, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
