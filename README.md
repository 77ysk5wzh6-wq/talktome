# talk to number

Multilingual speech overlay where 한국어, 中文, 日本語, English captions stream
in parallel and are wired together by Universal POS category — a visual
statement that all four languages reduce to the same machine grammar.

The repository tracks code only. The audio clips themselves
(`web/audio/` etc.) are not committed; they are regenerated locally from
public datasets using the scripts in this directory.

## Datasets

| Language | Dataset | License | Tone |
|----------|---------|---------|------|
| English  | MLCommons The People's Speech (`clean/test`) | CC-BY-SA 4.0 | meetings, podcasts |
| Chinese  | MagicData Mandarin Conversational Speech Corpus | CC-BY-NC-ND 4.0 | natural conversation |
| Korean   | KsponSpeech_01 (AI Hub) | AI Hub terms | natural conversation |
| Japanese | ReazonSpeech test mirror (`japanese-asr/ja_asr.reazonspeech_test`) | CDLA-Sharing-1.0 | TV broadcast |

Cite each dataset when publishing. The English / Japanese mirrors live on
Hugging Face and require `huggingface-cli login` plus dataset terms
acceptance.

## Build pipeline

```bash
# 1. Pull English clips from People's Speech (200, BGM-clean)
python download_pls_parquet.py
python audit_english_audio.py --audio web/english_audio --prefix en_pls --apply

# 2. Pull Japanese clips from ReazonSpeech (200, BGM-filtered)
python download_japanese_clips.py

# 3. Slice 200 conversational segments from MagicData (already on disk)
python extract_chinese_clips.py

# 4. Slice 200 segments from KsponSpeech (already on disk)
python extract_korean_clips.py

# 5. Tag every transcript with Universal POS via Stanza, write web/clips.json
python build_clips_with_upos.py
```

`build_clips_with_upos.py` consolidates all four language folders into
`web/audio/`, normalises loudness, and emits per-clip token tags
(`{surf, upos, cat}`). The runtime reads `cat` (ENTITY/ACTION/QUALITY/
RELATION/OTHER) to draw cross-language wires.

## Running

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

Click 시작 to begin. The piece is intended for ambient / installation
playback at silent or near-silent room levels.

## Visual grammar

| Category | UPOS | Color | Box | Wire |
|----------|------|-------|-----|------|
| ENTITY   | NOUN, PROPN, PRON | magenta | sharp rect | hard right-angle trace |
| ACTION   | VERB, AUX | cyan | rounded rect | rounded right-angle trace |
| QUALITY  | ADJ, ADV | lime | ellipse | smooth bezier |
| RELATION | ADP, CCONJ, SCONJ, DET, PART | gray | dashed rect | dashed straight |
| OTHER    | NUM, INTJ, PUNCT, SYM, X | — | — | not drawn |
