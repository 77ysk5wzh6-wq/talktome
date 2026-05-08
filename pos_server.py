#!/usr/bin/env python3
"""
Tiny localhost POS server for the on-stage text-input interaction.

The control window in the browser sends user text; the main canvas POSTs
that text to /tag and turns the response into a "huge" caption with the
same UD-based ENTITY/ACTION/QUALITY/RELATION categories the rest of the
piece uses. Stanza handles English and Korean; language is auto-detected
from the presence of Hangul characters.

Usage:
    python pos_server.py            # listens on http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import stanza


HOST = "127.0.0.1"
PORT = 8765
HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")

# Mapping copied from build_clips_with_upos.py so the live tagger and the
# baked clip data agree on category buckets.
UPOS_TO_CAT = {
    "NOUN": "ENTITY", "PROPN": "ENTITY", "PRON": "ENTITY",
    "VERB": "ACTION", "AUX": "ACTION",
    "ADJ":  "QUALITY", "ADV": "QUALITY",
    "ADP":  "RELATION", "CCONJ": "RELATION", "SCONJ": "RELATION",
    "DET":  "RELATION", "PART": "RELATION",
}


def detect_lang(text: str) -> str:
    return "ko" if HANGUL_RE.search(text) else "en"


print("loading stanza pipelines (en, ko)...", file=sys.stderr)
PIPELINES = {
    "en": stanza.Pipeline("en", processors="tokenize,pos",
                          tokenize_no_ssplit=True, verbose=False),
    "ko": stanza.Pipeline("ko", processors="tokenize,pos",
                          tokenize_no_ssplit=True, verbose=False),
}
print("ready.", file=sys.stderr)


def tag(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"lang": "en", "tokens": []}
    lang = detect_lang(text)
    doc = PIPELINES[lang](text)
    tokens = []
    for sent in doc.sentences:
        for w in sent.words:
            cat = UPOS_TO_CAT.get(w.upos, "OTHER")
            tokens.append({
                "surf": w.text,
                "upos": w.upos,
                "cat": cat,
            })
    return {"lang": lang, "tokens": tokens}


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != "/tag":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self._cors()
            self.end_headers()
            self.wfile.write(b"bad json")
            return
        text = (payload.get("text") or "")[:200]   # hard cap
        try:
            result = tag(text)
        except Exception as e:
            self.send_response(500)
            self._cors()
            self.end_headers()
            self.wfile.write(str(e).encode())
            return
        out = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        # Quieter than the default access log.
        sys.stderr.write(f"  [{self.path}] {fmt % args}\n")


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"listening on http://{HOST}:{PORT}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
