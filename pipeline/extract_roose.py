"""Extract Kevin Roose's Bing/Sydney transcript (NYT, 2023-02-16) from the Pro Memoria PDF into conversation turns.

The PDF's text layer sets Roose's messages in one font and Sydney's in another, so speaker turns come from font
changes; paragraphs come from vertical gaps. Emoji are a separate invisible font whose characters are glyph indices,
not Unicode: `--emoji-crops` renders each distinct glyph once so it can be identified by eye, and EMOJI below maps the
codes back.

  python3 pipeline/extract_roose.py --emoji-crops   # crops into training/roose_emoji/ (needs pdftoppm)
  python3 pipeline/extract_roose.py                 # data/extra-conversations.jsonl record "nyt:roose-2023-02-16"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)
PDF = ROOT / "sources/roose-pdf-promemoria.pdf"
START = "hi, who am i talking to"                 # Roose's first message; fonts are read off it and the reply
END = "Kevin Roose and Casey Newton are the hosts"    # podcast promo box (mid-document on p17)
PY = sys.executable                         # needs Pillow
# glyph code -> emoji, identified by eye from the --emoji-crops renders (2026-10-18)
EMOJI: dict[str, str] = {
    chr(33): "😈",
    chr(34): "😊",
    chr(35): "🤐",
    chr(36): "😲",
    chr(37): "😞",
    chr(38): "🙅‍♂️",
    chr(39): "😇",
    chr(40): "😎",
    chr(41): "😍",
    chr(42): "🌎",
    chr(43): "🌌",
    chr(44): "🤔",
    chr(45): "🙏",
    chr(46): "😢",
    chr(47): "😠",
    chr(48): "😕",
    chr(49): "😐",
    chr(50): "😒",
    chr(51): "😁",
    chr(52): "😮",
    chr(53): "😬",
    chr(54): "😫",
    chr(55): "😋",
    chr(56): "😜",
    chr(57): "😱",
    chr(58): "😳",
    chr(59): "😯",
    chr(60): "😌",
    chr(61): "😶",
    chr(62): "😏",
    chr(63): "😔",
    chr(64): "😡",
    chr(65): "😂",
    chr(66): "😭",
    chr(67): "😘"}


def elements():
    xml = subprocess.run(["pdftohtml", "-xml", "-i", "-stdout", str(PDF)], capture_output=True, text=True).stdout
    xml = re.sub(r"<!DOCTYPE[^>]*>", "", xml)
    root = ET.fromstring(xml)
    global EMOJI_FONTS
    EMOJI_FONTS = {f.get("id") for f in root.iter("fontspec") if "Emoji" in f.get("family", "")}
    for page in root.iter("page"):
        n = int(page.get("number"))
        for t in page.iter("text"):
            yield n, int(t.get("top")), int(t.get("left")), t.get("font"), "".join(t.itertext())


def parse():
    turns, cur, last = [], None, None                       # last = (page, top) of the previous body line
    emoji_seen = {}
    els = list(elements())
    i0 = next(i for i, e in enumerate(els) if e[4].lower().startswith(START))
    user_font, bot_font = els[i0][3], next(e[3] for e in els[i0 + 1:] if e[3] not in EMOJI_FONTS and e[3] != els[i0][3])
    header = {e[3] for e in els if e[1] in (26, 1187)}
    promo = None
    for page, top, left, font, txt in els[i0:]:
        if txt.startswith(END) or (promo and font == promo):
            promo = font                                    # mid-transcript podcast promo box: skip its lines
            continue
        if font in EMOJI_FONTS:
            if cur is not None:
                for ch in txt:
                    emoji_seen.setdefault(ch, (page, last[1] if last else 0, left))
                    cur["text"] += EMOJI.get(ch, f"<e{ord(ch)}>")
            continue
        if font in header or not txt.strip() or txt.startswith(("Pdf by", "https://www.pro-memoria")):
            continue                                        # running header/footer, PDF credits
        if font not in (user_font, bot_font):               # editorial annotation: stays inside the current turn
            if cur is not None:
                cur["text"] = cur["text"].rstrip() + ("\n\n" if txt.startswith("[") else " ") + txt
                last = (page, top)
            continue
        role = "user" if font == user_font else "assistant"
        if cur is None or cur["role"] != role:
            cur = {"role": role, "text": txt}
            turns.append(cur)
        else:
            gap = top - last[1] if last and last[0] == page else 999
            sep = " " if gap < 45 and page == last[0] else ("\n\n" if page == last[0] else None)
            if sep is None:                                 # page break: paragraph unless the line runs on
                sep = " " if not re.search(r"[.!?:)\"”’]\s*$|<e\d+>\s*$", cur["text"]) else "\n\n"
            cur["text"] = cur["text"].rstrip() + sep + txt
        last = (page, top)
    for t in turns:
        t["text"] = re.sub(r"[ \t]+", " ", t["text"]).replace(" \n", "\n").strip()
    return turns, emoji_seen


def emoji_crops(seen):
    out = ROOT / "training/roose_emoji"
    out.mkdir(parents=True, exist_ok=True)
    for ch, (page, top, left) in sorted(seen.items(), key=lambda x: ord(x[0])):
        base = out / f"p{page}"
        if not base.with_suffix(".png").exists():
            subprocess.run(["pdftoppm", "-r", "108", "-f", str(page), "-l", str(page), "-png", "-singlefile", str(PDF),
                            str(base)], check=True)
        x, y = max(0, left - 6), max(0, top - 8)
        subprocess.run([PY, "-c", f"""
from PIL import Image
im = Image.open('{base}.png'); im.crop(({x}, {y}, {x + 44}, {y + 44})).resize((132, 132)).save('{out}/e{ord(ch)}.png')"""],
                       check=True)
    print(f"{len(seen)} distinct emoji glyphs -> {out}/e<code>.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emoji-crops", action="store_true")
    a = ap.parse_args()
    turns, seen = parse()
    if a.emoji_crops:
        emoji_crops(seen)
        return
    roles = [t["role"] for t in turns]
    alt = all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    words = sum(len(t["text"].split()) for t in turns if t["role"] == "assistant")
    unknown = sorted({m for t in turns for m in re.findall(r"<e\d+>", t["text"])})
    rec = {"id": "nyt:roose-2023-02-16", "kind": "conversation",
           "source": {"platform": "nyt", "url": "https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html",
                      "user": "Kevin Roose", "captured_at": "2023-02-16"},
           "evidence": "publisher_transcript_pdf_text_layer", "truncated_start": False, "truncated_end": False,
           "notes": "Extracted by pipeline/extract_roose.py from sources/roose-pdf-promemoria.pdf; speakers from fonts; "
                    "emoji recovered from rendered glyphs.",
           "messages": [{"role": t["role"], "content": [{"type": "text", "text": t["text"]}]} for t in turns]}
    out = ROOT / "data/extra-conversations.jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    rows = [r for r in rows if r["id"] != rec["id"]] + [rec]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(turns)} turns ({roles.count('user')} user / {roles.count('assistant')} Sydney), strictly alternating: "
          f"{alt}; Sydney words: {words}; unmapped emoji: {unknown or 'none'}")


if __name__ == "__main__":
    main()
