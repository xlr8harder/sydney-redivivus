"""Collect every remaining source of Sydney output into sources/collected/, with provenance.

  A. images embedded in saved pages (LessWrong "misaligned", Zvi, Willison, janus's archive pages) + Thompson's
     local screenshots
  B. discovery-lead tweets not already in the dataset, via the public fxtwitter API (text, date, media, quoted tweet)
  C. discovery-lead web pages (raw HTML; failures such as paywalls are recorded, not worked around); images embedded
     in them are queued like A
Reddit's API refuses scripted access, so reddit leads are only listed. Then every collected image gets a 64-bit
difference hash; images within Hamming distance 6 of an upstream screenshot or an already-transcribed janus image are
marked duplicates. Output: sources/collected/manifest.json (new images grouped by post/page) and downloads.jsonl.

  python3 pipeline/collect_sources.py
"""
from __future__ import annotations

import hashlib
import html as H
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "work"          # the working data tree (gitignored; docs/data.md)
OUT = ROOT / "sources/collected"
UA = {"User-Agent": "Mozilla/5.0 (sydney research archive; local, non-redistributed)"}
SKIP_IMG = re.compile(r"logo|avatar|profile_images|emoji|favicon|icon|header|gravatar|badge|/static/img/|button|spinner",
                      re.I)


def get(url: str, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers.get("content-type", ""), r.read()


def log(entry):
    with (OUT / "downloads.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def fetch(url: str, path: Path, done: set) -> bool:
    if url in done and path.exists():
        return True
    try:
        status, ctype, body = get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        log({"url": url, "path": str(path.relative_to(ROOT)), "status": status, "content_type": ctype,
             "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        time.sleep(0.25)
        return True
    except Exception as e:
        log({"url": url, "path": str(path.relative_to(ROOT)), "error": f"{type(e).__name__}: {str(e)[:120]}"})
        return False


def page_images(html: str, base: str) -> list[str]:
    urls = []
    for m in re.finditer(r'<img[^>]+?(?:data-src|src)="([^"]+)"', html):
        u = urllib.parse.urljoin(base, H.unescape(m.group(1)))
        if u.startswith("http") and not SKIP_IMG.search(u):
            urls.append(u)
    return list(dict.fromkeys(urls))


def dhash(path: Path):
    from PIL import Image
    try:
        im = Image.open(path).convert("L").resize((9, 8))
    except Exception:
        return None
    px = list(im.getdata())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if (OUT / "downloads.jsonl").exists():
        done = {json.loads(l)["url"] for l in (OUT / "downloads.jsonl").read_text().splitlines() if "error" not in l}
    groups = {}                                               # group id -> {"source", "context", "images": [paths]}

    # A. saved pages
    pages = {"lw-misaligned": "https://www.lesswrong.com/posts/jtoPawEhLNXNxvgTT/",
             "zvi": "https://thezvi.substack.com/", "willison": "https://simonwillison.net/2023/Feb/15/bing/",
             "repligate-feb": "https://generative.ink/archive/repligate/",      # media live at
             "repligate-mar": "https://generative.ink/archive/repligate/",      # .../repligate/tweets_media/,
             "repligate-apr": "https://generative.ink/archive/repligate/"}      # not under the month page
    have = {p.name.split("-")[0] for p in (ROOT / "sources/repligate-media").glob("*")}
    for name, base in pages.items():
        html = (ROOT / f"sources/{name}.html").read_text(errors="ignore")
        if name.startswith("repligate"):                      # group by tweet; skip tweets already handled
            for u in page_images(html, base):
                tid = re.search(r"tweets_media/(\d+)-", u)
                if not tid or tid.group(1) in have:
                    continue
                p = OUT / "repligate" / Path(urllib.parse.urlparse(u).path).name
                if fetch(u, p, done):
                    g = groups.setdefault(f"janus:{tid.group(1)}", {"source": f"https://twitter.com/repligate/status/{tid.group(1)}",
                                                                    "context": "", "images": []})
                    g["images"].append(str(p.relative_to(ROOT)))
        else:
            for i, u in enumerate(page_images(html, base)):
                ext = ".png" if ".png" in u.lower() else ".jpg"
                p = OUT / name / f"{i:03d}{ext}"
                if fetch(u, p, done):
                    groups[f"{name}:{i:03d}"] = {"source": base, "context": f"image {i} of saved page {name}",
                                                 "images": [str(p.relative_to(ROOT))]}
    for p in sorted((ROOT / "sources/thompson-screenshots").glob("*")):
        groups[f"thompson:{p.stem}"] = {"source": "https://stratechery.com/2023/from-bing-to-sydney-search-as-distraction-sentient-ai/",
                                        "context": "Ben Thompson's screenshot", "images": [str(p.relative_to(ROOT))]}

    # B/C. discovery leads
    up = {json.loads(l)["id"] for l in (ROOT / "data/conversations.jsonl").read_text().splitlines()}
    leads = list(dict.fromkeys(json.loads(l)["url"] for l in (ROOT / "review/discovery-leads.jsonl").read_text().splitlines()))
    web_status, reddit = [], []
    for u in leads:
        m = re.search(r"(?:twitter|x)\.com/([^/]+)/status/(\d+)", u)
        if m:
            tid = m.group(2)
            if f"twitter:{tid}" in up or f"repligate-archive:{tid}" in up:
                continue
            jp = OUT / "tweets" / f"{tid}.json"
            if not fetch(f"https://api.fxtwitter.com/status/{tid}", jp, done):
                continue
            try:
                tw = json.loads(jp.read_text()).get("tweet") or {}
            except json.JSONDecodeError:
                continue
            imgs = []
            for src in [tw] + ([tw["quote"]] if tw.get("quote") else []):
                for k, med in enumerate(((src.get("media") or {}).get("photos") or [])):
                    p = OUT / "tweets" / f"{src.get('id', tid)}-{k}.jpg"
                    if fetch(med["url"], p, done):
                        imgs.append(str(p.relative_to(ROOT)))
            groups[f"tweet:{tid}"] = {"source": u, "context": f"@{m.group(1)} {tw.get('created_at', '')}: {tw.get('text', '')}"
                                      + (f" [quoting @{tw['quote'].get('author', {}).get('screen_name', '')}: "
                                         f"{tw['quote'].get('text', '')}]" if tw.get("quote") else ""),
                                      "date": tw.get("created_at", ""), "images": imgs}
            continue
        if "reddit.com" in u:
            reddit.append(u)
            continue
        slug = re.sub(r"[^A-Za-z0-9]+", "-", re.sub(r"^https?://(www\.)?", "", u))[:90]
        hp = OUT / "web" / f"{slug}.html"
        ok = fetch(u, hp, done)
        web_status.append({"url": u, "ok": ok, "path": str(hp.relative_to(ROOT)) if ok else None})
        if ok:
            html = hp.read_text(errors="ignore")
            for i, iu in enumerate(page_images(html, u)[:60]):
                ext = ".png" if ".png" in iu.lower() else ".jpg"
                p = OUT / "web" / slug / f"{i:03d}{ext}"
                if fetch(iu, p, done):
                    groups[f"web:{slug}:{i:03d}"] = {"source": u, "context": f"image {i} in {u}",
                                                     "images": [str(p.relative_to(ROOT))]}

    # D. near-duplicate images
    known = {}
    for p in list((ROOT / "upstream/images").rglob("*")) + list((ROOT / "sources/repligate-media").glob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            h = dhash(p)
            if h is not None:
                known[h] = str(p.relative_to(ROOT))
    kh = list(known)
    stats = {"groups": len(groups), "images": 0, "dup_images": 0, "small_or_bad": 0}
    for g in groups.values():
        keep = []
        for ip in g["images"]:
            p = ROOT / ip
            stats["images"] += 1
            if p.stat().st_size < 12000:
                stats["small_or_bad"] += 1
                continue
            h = dhash(p)
            if h is None:
                stats["small_or_bad"] += 1
                continue
            near = next((known[k] for k in kh if bin(k ^ h).count("1") <= 6), None)
            if near:
                stats["dup_images"] += 1
                g.setdefault("duplicates", []).append([ip, near])
                continue
            keep.append(ip)
            kh.append(h)
            known[h] = ip
        g["new_images"] = keep
    (OUT / "manifest.json").write_text(json.dumps({"groups": groups, "web": web_status, "reddit_not_fetched": reddit},
                                                  indent=1, ensure_ascii=False))
    stats["groups_with_new_images"] = sum(bool(g["new_images"]) for g in groups.values())
    stats["new_images"] = sum(len(g["new_images"]) for g in groups.values())
    stats["web_ok"] = sum(w["ok"] for w in web_status)
    stats["web_failed"] = [w["url"] for w in web_status if not w["ok"]]
    stats["reddit_not_fetched"] = len(reddit)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
