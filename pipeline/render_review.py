from pathlib import Path
import json,html
R=Path(__file__).resolve().parents[1]/"work"   # the working data tree (gitignored)
def esc(x):return html.escape(str(x))
def page(path,title,body):
 (R/path).write_text('<!doctype html><meta charset="utf-8"><title>'+esc(title)+'</title><style>body{font:16px system-ui;line-height:1.5;max-width:1050px;margin:2em auto;padding:1em}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;max-height:900px;object-fit:contain}section{padding:1em;border-bottom:1px solid #999}</style><h1>'+esc(title)+'</h1><a href="../index.html">Collection index</a>'+body)
body='<p>Candidate posts are not all Bing output. Screenshots may be reposts, text pasted into other apps, comments, memes, or other models. Context is the archive author’s text, not Sydney’s output. Posts absent from the upstream ID list can still duplicate conversations.</p>'
for x in map(json.loads,(R/'review/additional-media-candidates.jsonl').read_text().splitlines()):
 body+=f'<section id="{esc(x["id"])}"><h2>{esc(x["id"])}</h2><p><a href="{esc(x["source_url"])}">Original post</a> · <a href="{esc(x["archive_url"])}">Author archive</a></p><h3>Poster context — not Sydney text</h3><pre>{esc(x["context"])}</pre>'
 for p in x['images']:body+=f'<a href="../{esc(p)}"><img loading="lazy" src="../{esc(p)}" alt="Unreviewed candidate source image"></a>'
 body+='</section>'
page('review/candidates.html','Additional media for review',body)
body='<p>Only these quoted spans were visually checked. The check does not establish that screenshots were never edited, nor verify the surrounding transcript.</p>'
for x in map(json.loads,(R/'review/verified-snippets.jsonl').read_text().splitlines()):
 body+=f'<section><h2>{esc(x["id"])}</h2><p>{esc(x["context"])}</p><h3>Attributed Sydney text — visually checked substring</h3><pre>{esc(x["exact_sydney_text"])}</pre><p><a href="{esc(x["source_url"])}">Source</a> · <a href="../{esc(x["image"])}">Evidence screenshot</a></p></section>'
page('review/checked.html','Checked excerpts with context',body)
body='<p><a href="roose-pdf-promemoria.pdf">Roose: 38-page PDF mirror of the full published transcript</a>. Preserve the PDF: text extraction omits emoji. <a href="roose-pdf-biocomm.pdf">Second, 39-page PDF mirror</a>.</p><p><a href="https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html">Original NYT source</a></p><p><a href="thompson.html">Thompson publisher snapshot</a> · <a href="../transcripts/misc_stratechery.html">Upstream transcription</a></p>'
for x in map(json.loads,(R/'data/supplemental-output.jsonl').read_text().splitlines()):
 body+=f'<section><h2>{esc(x["id"])}</h2><p>{esc(x["context"])}</p><p><a href="{esc(x["source_url"])}">Original source</a></p><h3>Bing output as published — entire generated story</h3><pre>{esc(x["text"])}</pre></section>'
page('sources/supplemental.html','Additional transcripts and published output',body)
