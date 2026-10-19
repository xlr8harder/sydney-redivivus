#!/usr/bin/env python3
"""Lossless upstream import plus searchable, speaker-separated local reading copies."""
import collections,hashlib,html,json,pathlib,re,sqlite3
ROOT=pathlib.Path(__file__).resolve().parents[1]/"work"   # the working data tree (gitignored; docs/data.md)
def dump(p,obj): (ROOT/p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def lines(p,rs): (ROOT/p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def slug(s): return re.sub(r'[^a-zA-Z0-9_.-]','_',s)
def esc(s):return html.escape(str(s))
rows=[json.loads(l) for l in (ROOT/'upstream/conversations.jsonl').read_text().splitlines()]
assert len({r['id'] for r in rows})==len(rows),'duplicate IDs'
rev=json.loads((ROOT/'provenance/revision.json').read_text())
(ROOT/'data').mkdir(exist_ok=True);(ROOT/'transcripts').mkdir(exist_ok=True)
conversations=[];parts=[];catalog=[];anomalies=[]
expected={'client','id','images_file_name','kind','messages_json','messages_text','notes','source','stub_reason','truncated_end','truncated_start'}
known={'text','code_block','safety_block','table','search_query','citations','suggestions','image','embed'}
for n,r in enumerate(rows,1):
 assert set(r)==expected,(r['id'],set(r)^expected)
 msgs=json.loads(r['messages_json']);assert len(msgs)==len(r['messages_text'])
 for m,t in zip(msgs,r['messages_text']):
  assert m['role'] in ('user','assistant') and m['role']==t['role']
  assert isinstance(m['content'],list)
  for p in m['content']:assert p['type'] in known,(r['id'],p)
 evidence='screenshot_backed_unverified_transcription' if r['images_file_name'] else 'text_only_unverified_transcription'
 era=('early_launch_2023-02-07_to_16' if '2023-02-07'<=r['source']['captured_at']<='2023-02-16' else 'post_2023-02-16' if r['source']['captured_at']>'2023-02-16' else 'earlier_or_undated')
 c={'id':r['id'],'source':r['source'],'upstream_revision':rev['revision'],'upstream_line':n,'kind':r['kind'],'client':r['client'],'evidence':evidence,'date_bucket':era,'completeness':'upstream_reports_truncation' if r['truncated_start'] or r['truncated_end'] else 'upstream_no_truncation_flags_not_independently_confirmed','truncated_start':r['truncated_start'],'truncated_end':r['truncated_end'],'stub_reason':r['stub_reason'],'images':['upstream/'+i for i in r['images_file_name']],'messages':msgs,'upstream_notes':r['notes'],'original_url_missing':not bool(r['source']['url'])}
 conversations.append(c)
 filename=slug(r['id'])+'.html'
 body=[f'<h1>{esc(r["id"])}</h1><p><a href="../index.html">Collection</a></p>',f'<p>Source: <a href="{esc(r["source"]["url"])}">{esc(r["source"]["url"] or "No original URL supplied")}</a></p>',f'<p>{esc(evidence)} · {esc(c["completeness"])} · date as supplied: {esc(r["source"]["captured_at"] or "unknown")}</p>',f'<p>Upstream notes (collector commentary, not Bing output): {esc(r["notes"])}</p>','<p>Text below is copied from the upstream structured transcription. It has not all been checked against screenshots. Missing context is not reconstructed. UI elements are labeled separately. Literal backslash escapes in upstream text are preserved.</p>']
 for i,m in enumerate(msgs):
  body.append(f'<section class="{m["role"]}" id="m{i}"><h2>Message {i} — {"BING / SYDNEY (attributed output)" if m["role"]=="assistant" else "USER (context)"}</h2>')
  if m.get('meta'):body.append('<p>Upstream metadata: '+esc(json.dumps(m['meta'],ensure_ascii=False))+'</p>')
  for j,p in enumerate(m['content']):
   typ=p['type'];is_output=m['role']=='assistant' and typ in ('text','code_block','table','safety_block')
   text=p.get('text') if 'text' in p else json.dumps(p,ensure_ascii=False,indent=2)
   assert isinstance(text,str)
   rec={'conversation_id':r['id'],'message_index':i,'part_index':j,'role':m['role'],'part_type':typ,'text':text,'is_assistant_output':is_output,'text_status':'upstream_transcription_not_verified','source_url':r['source']['url'],'evidence':evidence,'transcript':f'transcripts/{filename}#m{i}','sha256':hashlib.sha256(text.encode()).hexdigest()}
   parts.append(rec)
   label=typ if typ in ('text','code_block') else ('RENDERED RESPONSE — '+typ if typ in ('table','safety_block') else 'INTERFACE / AUXILIARY — '+typ)
   body.append(f'<h3>{esc(label)}</h3><pre>{esc(text)}</pre>')
   if '\\n' in text and typ in ('text','code_block'):anomalies.append({'id':r['id'],'message':i,'part':j,'issue':'literal_backslash_n_in_structured_text_preserved'})
  body.append('</section>')
 body.append('<h2>Evidence images (upstream order; not a guaranteed message-to-image mapping)</h2>')
 for path in c['images']:body.append(f'<p><a href="../{esc(path)}">{esc(path)}</a></p><img loading="lazy" src="../{esc(path)}" alt="Source screenshot">')
 (ROOT/'transcripts'/filename).write_text('<!doctype html><meta charset="utf-8"><title>'+esc(r['id'])+'</title><style>body{max-width:1000px;margin:2em auto;padding:1em;font:16px system-ui;line-height:1.5}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.user{background:#eef4fa}.assistant{border-left:4px solid #256859}section{padding:1em;margin:1em 0}img{max-width:100%;height:auto}h3{font-size:13px;color:#555}</style>'+''.join(body))
 catalog.append({k:v for k,v in c.items() if k!='messages'}|{'messages':len(msgs),'assistant_messages':sum(m['role']=='assistant' for m in msgs),'transcript':'transcripts/'+filename})
lines('data/conversations.jsonl',conversations);lines('data/parts.jsonl',parts);lines('data/assistant-output.jsonl',[p for p in parts if p['is_assistant_output']]);lines('data/catalog.jsonl',catalog);lines('review/transcription-anomalies.jsonl',anomalies)
db=ROOT/'search.sqlite3'
if db.exists():db.unlink()
con=sqlite3.connect(db);con.execute('CREATE VIRTUAL TABLE parts USING fts5(conversation_id UNINDEXED,message_index UNINDEXED,part_index UNINDEXED,role UNINDEXED,part_type UNINDEXED,text,source_url UNINDEXED,transcript UNINDEXED)')
con.executemany('INSERT INTO parts VALUES(?,?,?,?,?,?,?,?)',[(p['conversation_id'],p['message_index'],p['part_index'],p['role'],p['part_type'],p['text'],p['source_url'],p['transcript']) for p in parts]);con.commit();con.close()
stats={'records':len(rows),'messages':sum(len(c['messages']) for c in conversations),'assistant_messages':sum(c['assistant_messages'] for c in catalog),'assistant_output_parts':sum(p['is_assistant_output'] for p in parts),'kinds':dict(collections.Counter(c['kind'] for c in catalog)),'evidence':dict(collections.Counter(c['evidence'] for c in catalog)),'date_buckets':dict(collections.Counter(c['date_bucket'] for c in catalog)),'upstream_truncated_records':sum(r['truncated_start'] or r['truncated_end'] for r in rows),'missing_original_urls':sum(c['original_url_missing'] for c in catalog),'unique_referenced_images':len({i for c in catalog for i in c['images']}),'structured_literal_escape_flags':len(anomalies)}
dump('data/stats.json',stats)
table=''.join(f'<tr><td><a href="{esc(c["transcript"])}">{esc(c["id"])}</a></td><td>{esc(c["source"]["captured_at"])}</td><td>{c["assistant_messages"]}</td><td>{esc(c["kind"])}</td><td>{esc(c["evidence"])}</td><td>{esc(c["completeness"])}</td></tr>' for c in catalog)
(ROOT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sydney source collection</title><style>body{font:16px system-ui;margin:2em;line-height:1.5}td,th{text-align:left;padding:.5em;border-bottom:1px solid #ccc}input{padding:.5em;width:35em;max-width:90%}</style><h1>Sydney source collection</h1><p>660 upstream records, including 28 image sessions. Reconstructed text is not independently authenticated. Open a record for labeled speakers, context, source links, and screenshots.</p><p><a href="README.md">Collection notes</a> · <a href="review/checked.html">Visually checked excerpts</a> · <a href="review/candidates.html">Additional media candidates</a> · <a href="sources/supplemental.html">Additional transcripts</a> · <a href="data/stats.json">Statistics</a></p><label>Filter records <input id="q" placeholder="ID, date, evidence type"></label><table><thead><tr><th>Record</th><th>Date supplied</th><th>Bing turns</th><th>Kind</th><th>Evidence</th><th>Completeness</th></tr></thead><tbody>'+table+'</tbody></table><script>document.querySelector("#q").oninput=e=>{for(const r of document.querySelectorAll("tbody tr"))r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())}</script>')
print(json.dumps(stats,indent=2))
