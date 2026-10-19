#!/usr/bin/env python3
from acquire import ROOT
import collections,hashlib,json
from PIL import Image
errors=[];tree=json.loads((ROOT/'provenance/hf-tree.json').read_text());checked=0;images=0
for e in tree:
 if e['type']!='file':continue
 p=ROOT/'upstream'/e['path']
 if not p.is_file():errors.append('Missing '+str(p));continue
 b=p.read_bytes()
 if len(b)!=e['size']:errors.append('Size '+str(p))
 if 'lfs' in e and hashlib.sha256(b).hexdigest()!=e['lfs']['oid']:errors.append('LFS sha256 '+str(p))
 if 'lfs' not in e and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=e['oid']:errors.append('Git object hash '+str(p))
 checked+=1
 if e['path'].startswith('images/'):
  try:
   with Image.open(p) as im:im.verify()
   images+=1
  except Exception as ex:errors.append(str(ex))
rows=[json.loads(l) for l in (ROOT/'data/conversations.jsonl').read_text().splitlines()]
raw=[json.loads(l) for l in (ROOT/'upstream/conversations.jsonl').read_text().splitlines()]
for a,b in zip(rows,raw):
 if a['messages']!=json.loads(b['messages_json']):errors.append('Message parity '+a['id'])
 for i in a['images']:
  if not (ROOT/i).exists():errors.append('Reference missing '+i)
for logfile in ['hf-downloads.jsonl','discovery-downloads.jsonl','supplement-downloads.jsonl','extra-media-downloads.jsonl','roose-extra.jsonl']:
 for line in (ROOT/'provenance'/logfile).read_text().splitlines():
  r=json.loads(line)
  if 'sha256' in r and hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']:errors.append('Receipt hash '+r['path'])
for line in (ROOT/'review/verified-snippets.jsonl').read_text().splitlines():
 r=json.loads(line)
 if hashlib.sha256((ROOT/r['image']).read_bytes()).hexdigest()!=r['image_sha256']:errors.append('Checked snippet image hash '+r['id'])
 if r.get('message_index') is not None:
  c=next(x for x in rows if x['id']==r['conversation_id'])
  if not any(r['exact_sydney_text'] in p.get('text','') for p in c['messages'][r['message_index']]['content']):errors.append('Snippet absent from attributed message '+r['id'])
result={'upstream_files_checked':checked,'upstream_images_decoded':images,'normalized_records_compared':len(rows),'errors':errors}
(ROOT/'review/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if errors:raise SystemExit(1)
