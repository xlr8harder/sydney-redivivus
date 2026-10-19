#!/usr/bin/env python3
"""Capture a pinned upstream dataset, retaining bytes and per-request receipts."""
import concurrent.futures, datetime, hashlib, json, pathlib, time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]/"work"   # the working data tree (gitignored; docs/data.md)
REPO='david-ar/bing-chat-logs'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def capture(url, path, expected=None):
    path=ROOT/path; path.parent.mkdir(parents=True,exist_ok=True)
    receipt={'url':url,'path':str(path.relative_to(ROOT)), 'started_at':now()}
    for attempt in range(3):
        try:
            r=requests.get(url, timeout=90)
            receipt.update(status=r.status_code,content_type=r.headers.get('content-type'),etag=r.headers.get('etag'))
            if not r.ok:
                failure=path.with_name(path.name+'.http-error')
                failure.write_bytes(r.content)
                receipt['error_body_path']=str(failure.relative_to(ROOT))
            r.raise_for_status()
            b=r.content; h=hashlib.sha256(b).hexdigest()
            if expected and h!=expected: raise ValueError(f'hash mismatch {path}')
            path.write_bytes(b); receipt.pop('error',None); receipt.update(bytes=len(b),sha256=h,finished_at=now()); return receipt
        except Exception as e:
            receipt['error']=str(e)
            if attempt<2: time.sleep(2*(attempt+1))
    return receipt
if __name__=='__main__':
    pin=ROOT/'provenance/revision.json'
    revision=json.loads(pin.read_text())['revision'] if pin.exists() else 'main'
    meta=requests.get(f'https://huggingface.co/api/datasets/{REPO}/revision/{revision}',timeout=60);meta.raise_for_status()
    (ROOT/'provenance/hf-api.json').write_bytes(meta.content)
    revision=meta.json()['sha']
    (ROOT/'provenance/revision.json').write_text(json.dumps({'repo':REPO,'revision':revision,'retrieved_at':now()},indent=2)+'\n')
    entries=meta.json()['siblings']
    with (ROOT/'provenance/hf-downloads.jsonl').open('w') as log, concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        jobs=[ex.submit(capture,f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{e["rfilename"]}',pathlib.Path('upstream')/e['rfilename']) for e in entries]
        for i,f in enumerate(concurrent.futures.as_completed(jobs),1):
            result=f.result();log.write(json.dumps(result)+'\n');log.flush()
            if i%100==0 or 'error' in result: print(i,result['path'],result.get('error','ok'),flush=True)
    print('DONE',revision,len(entries),flush=True)
