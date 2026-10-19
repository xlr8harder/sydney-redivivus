from acquire import ROOT,capture
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pathlib,json,re,concurrent.futures
jobs=[];candidates=[]
for month in ['feb','mar','apr']:
 p=ROOT/f'sources/repligate-{month}.html'
 if not p.exists():continue
 base=f'https://generative.ink/archive/repligate/tweets_2023-'+{'feb':'02','mar':'03','apr':'04'}[month]
 s=BeautifulSoup(p.read_bytes(),'html.parser')
 for t in s.select('div.tweet'):
  link=t.select_one('.tweet-time a');images=t.select('img')
  if not link or not images:continue
  text='\n'.join(x.get_text(' ',strip=True) for x in t.find_all('p',recursive=False))
  if not re.search(r'bing|sydney|waluigi|venom',text,re.I):continue
  tid=link['href'].split('/')[-1];assets=[]
  for i,img in enumerate(images):
   url=urljoin(base,img.get('data-src') or img['src']);path=pathlib.Path('sources/repligate-media')/(tid+'-'+str(i)+pathlib.Path(url).suffix)
   assets.append(str(path));jobs.append((url,path))
  candidates.append({'id':'repligate-archive:'+tid,'source_url':link['href'],'archive_url':base,'context':text,'images':assets,'status':'unreviewed_media_candidate','exact_sydney_text':None})
s=BeautifulSoup((ROOT/'sources/thompson.html').read_bytes(),'html.parser')
for a in s.select('a[href]'):
 if a.get_text(strip=True).lower()=='screenshot':
  url=urljoin('https://stratechery.com',a['href']);path=pathlib.Path('sources/thompson-screenshots')/pathlib.Path(url).name;jobs.append((url,path))
(ROOT/'review/additional-media-candidates.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in candidates))
with (ROOT/'provenance/extra-media-downloads.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for rec in ex.map(lambda j:capture(*j),dict.fromkeys(jobs)):
  f.write(json.dumps(rec)+'\n');f.flush()
print('candidate posts',len(candidates),'media requests',len(set(jobs)))
