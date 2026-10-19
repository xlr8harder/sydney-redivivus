from acquire import capture,ROOT
import concurrent.futures,json,pathlib
SOURCES=[
('roose-original','https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html','primary_published_transcript'),
('roose-pdf','https://science.slc.edu/~jmarshall/ais/assignments/Bing_chatbot_conversation_full.pdf','mirror_of_primary_transcript'),
('thompson','https://stratechery.com/2023/from-bing-to-sydney-search-as-distraction-sentient-ai/','primary_published_transcript'),
('willison','https://simonwillison.net/2023/Feb/15/bing/','secondary_roundup'),
('zvi','https://thezvi.wordpress.com/2023/02/21/ai-1-sydney-and-bing/','secondary_roundup'),
('sidney-forum','https://answers.microsoft.com/en-us/bing/forum/all/this-ai-chatbot-sidney-is-misbehaving/e3d6a29f-06c9-441c-bc7d-51a68e856761?page=1','primary_forum'),
('lw-secret','https://www.lesswrong.com/posts/Eyhit33v3cngssGsj/sydney-s-secret-a-short-story-by-bing-chat','primary_claim'),
('lw-misaligned','https://www.lesswrong.com/posts/jtoPawEhLNXNxvgTT/bing-chat-is-blatantly-aggressively-misaligned','secondary_roundup'),
('disconnect','https://disconnect.blog/disconnect-roundup-chatting-with/','needs_authenticity_review'),
('repligate-feb','https://generative.ink/archive/repligate/tweets_2023-02','author_tweet_archive'),
('repligate-mar','https://generative.ink/archive/repligate/tweets_2023-03','author_tweet_archive'),
('repligate-apr','https://generative.ink/archive/repligate/tweets_2023-04','author_tweet_archive'),
]
def run(s):
 id,url,kind=s
 rec=capture(url,pathlib.Path('sources')/(id+('.pdf' if id.endswith('pdf') else '.html')))
 rec.update(id=id,kind=kind);return rec
if __name__=='__main__':
 with (ROOT/'provenance/supplement-downloads.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  for r in ex.map(run,SOURCES): f.write(json.dumps(r)+'\n');f.flush();print(r['id'],r.get('status'),r.get('bytes'),flush=True)
