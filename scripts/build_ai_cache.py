"""Record actual compact-model outputs for low-load replay. No invented outputs."""
import sys,json,time,platform,hashlib,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.nlp_backend import CompactNLP,text_key,PROMPT_HASH


def main():
    start=time.perf_counter();model=CompactNLP();load=time.perf_counter()-start
    cachefile=ROOT/'data/prediction_cache.json'
    cache=json.loads(cachefile.read_text(encoding='utf-8')) if cachefile.exists() else {'predictions':{}}
    specs=json.loads((ROOT/'data/model_specs.json').read_text(encoding='utf-8'))
    revisions={k:v['revision'] for k,v in specs.items()}
    names=['public_replay.json','public_feed_snapshot.json','news_feed.json','social_feed.json','public_eval.json','event_eval.json','challenge_set.json']
    total=[]
    for name in names:total.extend(json.loads((ROOT/'data'/name).read_text(encoding='utf-8')))
    for i,item in enumerate(total):
        key=text_key(item['text']);old=cache['predictions'].get(key)
        if not old or old.get('model_revisions')!=revisions or old.get('prompt_hash')!=PROMPT_HASH:
            cache['predictions'][key]=model.predict(item['text'])
        if i%10==0:
            cachefile.write_text(json.dumps(cache,indent=2),encoding='utf-8');print('Recorded',i+1,'/',len(total),flush=True)
    cache['provenance']={'generated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'host':platform.platform(),
        'model_load_seconds':load,'threads':2,'torch_compile_disabled':True,'purpose':'Recorded model inference, not new live inference. Metadata and risk controls recompute on replay.',
        'max_resident_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'prompt_hash':PROMPT_HASH,'model_revisions':revisions}
    cachefile.write_text(json.dumps(cache,indent=2),encoding='utf-8');print('Recorded AI cache complete')

if __name__=='__main__':main()
