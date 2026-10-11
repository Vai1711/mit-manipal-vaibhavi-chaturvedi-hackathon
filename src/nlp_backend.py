"""Compact CPU NLP with an explicit, portable replay-prediction cache."""
import hashlib,json,time,re,os
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DESCRIPTIONS={
 'Macroeconomic':'central-bank interest-rate decisions, inflation, recession or macroeconomic policy',
 'Credit Event':'a debt default, missed debt payment, insolvency, bankruptcy or borrower credit-rating change',
 'Geopolitical':'international war, military invasion, sanctions, trade embargo or blockade',
 'Merger/Acquisition':'a corporate merger, takeover or acquisition of a business',
 'Product Launch':'the launch, introduction or unveiling of a new commercial product or service',
 'Other/Uncertain':'earnings, personnel changes, stock commentary or another topic outside these event categories'}


def clean_text(text):
    return re.sub(r'\s+',' ',re.sub(r'https?://\S+','',text)).strip()


PROMPT_HASH=hashlib.sha256(json.dumps(DESCRIPTIONS,sort_keys=True).encode()).hexdigest()


def text_key(text):return hashlib.sha256(clean_text(text).encode()).hexdigest()


class CompactNLP:
    def __init__(self, load_sentiment=True):
        os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
        import torch
        torch.set_num_threads(2)
        from transformers import AutoTokenizer,AutoModelForSequenceClassification,pipeline
        specs=json.loads((ROOT/'data/model_specs.json').read_text(encoding='utf-8'))
        self.specs=specs
        self.sentiment=self._load(specs['sentiment'],'text-classification',pipeline,AutoTokenizer,AutoModelForSequenceClassification) if load_sentiment else None
        self.events=self._load(specs['events'],'zero-shot-classification',pipeline,AutoTokenizer,AutoModelForSequenceClassification)

    @staticmethod
    def _load(spec,task,pipeline,tokenizer,model):
        folder=ROOT/spec['local_directory']
        if not (folder/'model.safetensors').exists():
            raise RuntimeError('Local compact models are not installed. Run python scripts/download_models.py after installing requirements-nlp.txt.')
        tok=tokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        extra={'reference_compile':False} if 'ModernBERT' in spec['repository'] else {}
        net=model.from_pretrained(folder,local_files_only=True,trust_remote_code=False,attn_implementation='eager',**extra)
        return pipeline(task,model=net,tokenizer=tok,device=-1)

    def predict(self,text):
        start=time.perf_counter();clean=clean_text(text)
        predictions=self.sentiment(clean,top_k=None,truncation=True,max_length=256) if self.sentiment else []
        probabilities={p['label'].lower():float(p['score']) for p in predictions}
        # Some classifiers use LABEL_n; pinned config currently uses descriptive labels.
        aliases={'label_0':'negative','label_1':'neutral','label_2':'positive'}
        probabilities={aliases.get(k,k):v for k,v in probabilities.items()}
        result=self.events(clean,candidate_labels=list(DESCRIPTIONS.values()),hypothesis_template='This text describes {}.',multi_label=False)
        reverse={v:k for k,v in DESCRIPTIONS.items()}
        return {'sentiment_probabilities':probabilities or None,'model_event_classification':reverse[result['labels'][0]],
                'event_model_score':float(result['scores'][0]),'event_top_two_margin':float(result['scores'][0]-result['scores'][1]),
                'model_event_scores':{reverse[k]:float(v) for k,v in zip(result['labels'],result['scores'])},
                'inference_ms':(time.perf_counter()-start)*1000,'backend':'compact_models',
                'prompt_hash':PROMPT_HASH,'model_revisions':{k:v['revision'] for k,v in self.specs.items()}}


@lru_cache(maxsize=1)
def shared_compact_model():return CompactNLP(load_sentiment=False)


class PredictionProvider:
    def __init__(self,mode='cached'):
        if mode not in ('cached','hybrid','compact'):raise ValueError('Unknown prediction mode')
        self.mode=mode;self.model=None
        file=ROOT/'data/prediction_cache.json'
        self.cache=json.loads(file.read_text(encoding='utf-8')).get('predictions',{}) if file.exists() else {}

    @lru_cache(maxsize=256)
    def predict(self,text):
        if self.mode=='cached':
            cached=self.cache.get(text_key(text))
            expected=json.loads((ROOT/'data/model_specs.json').read_text(encoding='utf-8'))
            revisions={k:v['revision'] for k,v in expected.items()}
            valid=cached and cached.get('prompt_hash')==PROMPT_HASH and cached.get('model_revisions')==revisions
            return {**cached,'backend':'recorded_compact_models','cache_hit':True} if valid else None
        if self.model is None:self.model=shared_compact_model()
        return {**self.model.predict(text),'cache_hit':False}
