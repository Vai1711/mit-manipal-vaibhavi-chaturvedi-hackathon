"""External sentiment validation and explicitly curated event evaluation."""
import sys,json,time,platform,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.risk_engine import RiskEngine,baseline_sentiment_score
from src.pipeline import load_portfolio
from src.learned_sentiment import scorer
from src.nlp_backend import CompactNLP,text_key
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix


def metrics(actual,predicted):
    labels=sorted(set(actual)|set(predicted))
    return {'accuracy':float(accuracy_score(actual,predicted)),'macro_f1':float(f1_score(actual,predicted,average='macro',zero_division=0)),
        'per_class':classification_report(actual,predicted,output_dict=True,zero_division=0),
        'confusion_labels':labels,'confusion_matrix':confusion_matrix(actual,predicted,labels=labels).tolist()}


def item(row):return {'source':'Benchmark','source_type':'manual','timestamp':'2022-07-01T00:00:00Z',**row}


def main():
    sentiment=json.loads((ROOT/'data/sentiment_eval.json').read_text(encoding='utf-8'))
    events=json.loads((ROOT/'data/event_eval.json').read_text(encoding='utf-8'))
    rules=RiskEngine(load_portfolio(),'baseline');linear=scorer()
    actual=[r['expected_sentiment'] for r in sentiment];sentiment_reports=[]
    def rules_label(t):
        score=baseline_sentiment_score(t)
        return 'positive' if score>.15 else 'negative' if score<-.15 else 'neutral'
    def learned_label(t):
        probabilities=linear.predict(t)
        return max(probabilities,key=probabilities.get)
    for name,predictor in [('Rules',rules_label),('Learned linear',learned_label)]:
        predictor('The company reports a profit.') # warm initialization excluded
        start=time.perf_counter();pred=[predictor(r['text']) for r in sentiment];elapsed=(time.perf_counter()-start)*1000/len(sentiment)
        details=metrics(actual,pred)
        sentiment_reports.append({'mode':name,'n':len(actual),**details,'mean_inference_ms':elapsed,
          'cases':[{'id':r['id'],'actual':r['expected_sentiment'],'predicted':p,'correct':r['expected_sentiment']==p} for r,p in zip(sentiment,pred)]})
    # Optional Transformer reference on the exact same 240 rows. No label tuning.
    model=CompactNLP();clean=__import__('src.nlp_backend',fromlist=['clean_text']).clean_text
    model.sentiment(clean('The company reports a profit.'),top_k=None,truncation=True,max_length=256)
    start=time.perf_counter();probs=model.sentiment([clean(r['text']) for r in sentiment],top_k=None,truncation=True,max_length=256,batch_size=16)
    elapsed=(time.perf_counter()-start)*1000/len(sentiment)
    pred=[max(ps,key=lambda p:p['score'])['label'].lower() for ps in probs]
    details=metrics(actual,pred)
    sentiment_reports.append({'mode':'DistilRoBERTa reference (batch 16)','n':len(actual),**details,'mean_inference_ms':elapsed,
        'cases':[{'id':r['id'],'actual':r['expected_sentiment'],'predicted':p,'correct':r['expected_sentiment']==p} for r,p in zip(sentiment,pred)]})
    cache=json.loads((ROOT/'data/prediction_cache.json').read_text(encoding='utf-8'))['predictions']
    event_reports=[];actual=[r['expected_category'] for r in events]
    for name,mode in [('Rules','baseline'),('Compact semantic model','compact'),('Hybrid replay','cached')]:
        engine=RiskEngine(load_portfolio(),mode)
        if mode=='compact':
            class FrozenProvider:
                def predict(self,text):return cache[text_key(text)]
            engine.prediction_provider=FrozenProvider()
        start=time.perf_counter();signals=[engine.analyze(item(r)) for r in events];wall=(time.perf_counter()-start)*1000/len(events)
        pred=[s['event_classification'] for s in signals];details=metrics(actual,pred)
        event_reports.append({'mode':name,'n':len(actual),**details,'replay_analysis_ms':wall,
          'recorded_full_model_inference_ms':statistics.mean(cache[text_key(r['text'])]['inference_ms'] for r in events) if mode!='baseline' else None,
          'review_flag_fraction':sum(bool(s['review_reasons']) for s in signals)/len(signals),
          'cases':[{'id':r['id'],'text':r['text'],'actual':r['expected_category'],'predicted':s['event_classification'],'correct':r['expected_category']==s['event_classification'],'route':s['classification_route'],'review_reasons':s['review_reasons']} for r,s in zip(events,signals)]})
    result={'generated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'environment':{'python':platform.python_version(),'platform':platform.platform(),'model_threads':2},
      'sentiment':{'n':len(sentiment),'summary':[{k:r[k] for k in ['mode','n','accuracy','macro_f1','mean_inference_ms']} for r in sentiment_reports],'reports':sentiment_reports,
        'limitation':'240 stratified original-validation texts with publisher labels, 80 per class. URL-stripped normalized train overlaps removed. Linear model selected only on a grouped/deduplicated training-development split. Transformer pretraining overlap cannot be ruled out. Balanced sample accuracy does not estimate natural prevalence. Batched Transformer timing and sequential small-scorer timing are different workloads.'},
      'events':{'n':len(events),'summary':[{k:r[k] for k in ['mode','n','accuracy','macro_f1','replay_analysis_ms','recorded_full_model_inference_ms','review_flag_fraction']} for r in event_reports],'reports':event_reports,
        'limitation':'108 builder-curated event labels from public topic-validation text, covering all five target classes plus Other. Keyword-enriched selection, subjective mixed-event labels, not independently annotated or blinded. Original topic labels are not used as six-class ground truth. No representative-accuracy claim. Recorded model latency includes sentiment plus event inference; replay time does not measure new-text inference.'},
      'selection_note':'The small sentiment model and its hyperparameter were chosen on training-development data before this validation run. Hybrid rule routing is an explicit design choice, not a learned confidence probability.'}
    (ROOT/'docs/benchmark-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'sentiment':result['sentiment']['summary'],'events':result['events']['summary']},indent=2))

if __name__=='__main__':main()
