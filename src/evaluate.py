"""Measured evaluation; builder labels and dataset limitations are explicit."""
import json
import time
from collections import Counter
from .pipeline import ROOT


def evaluate(engine, suite='public'):
    file = {'public':'public_eval.json', 'synthetic':'eval_set.json', 'challenge':'challenge_set.json'}[suite]
    rows = json.loads((ROOT / 'data' / file).read_text(encoding='utf-8'))
    cases, timings = [], []
    for row in rows:
        start = time.perf_counter()
        signal = engine.analyze({'source':'Validation','source_type':'manual','timestamp':'2026-10-10T00:00:00Z', **row})
        elapsed = (time.perf_counter()-start)*1000
        timings.append(elapsed)
        score = signal['sentiment_score']
        predicted = signal['sentiment_label']
        cases.append({**row, 'predicted_category':signal['event_classification'], 'predicted_sentiment':predicted,
                      'predicted_risk_factor':signal['risk_factor'],'predicted_assertion_status':signal['assertion_status'],
                      'category_correct':signal['event_classification']==row['expected_category'],
                      'sentiment_correct':predicted==row['expected_sentiment'],
                      'direction_correct':signal['risk_factor']==row.get('expected_risk_factor'),
                      'assertion_correct':signal['assertion_status']==row.get('expected_assertion_status'),
                      'latency_ms':round(elapsed,3),'review_reasons':signal['review_reasons']})
    labels=sorted(set(r['expected_category'] for r in rows))
    per_class={}
    for label in labels:
        tp=sum(c['expected_category']==label and c['predicted_category']==label for c in cases)
        fp=sum(c['expected_category']!=label and c['predicted_category']==label for c in cases)
        fn=sum(c['expected_category']==label and c['predicted_category']!=label for c in cases)
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        per_class[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}
    confusion=Counter((c['expected_category'],c['predicted_category']) for c in cases)
    limitations={'public':'Small builder-labelled public set, disjoint from packaged demo text/URLs. Not blinded or representative; pretraining contamination possible. Labels are interpretive.',
                 'synthetic':'Authored synthetic examples; illustrative, not real-world validation.',
                 'challenge':'Authored assertion/direction edge cases; sentiment labels are placeholders, so sentiment accuracy is not informative.'}
    return {'mode':engine.mode,'suite':suite,'n':len(cases),
            'event_accuracy':sum(c['category_correct'] for c in cases)/len(cases),
            'sentiment_accuracy':sum(c['sentiment_correct'] for c in cases)/len(cases),
            'event_macro_f1':sum(v['f1'] for v in per_class.values())/len(per_class),
            'direction_accuracy':sum(c['direction_correct'] for c in cases)/len(cases) if suite!='synthetic' else None,
            'assertion_accuracy':sum(c['assertion_correct'] for c in cases)/len(cases) if suite!='synthetic' else None,
            'mean_inference_ms':sum(timings)/len(timings),'per_class':per_class,
            'confusion':[{'actual':a,'predicted':b,'count':n} for (a,b),n in confusion.items()],
            'cases':cases,'limitation':limitations[suite],
            'measurement_note':'Sequential CPU inference; model initialization excluded. Direction/assertion controls shared by both modes. Severity and losses are assumptions, not forecast accuracy metrics.'}
