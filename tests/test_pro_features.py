import json,re
from pathlib import Path
import numpy as np
from src.pipeline import ROOT,load_portfolio
from src.risk_engine import RiskEngine
from src.source_manager import collect_sources
from src.store import SignalStore
from src.learned_sentiment import scorer


def test_failed_feeds_are_explicit_snapshot_fallback():
    def failure(kind,limit):raise TimeoutError('offline')
    out=collect_sources(fetcher=failure)
    assert out['delivery_mode']=='snapshot_fallback'
    assert {r['source_type'] for r in out['items']}=={'news','social'}
    assert all(r['delivery_mode']=='snapshot_fallback' and not r.get('fetched_at') for r in out['items'])
    assert all(s['error'] for s in out['source_status'])


def test_partial_live_failure_keeps_source_identity():
    def fetch(kind,limit):
        if kind=='social':raise OSError('blocked')
        return [{'text':'Fed raises rates.','source':'News','source_type':'news','timestamp':'2026-10-11T00:00:00Z'}]
    out=collect_sources(fetcher=fetch)
    assert out['delivery_mode']=='mixed_live_and_snapshot'
    assert out['items'][0]['delivery_mode']=='live_rss'
    assert all(r['delivery_mode']=='snapshot_fallback' for r in out['items'][1:])


def test_signal_history_is_idempotent_and_persistent(tmp_path):
    signal={'text':'Same report','timestamp':'2026-10-11T00:00:00Z','source':'Test','actual_nlp_backend':'rules'}
    store=SignalStore(tmp_path/'signals.sqlite');store.save([signal]);store.save([signal])
    assert len(SignalStore(tmp_path/'signals.sqlite').list())==1
    store.save([{**signal,'source':'Another source'}])
    assert len(store.list())==2


def test_external_validation_has_no_normalized_training_overlap():
    import csv
    norm=lambda t: re.sub(r'\s+',' ',re.sub(r'https?://\S+','',t)).strip().casefold()
    train={norm(r['text']) for r in csv.DictReader((ROOT/'data/public/sent_train.csv').open())}
    rows=json.loads((ROOT/'data/sentiment_eval.json').read_text())
    assert len(rows)==240 and len({norm(r['text']) for r in rows})==240
    assert not {norm(r['text']) for r in rows}&train
    assert {label:sum(r['expected_sentiment']==label for r in rows) for label in ['negative','positive','neutral']}=={'negative':80,'positive':80,'neutral':80}


def test_portable_model_matches_sklearn_vectorization():
    from sklearn.feature_extraction.text import TfidfVectorizer
    model=json.loads((ROOT/'data/sentiment_model.json').read_text())
    v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode',vocabulary=model['vocabulary'])
    v.idf_=np.asarray(model['idf'])
    text='Boeing reports stronger profits but weaker margins.'
    logits=v.transform([text])@np.asarray(model['coefficients']).T+np.asarray(model['intercepts'])
    probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
    actual=scorer().predict(text)
    assert np.allclose([actual[{'0':'negative','1':'positive','2':'neutral'}[c]] for c in model['classes']],probabilities[0],atol=1e-10)


def test_uncached_text_uses_real_learned_sentiment_and_flags_event_rules():
    s=RiskEngine(load_portfolio(),'cached').analyze({'text':'Boeing enjoys distinctly improved margins today.','source':'Test','source_type':'news','timestamp':'2026-10-11T00:00:00Z'})
    assert s['actual_nlp_backend']=='learned_sentiment+event_rules'
    assert s['sentiment_model_version'] and s['sentiment_probabilities']
    assert any('explicit supported risk-action' in r for r in s['review_reasons'])


def test_equity_analyst_downgrade_cannot_trigger_credit_shock():
    s=RiskEngine(load_portfolio()).analyze({'text':'Microsoft downgraded to sell, price target cut.','source':'Test','source_type':'news','timestamp':'2026-10-11T00:00:00Z'})
    assert s['risk_factor']=='unspecified'
    assert s['review_reasons']
