import json
import pytest
from src.pipeline import ROOT,load_portfolio,run_pipeline
from src.risk_engine import RiskEngine
from src.event_grouper import group_signals
from src.stress_engine import run_scenario
from src.comparison import compare_scenarios


def signal(text):
    return RiskEngine(load_portfolio()).analyze({'id':'x','text':text,'source':'Test','source_type':'news','timestamp':'2026-10-10T00:00:00Z'})


@pytest.mark.parametrize('text,status',[
    ('Federal Reserve does not raise interest rates.','negated'),
    ('Rite Aid avoided bankruptcy.','negated'),
    ('Boeing denies default.','negated'),
    ('Fed may cut interest rates next month.','uncertain'),
    ('Fed raises rates today but cuts rates next year.','conflicting')])
def test_assertion_checks_pause_automatic_stress(text,status):
    s=signal(text)
    assert s['assertion_status']==status
    assert not group_signals([s])[0]['eligible']


def test_ecb_does_not_shock_usd_book():
    s=signal('ECB raises interest rates.')
    assert s['risk_currency']=='EUR'
    assert run_scenario(load_portfolio(),group_signals([s])[0])['affected_asset_count']==0


def test_unknown_rate_jurisdiction_needs_review():
    assert signal('An unidentified central bank raises rates.')['review_reasons']


def test_evaluation_text_and_urls_are_disjoint_from_demo():
    evaluation=json.loads((ROOT/'data/public_eval.json').read_text())
    demo=json.loads((ROOT/'data/public_replay.json').read_text())+json.loads((ROOT/'data/public_feed_snapshot.json').read_text())
    assert len(evaluation)==13
    assert not {r['text'] for r in evaluation}&{r['text'] for r in demo}
    assert not {r['url'] for r in evaluation}&{r['url'] for r in demo}
    assert {r['source_type'] for r in demo}=={'news','social'}


def test_credit_comparison_selectivity():
    items=json.loads((ROOT/'data/public_replay.json').read_text())
    e=next(e for e in run_pipeline(items=items)['events'] if e['id']=='event-rite-aid')
    r=compare_scenarios(load_portfolio(),e)
    assert r['exposure_aware']['affected_asset_count']==2
    assert len(r['blanket_only_asset_ids'])>0
    assert r['exposure_aware']['ecl_increase']==pytest.approx(2.7e6)


def test_opposite_direction_needs_explicit_what_if():
    event=group_signals([signal('Federal Reserve cuts interest rates.')])[0]
    assert run_scenario(load_portfolio(),event,rate_bps=100)['status']=='not triggered'
    assert run_scenario(load_portfolio(),event,rate_bps=100,manual=True)['status']=='manual scenario'


def test_opposing_nearby_reports_pause_both_clusters():
    clusters=group_signals([signal('Federal Reserve raises interest rates.'),signal('Federal Reserve cuts interest rates.')])
    assert len(clusters)==2
    assert all(not c['eligible'] and any('disagree' in r for r in c['review_reasons']) for c in clusters)
