"""Authored regression cases, not independent real-headline accuracy evidence."""
import pytest
from src.pipeline import load_portfolio,run_pipeline
from src.risk_engine import RiskEngine
from src.stress_engine import run_scenario


def analyze(text,source_type='news'):
    item={'id':'fresh-test','text':text,'source':'Authored regression','source_type':source_type,'timestamp':'2026-10-11T00:00:00Z'}
    out=run_pipeline('cached',[item])
    return out['signals'][0],out['events'][0]


@pytest.mark.parametrize('text,category,factor',[
 ('Fed raises rates by 75 basis points.','Macroeconomic','rates_up'),
 ('The Federal Reserve lowers interest rates by 25 basis points.','Macroeconomic','rates_down'),
 ("Boeing misses interest payment; Moody’s cuts rating to junk.",'Credit Event','credit'),
 ('Rite Aid files for Chapter 11 bankruptcy.','Credit Event','credit'),
 ('Houthi attacks disrupt Red Sea shipping.','Geopolitical','geopolitical'),
])
def test_clear_fresh_reports_are_not_blocked_by_cache_miss(text,category,factor):
    signal,event=analyze(text)
    assert not signal['cache_hit']
    assert signal['event_classification']==category
    assert signal['risk_factor']==factor
    assert signal['classification_route']=='explicit event evidence + learned sentiment'
    assert signal['actual_nlp_backend']=='learned_sentiment+event_rules'
    assert not signal['review_reasons']
    assert event['eligible']


@pytest.mark.parametrize('text',[
 'Rumor: Fed may raise rates by 75 basis points.',
 'Fed denies raising interest rates by 75 basis points.',
 'Fed raises rates now but cuts rates tomorrow.',
 'Central bank raises interest rates by 75 basis points.',
 'Boeing denies missing interest payment.',
 'Microsoft downgraded to sell; price target cut.',
 'Houthi representatives discuss Red Sea shipping schedules.',
 'Boeing reports improved margins.',
])
def test_new_text_keeps_uncertainty_and_context_gates(text):
    _,event=analyze(text)
    assert not event['eligible']


def test_social_only_report_still_needs_verification():
    _,event=analyze('Fed raises rates by 75 basis points.','social')
    assert not event['eligible']
    assert any('Social-only' in x for x in event['review_reasons'])


def test_fresh_borrower_event_maps_to_boeing_assets():
    _,event=analyze('Boeing misses interest payment and defaults on debt.')
    result=run_scenario(load_portfolio(),event)
    assert result['status']=='triggered'
    assert {row['entity'] for row in result['rows'] if row['exposure_reason']!='Not affected'}=={'Boeing'}
