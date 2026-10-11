import copy
import math
import pytest
from src.pipeline import run_pipeline, load_portfolio
from src.risk_engine import RiskEngine
from src.ingestion import prepare
from src.event_grouper import group_signals
from src.stress_engine import run_scenario


def event(category='Macroeconomic', factor='rates_up', entities=None, sectors=None):
    return {'event_classification': category, 'risk_factor': factor, 'entities': entities or [],
            'sectors': sectors or [], 'eligible': True}


def reference_portfolio():
    p = load_portfolio()
    p['assets'] = [a for a in p['assets'] if a['id'] in ('B01', 'B02', 'S01')]
    return p


def test_reference_bond_and_swap_calculation():
    result = run_scenario(reference_portfolio(), event(), rate_bps=100)
    assert result['unhedged_market_pnl'] == pytest.approx(-21e6)
    assert result['hedge_pnl'] == pytest.approx(1.25e6)
    assert result['hedged_market_pnl'] == pytest.approx(-19.75e6)
    assert result['market_value_before'] == 300e6  # Notional not added.
    assert result['market_value_after'] == pytest.approx(280.25e6)


def test_negative_shock_and_overhedging_preserve_signed_gains():
    p = reference_portfolio()
    result = run_scenario(p, event(factor='rates_down'), rate_bps=-100)
    assert result['hedged_market_pnl'] == pytest.approx(19.75e6)
    p['assets'][-1]['signed_rate_sensitivity_per_bp'] = 300000
    result = run_scenario(p, event())
    assert result['hedged_market_pnl'] == pytest.approx(9e6)


def test_credit_is_exposure_specific_and_not_market_price():
    result = run_scenario(load_portfolio(), event('Credit Event', 'credit', ['Boeing']))
    assert result['ecl_increase'] == pytest.approx(200e6 * (.045 - .015) * .45)
    assert result['unhedged_market_pnl'] == pytest.approx(-120e6 * 7 * .015)
    affected = [r['asset_id'] for r in result['rows'] if r['exposure_reason'] != 'Not affected']
    assert set(affected) == {'L01', 'B02'}
    assert result['market_value_after'] == result['market_value_before'] + result['hedged_market_pnl']


def test_geopolitical_sector_mapping():
    result = run_scenario(load_portfolio(), event('Geopolitical', 'geopolitical', sectors=['energy']))
    affected = {r['asset_id'] for r in result['rows'] if r['exposure_reason'] != 'Not affected'}
    assert affected == {'B03', 'L03', 'E01'}
    assert result['hedge_pnl'] == 0


def test_unknown_entity_does_not_stress_entire_loan_book():
    result = run_scenario(load_portfolio(), event('Credit Event', 'credit', ['Unknown Corp']))
    assert result['affected_asset_count'] == 0


def test_review_blocks_auto_but_allows_explicit_what_if():
    e = event('Credit Event', 'credit', ['Rite Aid'])
    e['eligible'] = False
    assert run_scenario(load_portfolio(), e)['ecl_increase'] == 0
    assert run_scenario(load_portfolio(), e, manual=True)['ecl_increase'] > 0


def test_duplicate_normalization_keeps_provenance():
    item = {'id': 'a', 'text': ' Rate  hike! ', 'source': 'A', 'source_type': 'news', 'timestamp': '2026-10-10T00:00:00Z'}
    other = {**item, 'id': 'b', 'text': 'rate hike!', 'source': 'B'}
    unique, duplicates = prepare([item, other])
    assert len(unique) == 1 and len(duplicates) == 1
    assert unique[0]['duplicate_sources'] == ['B']


def test_grouping_mean_and_time_window():
    engine = RiskEngine(load_portfolio())
    text = 'Central bank announces rate hike of 100 basis points.'
    signals = [engine.analyze({'id': str(i), 'text': text, 'source': 'A', 'source_type': 'news',
                             'timestamp': f'2026-10-10T0{i}:00:00Z'}) for i in range(3)]
    for s, value in zip(signals, [-1, 0, 1]): s['sentiment_score'] = value
    clusters = group_signals(signals)
    assert len(clusters) == 1 and clusters[0]['sentiment_score'] == 0
    later = {**signals[0], 'id': 'later', 'timestamp': '2026-10-11T00:00:00Z'}
    assert len(group_signals(signals + [later])) == 2


def test_rumor_and_negation_review_and_no_fake_confidence():
    engine = RiskEngine(load_portfolio())
    for text in ['Unconfirmed rumor: Rite Aid faces bankruptcy.', 'Boeing denies default.']:
        s = engine.analyze({'id': 'x', 'text': text, 'source': 'A', 'source_type': 'news', 'timestamp': '2026-10-10T00:00:00Z'})
        assert s['review_reasons']
        assert s['event_model_score'] is None
        assert not group_signals([s])[0]['eligible']


@pytest.mark.parametrize('kwargs', [{'rate_bps': math.nan}, {'rate_bps': 301}, {'pd_multiplier': 0}, {'geopolitical_decline': 2}])
def test_invalid_scenario_parameters(kwargs):
    with pytest.raises(ValueError): run_scenario(load_portfolio(), event(), **kwargs)


def test_all_required_categories_and_outputs():
    output = run_pipeline()
    assert set(s['event_classification'] for s in output['signals']) >= {
        'Credit Event', 'Geopolitical', 'Macroeconomic', 'Merger/Acquisition', 'Product Launch'}
    assert len(output['duplicates']) == 1
    assert {s['source_type'] for s in output['signals']} == {'news', 'social'}
    assert all(-1 <= s['sentiment_score'] <= 1 and 1 <= s['impact_score'] <= 10 for s in output['signals'])
