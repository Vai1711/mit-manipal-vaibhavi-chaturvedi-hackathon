"""Signed market P&L and separate one-year expected credit loss proxy."""
import math


def run_scenario(portfolio, event, rate_bps=100, spread_bps=150, pd_multiplier=3,
                 geopolitical_decline=.10, manual=False):
    if not all(math.isfinite(x) for x in (rate_bps, spread_bps, pd_multiplier, geopolitical_decline)):
        raise ValueError('Scenario parameters must be finite.')
    if not (-300 <= rate_bps <= 300 and 0 <= spread_bps <= 500 and 1 <= pd_multiplier <= 10 and 0 <= geopolitical_decline <= .5):
        raise ValueError('Scenario parameters outside supported bounds.')
    rows = []
    category = event['event_classification']
    rate_event = event['risk_factor'] in ('rates_up', 'rates_down')
    direction_mismatch = (event['risk_factor'] == 'rates_up' and rate_bps < 0) or (event['risk_factor'] == 'rates_down' and rate_bps > 0)
    ready = manual or (event['eligible'] and not direction_mismatch)
    market_before = sum(a.get('market_value', 0) for a in portfolio['assets'] if a['type'] != 'loan')
    ecl_before = sum(a['ead'] * a['pd'] * a['lgd'] for a in portfolio['assets'] if a['type'] == 'loan')
    for a in portfolio['assets']:
        reason = None
        if ready:
            if rate_event and a['type'] in ('bond', 'swap') and a.get('currency') == event.get('risk_currency', 'USD'):
                reason = a['currency'] + ' parallel rate-factor exposure'
            elif category == 'Credit Event' and a.get('entity') in event['entities'] and a['type'] in ('loan', 'bond'):
                reason = 'Direct borrower/issuer match'
            elif category == 'Geopolitical' and (a.get('entity') in event['entities'] or a.get('sector') in event['sectors']) and a['type'] in ('loan', 'bond', 'equity'):
                reason = 'Entity or sector match'
        pnl, credit_delta = 0., 0.
        formula = 'No mapped shock'
        if reason:
            if a['type'] == 'bond':
                shock = rate_bps if rate_event else spread_bps
                duration = a['modified_duration'] if rate_event else a['spread_duration']
                pnl = -a['market_value'] * duration * shock / 10000
                formula = f"-{a['market_value']:,.0f} × {duration:g} × {shock:g}/10000"
            elif a['type'] == 'swap':
                pnl = a['signed_rate_sensitivity_per_bp'] * rate_bps
                formula = f"{a['signed_rate_sensitivity_per_bp']:,.0f} × {rate_bps:g} bps"
            elif a['type'] == 'equity':
                pnl = -a['market_value'] * geopolitical_decline
                formula = f"-{a['market_value']:,.0f} × {geopolitical_decline:g}"
            elif a['type'] == 'loan':
                stressed_pd = min(1., a['pd'] * pd_multiplier)
                credit_delta = a['ead'] * (stressed_pd - a['pd']) * a['lgd']
                formula = f"{a['ead']:,.0f} × ({stressed_pd:.4f} - {a['pd']:.4f}) × {a['lgd']:g}"
        rows.append({'asset_id': a['id'], 'entity': a.get('entity', 'Rates hedge'), 'type': a['type'],
                     'exposure_reason': reason or 'Not affected', 'market_value_before': a.get('market_value'),
                     'market_pnl': pnl, 'ecl_increase': credit_delta, 'formula': formula})
    unhedged = sum(r['market_pnl'] for r in rows if r['type'] != 'swap')
    hedge = sum(r['market_pnl'] for r in rows if r['type'] == 'swap')
    credit = sum(r['ecl_increase'] for r in rows)
    return {'status': 'manual scenario' if manual else 'triggered' if ready else 'not triggered',
            'trigger_note': 'Shock direction opposes the event; explicit manual what-if required' if direction_mismatch and not manual else '',
            'market_value_before': market_before, 'market_value_after': market_before + unhedged + hedge,
            'unhedged_market_pnl': unhedged, 'hedge_pnl': hedge, 'hedged_market_pnl': unhedged + hedge,
            'baseline_ecl': ecl_before, 'stressed_ecl': ecl_before + credit, 'ecl_increase': credit,
            'affected_asset_count': sum(r['exposure_reason'] != 'Not affected' for r in rows),
            'scenario': {'rate_bps': rate_bps, 'spread_bps': spread_bps, 'pd_multiplier': pd_multiplier,
                         'geopolitical_decline': geopolitical_decline}, 'rows': rows}
