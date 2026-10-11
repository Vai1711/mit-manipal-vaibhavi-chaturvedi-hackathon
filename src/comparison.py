"""Counterfactual blanket-shock comparison, not evidence of predictive accuracy."""
from .stress_engine import run_scenario


def compare_scenarios(portfolio, event, **params):
    mapped=run_scenario(portfolio,event,**params)
    ready=mapped['status'] != 'not triggered'
    decline=params.get('geopolitical_decline',.1)
    multiplier=params.get('pd_multiplier',3)
    rows=[]
    for a in portfolio['assets']:
        pnl=-a.get('market_value',0)*decline if ready and a['type'] in ('bond','equity') else 0
        credit=a['ead']*(min(1,a['pd']*multiplier)-a['pd'])*a['lgd'] if ready and a['type']=='loan' else 0
        rows.append({'asset_id':a['id'],'market_pnl':pnl,'ecl_increase':credit})
    mapped_ids={r['asset_id'] for r in mapped['rows'] if r['exposure_reason']!='Not affected'}
    blanket_ids={r['asset_id'] for r in rows if r['market_pnl'] or r['ecl_increase']}
    return {'exposure_aware':mapped,'blanket':{'market_pnl':sum(r['market_pnl'] for r in rows),
            'ecl_increase':sum(r['ecl_increase'] for r in rows),'affected_asset_count':len(blanket_ids),'rows':rows},
            'blanket_only_asset_ids':sorted(blanket_ids-mapped_ids),
            'assumptions':f'Blanket counterfactual: {decline:.0%} decline in every bond/equity, PD ×{multiplier:g} on every loan, no hedge response. Both share trigger gate. Different shock assumptions; comparison shows mapping selectivity, not forecast superiority.'}
