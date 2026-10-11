"""Exploratory historical association. Six examples cannot calibrate severity."""
import csv,json,math
from .pipeline import ROOT,load_portfolio,run_pipeline


def rank(values):
    ordered=sorted(set(values))
    return [sum(i+1 for i,v in enumerate(sorted(values)) if v==x)/values.count(x) for x in values]


def correlation(x,y):
    xm=sum(x)/len(x);ym=sum(y)/len(y)
    denom=math.sqrt(sum((v-xm)**2 for v in x)*sum((v-ym)**2 for v in y))
    return sum((a-xm)*(b-ym) for a,b in zip(x,y))/denom if denom else None


def check_impact(mode='cached'):
    file=ROOT/'data/public/spy_daily.csv'
    if not file.exists():return {'available':False,'note':'No public price data bundled; no calibration claim.'}
    prices=list(csv.DictReader(file.open(encoding='utf-8')))
    reports=json.loads((ROOT/'data/public_replay.json').read_text(encoding='utf-8'))
    signals=run_pipeline(mode,reports)['signals'];rows=[]
    for s in signals:
        event_date=s['timestamp'][:10]
        i=next((i for i,p in enumerate(prices) if p['Date']>=event_date),None)
        if i is None or i==0:continue
        previous=float(prices[i-1]['Close']);close=float(prices[i]['Close'])
        rows.append({'event_id':s['id'],'category':s['event_classification'],'severity':s['impact_score'],
            'event_date':event_date,'trading_date':prices[i]['Date'],'previous_trading_date':prices[i-1]['Date'],
            'previous_close':previous,'close':close,'spy_return_percent':100*(close/previous-1),'url':s['url']})
    return {'available':True,'n':len(rows),'rows':rows,
        'spearman_severity_vs_absolute_spy_return':correlation(rank([r['severity'] for r in rows]),rank([abs(r['spy_return_percent']) for r in rows])) if len(rows)>1 else None,
        'note':'Exploratory same-day SPY close-to-close association, next trading day for weekends. Unadjusted closes, no dividend correction. Company credit/product events may have little SPY exposure. Other news and anticipation confound returns. Six selected examples cannot establish causation, train a severity model, or validate simulated portfolio losses.'}


if __name__ == '__main__':
    path = ROOT / 'docs/impact-check.json'
    path.write_text(json.dumps(check_impact(), indent=2) + '\n', encoding='utf-8')
    print('Wrote', path)
