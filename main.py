"""Reproducible JSON pipeline, evaluation, and CPU model comparison."""
import argparse
import json
import time
from pathlib import Path
from src.pipeline import ROOT, run_pipeline, load_portfolio
from src.risk_engine import RiskEngine
from src.stress_engine import run_scenario
from src.comparison import compare_scenarios
from src.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['baseline','cached','hybrid','compact'],default='cached')
    parser.add_argument('--input',choices=['historical','snapshot','synthetic'],default='historical')
    parser.add_argument('--output',default='outputs/signals.json')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--suite',choices=['public','synthetic','challenge'],default='public')
    args=parser.parse_args()
    portfolio=load_portfolio()
    if args.benchmark:
        result=[]
        for mode in ['baseline','cached']:
            start=time.perf_counter();engine=RiskEngine(portfolio,mode)
            load_seconds=time.perf_counter()-start
            report=evaluate(engine,args.suite);report['engine_load_seconds']=load_seconds;result.append(report)
    else:
        engine=RiskEngine(portfolio,args.mode)
        if args.evaluate: result=evaluate(engine,args.suite)
        else:
            file={'historical':'public_replay.json','snapshot':'public_feed_snapshot.json'}.get(args.input)
            items=json.loads((ROOT/'data'/file).read_text(encoding='utf-8')) if file else None
            result=run_pipeline(args.mode,items,engine)
            result['scenarios']={e['id']:run_scenario(portfolio,e,rate_bps=-100 if e['risk_factor']=='rates_down' else 100) for e in result['events']}
            result['comparisons']={e['id']:compare_scenarios(portfolio,e,rate_bps=-100 if e['risk_factor']=='rates_down' else 100) for e in result['events'] if e['eligible']}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(f'Wrote {path}')


if __name__=='__main__': main()
