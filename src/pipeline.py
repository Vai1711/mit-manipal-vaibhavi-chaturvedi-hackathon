import json
from pathlib import Path
from .ingestion import load_replay, prepare
from .risk_engine import RiskEngine
from .event_grouper import group_signals

ROOT = Path(__file__).resolve().parents[1]


def load_portfolio():
    return json.loads((ROOT / 'data/portfolio.json').read_text(encoding='utf-8'))


def run_pipeline(mode='baseline', items=None, engine=None):
    feed, duplicates = load_replay(ROOT / 'data') if items is None else prepare(items)
    engine = engine or RiskEngine(load_portfolio(), mode)
    signals = [engine.analyze(item) for item in feed]
    return {'mode': mode, 'signals': signals, 'events': group_signals(signals), 'duplicates': duplicates}
