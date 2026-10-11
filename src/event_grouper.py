"""Conservative similarity grouping with anchored six-hour windows."""
import re
from .ingestion import timestamp


def tokens(text):
    stop = {'the', 'a', 'an', 'of', 'to', 'in', 'on', 'and', 'for', 'is', 'its', 'with', 'that'}
    return set(re.findall(r'\b\w+\b', text.lower())) - stop


def group_signals(signals, window_hours=6):
    clusters = []
    for signal in sorted(signals, key=lambda s: timestamp(s['timestamp'])):
        key = (tuple(sorted(signal['entities'])), tuple(sorted(signal['sectors'])),
               signal['event_classification'], signal['risk_factor'], signal.get('risk_currency'), signal.get('assertion_status'))
        found = None
        for cluster in clusters:
            delta = (timestamp(signal['timestamp']) - timestamp(cluster['timestamp'])).total_seconds()
            a, b = tokens(signal['text']), tokens(cluster['signals'][0]['text'])
            similarity = len(a & b) / max(1, len(a | b))
            if cluster['_key'] == key and delta <= window_hours * 3600 and similarity >= .30:
                found = cluster
                break
        if found is None:
            found = {'id': 'event-' + signal['id'], '_key': key, 'timestamp': signal['timestamp'],
                     'entities': signal['entities'], 'sectors': signal['sectors'],
                     'event_classification': signal['event_classification'], 'risk_factor': signal['risk_factor'], 'signals': []}
            found['risk_currency'] = signal.get('risk_currency')
            found['assertion_status'] = signal.get('assertion_status', 'asserted')
            clusters.append(found)
        found['signals'].append(signal)
    for c in clusters:
        c.pop('_key')
        c['impact_score'] = max(s['impact_score'] for s in c['signals'])
        c['sentiment_score'] = round(sum(s['sentiment_score'] for s in c['signals']) / len(c['signals']), 4)
        c['sources'] = sorted(set(s['source'] for s in c['signals']))
        c['review_reasons'] = sorted(set(r for s in c['signals'] for r in s['review_reasons']))
        if all(s['source_type'] == 'social' for s in c['signals']):
            c['review_reasons'].append('Social-only event: verification needed')
        c['eligible'] = c['impact_score'] > 7 and not c['review_reasons'] and c['risk_factor'] in ('rates_up', 'rates_down', 'credit', 'geopolitical')
    # Keep opposing reports separate, but pause automation when nearby reports
    # disagree on the same currency's policy direction.
    for i, first in enumerate(clusters):
        for second in clusters[i + 1:]:
            nearby = abs((timestamp(first['timestamp']) - timestamp(second['timestamp'])).total_seconds()) <= window_hours * 3600
            opposed = {first['risk_factor'], second['risk_factor']} == {'rates_up', 'rates_down'}
            if nearby and opposed and first.get('risk_currency') and first.get('risk_currency') == second.get('risk_currency'):
                for c in (first, second):
                    c['review_reasons'] = sorted(set(c['review_reasons'] + ['Nearby reports disagree on policy-rate direction']))
                    c['eligible'] = False
    return clusters
