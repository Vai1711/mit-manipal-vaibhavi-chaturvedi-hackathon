"""Bounded, explicit live fetching with visible per-source snapshot fallback."""
import json
from datetime import datetime,timezone
from .pipeline import ROOT
from .ingestion import fetch_public


def collect_sources(fetcher=fetch_public,snapshot=None):
    snapshot=snapshot if snapshot is not None else json.loads((ROOT/'data/public_feed_snapshot.json').read_text(encoding='utf-8'))
    items,statuses=[],[]
    now=datetime.now(timezone.utc).isoformat(timespec='seconds')
    for kind in ('news','social'):
        try:
            rows=fetcher(kind,limit=5)
            if not rows:raise ValueError('Feed returned no usable reports')
            items.extend({**r,'delivery_mode':'live_rss','fetched_at':now} for r in rows)
            statuses.append({'source_type':kind,'status':'live','count':len(rows),'last_successful_fetch':now,'error':None})
        except Exception as exc:
            fallback=[r for r in snapshot if r['source_type']==kind][:5]
            items.extend({**r,'delivery_mode':'snapshot_fallback'} for r in fallback)
            statuses.append({'source_type':kind,'status':'saved snapshot','count':len(fallback),'last_successful_fetch':None,'error':f'{type(exc).__name__}: {exc}'})
    live=sum(s['status']=='live' for s in statuses)
    return {'items':items,'source_status':statuses,'attempted_at':now,
            'delivery_mode':'live_rss' if live==2 else 'mixed_live_and_snapshot' if live==1 else 'snapshot_fallback'}
