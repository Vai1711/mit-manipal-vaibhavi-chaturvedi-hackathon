"""File replay and bounded public RSS adapters. No credentials required."""
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


def timestamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamps must include a timezone.')
    return dt.astimezone(timezone.utc)


def normalize(text):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]*>', ' ', text))).strip()


def prepare(items):
    """Deduplicate text while retaining attribution; copies are not corroboration."""
    unique, duplicates, seen = [], [], {}
    for raw in items:
        item = dict(raw)
        text = normalize(str(item.get('text', '')))
        if not text:
            raise ValueError('Feed item text cannot be empty.')
        for field in ('source', 'source_type', 'timestamp'):
            if not item.get(field):
                raise ValueError(f'Feed item requires {field}.')
        if item['source_type'] not in ('news', 'social', 'manual'):
            raise ValueError('source_type must be news, social, or manual.')
        timestamp(item['timestamp'])
        digest = hashlib.sha256(text.casefold().encode()).hexdigest()
        item.update(text=text, id=item.get('id') or digest[:12])
        if digest in seen:
            duplicates.append({'id': item['id'], 'duplicate_of': seen[digest]['id'], 'source': item['source']})
            seen[digest]['duplicate_sources'].append(item['source'])
        else:
            item['duplicate_sources'] = []
            seen[digest] = item
            unique.append(item)
    return sorted(unique, key=lambda i: timestamp(i['timestamp'])), duplicates


def load_replay(data_dir):
    items = []
    for name in ('news_feed.json', 'social_feed.json'):
        items.extend(json.loads((Path(data_dir) / name).read_text(encoding='utf-8')))
    return prepare(items)


PUBLIC_FEEDS = {
    'news': ('Google News RSS', 'https://news.google.com/rss/search?q=interest+rates+OR+corporate+default+OR+sanctions&hl=en-US&gl=US&ceid=US:en'),
    'social': ('Reddit r/finance RSS', 'https://www.reddit.com/r/finance/new/.rss'),
}


def fetch_public(kind, limit=10):
    """Explicit refresh only; errors propagate, never replaced by fabricated live data."""
    source, url = PUBLIC_FEEDS[kind]
    request = Request(url, headers={'User-Agent': 'RiskRadarStudentPrototype/0.1'})
    with urlopen(request, timeout=12) as response:
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError('Feed exceeds 2 MB limit.')
    root = ET.fromstring(body)
    atom = {'a': 'http://www.w3.org/2005/Atom'}
    rows = []
    entries = root.findall('./channel/item') or root.findall('a:entry', atom)
    for entry in entries[:limit]:
        if kind == 'news':
            title = entry.findtext('title', '')
            link = entry.findtext('link', '')
            published = entry.findtext('pubDate')
            dt = parsedate_to_datetime(published) if published else datetime.now(timezone.utc)
            publisher = entry.findtext('source', source)
        else:
            title = entry.findtext('a:title', '', atom)
            element = entry.find('a:link', atom)
            link = element.get('href', '') if element is not None else ''
            published = entry.findtext('a:updated', None, atom)
            dt = timestamp(published) if published else datetime.now(timezone.utc)
            publisher = source
        rows.append({'text': title, 'source': publisher, 'source_type': kind,
                     'timestamp': dt.isoformat(), 'url': link, 'synthetic': False})
    return rows
