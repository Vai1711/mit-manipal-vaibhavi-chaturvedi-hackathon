"""Deterministic benchmark preparation. Never uses model predictions to pick cases."""
import csv,json,random,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    data=ROOT/'data'
    sentiment=list(csv.DictReader((data/'public/sent_valid.csv').open(encoding='utf-8')))
    train=list(csv.DictReader((data/'public/sent_train.csv').open(encoding='utf-8')))
    normalize=lambda t: re.sub(r'\s+',' ',re.sub(r'https?://\S+','',t)).strip().casefold()
    training_texts={normalize(r['text']) for r in train}
    seen=set();clean=[]
    for i,row in enumerate(sentiment):
        key=normalize(row['text'])
        if key in training_texts or key in seen:continue
        seen.add(key);clean.append((i,row))
    # 80 per class: fixed seed, publisher labels retained unchanged.
    chosen=[]
    for label in ['0','1','2']:
        chosen+=random.Random(20261011+int(label)).sample([r for r in clean if r[1]['label']==label],80)
    labels={'0':'negative','1':'positive','2':'neutral'}
    sample=[{'id':f'twitter-sent-valid-{i}','text':r['text'],'expected_sentiment':labels[r['label']],
             'publisher_label':int(r['label']),'dataset_split':'validation','source':'Twitter Financial News dataset',
             'source_type':'social','synthetic':False,'url':'https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment'} for i,r in sorted(chosen)]
    (data/'sentiment_eval.json').write_text(json.dumps(sample,indent=2),encoding='utf-8')
    topics=list(csv.DictReader((data/'public/topic_valid.csv').open(encoding='utf-8')))
    curated={
      'Credit Event':[613,750,831,868,905,911,1554,2032,2063,2259,2297,4060],
      'Product Launch':[93,118,153,192,202,220,226,231,250,268,269,326,308,327,330,371,417,427],
      'Geopolitical':[606,822,861,1313,1316,1347,1349,1409,1413,1428,1834,1838,1933,1975,2150,2475,2518,2604],
      'Macroeconomic':[i for i,r in enumerate(topics) if int(r['label'])==1][:20],
      'Merger/Acquisition':[i for i,r in enumerate(topics) if int(r['label'])==13 and re.search(r'\bacquir|\bacquisition|\bmerger',r['text'],re.I)][:20],
      'Other/Uncertain':[i for i,r in enumerate(topics) if int(r['label'])==5][:20]}
    events=[]
    for category,indices in curated.items():
        for i in indices:
            r=topics[i];links=re.findall(r'https?://\S+',r['text'])
            events.append({'id':f'twitter-event-valid-{i}','text':r['text'],'expected_category':category,
             'original_topic_label':int(r['label']),'label_provenance':'Builder curated fine-grained event label. Original topic labels are broader and are not ground truth for these six categories.',
             'dataset_split':'validation','source':'Twitter Financial News topic dataset','source_type':'social',
             'timestamp':'2022-07-01T00:00:00Z','timestamp_quality':'Dataset lacks timestamps: placeholder for pipeline only, not event chronology.',
             'synthetic':False,'url':links[0] if links else 'https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic'})
    for r in events:
        i=int(r['id'].split('-')[-1])
        if i in [1451,1458,1459,1460,2307]:
            r['expected_category']='Other/Uncertain'
            r['label_rationale']='Regulation, personnel/survey announcement, or equipment purchase outside the five target event types.'
        if i==1457:
            r['expected_category']='Product Launch'
            r['label_rationale']='Launch of a named financial benchmark product.'
    (data/'event_eval.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    metadata={'seed':20261011,'sentiment_n':len(sample),'sentiment_sampling':'80 per publisher sentiment class from original validation; normalized URL-stripped train overlap and internal duplicates excluded. No project training/tuning on these 240 cases. Pretrained-model overlap cannot be ruled out.',
      'event_n':len(events),'event_sampling':'Builder-curated keyword-enriched examples from original topic validation, covers six classes. Not random, blinded or independently labelled. Interpretive mixed-event labels possible. No claim of representative accuracy.',
      'original_row_counts':{'sent_train':len(train),'sent_valid':len(sentiment),'topic_valid':len(topics)},
      'dataset_card_mismatch':'Sentiment card split counts differ from actual downloaded CSV counts. Actual file counts and hashes are authoritative.',
      'test_hashes':{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['sentiment_eval.json','event_eval.json']}}
    (data/'benchmark_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Prepared',len(sample),'sentiment and',len(events),'event evaluation texts')

if __name__=='__main__':main()
