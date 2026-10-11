"""Fit a small supervised financial sentiment scorer. Development data only."""
import csv,json,re,hashlib,time,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[1]


def clean(text):return re.sub(r'\s+',' ',re.sub(r'https?://\S+','',text)).strip()


def main():
    rows=list(csv.DictReader((ROOT/'data/public/sent_train.csv').open(encoding='utf-8')))
    seen={};conflicts=set()
    for r in rows:
        text=clean(r['text']);key=text.casefold()
        if key in seen and seen[key]['label']!=r['label']:conflicts.add(key)
        else:seen[key]={'text':text,'label':r['label']}
    rows=[v for k,v in seen.items() if k not in conflicts]
    train,dev=train_test_split(rows,test_size=.20,random_state=20261011,stratify=[r['label'] for r in rows])
    vector=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=18000,sublinear_tf=True,strip_accents='unicode')
    x=vector.fit_transform([r['text'] for r in train]);dx=vector.transform([r['text'] for r in dev])
    candidates=[];best=None
    for c in [.3,1.,3.]:
        start=time.perf_counter();model=LogisticRegression(C=c,max_iter=600,class_weight='balanced',solver='lbfgs',random_state=20261011)
        model.fit(x,[r['label'] for r in train]);pred=model.predict(dx)
        score=f1_score([r['label'] for r in dev],pred,average='macro')
        candidates.append({'C':c,'development_macro_f1':score,'development_accuracy':accuracy_score([r['label'] for r in dev],pred),'fit_seconds':time.perf_counter()-start})
        if best is None or score>best[0]:best=(score,model,c)
    # Keep the selected train-only model. The development set is not refit.
    model=best[1]
    out={'model_type':'TF-IDF word/bigram + multinomial logistic regression','train_n':len(train),'development_n':len(dev),'seed':20261011,
       'selected_C':best[2],'selection':'Highest development macro F1. Publisher validation never used for vocabulary, fitting or selection.',
       'classes':model.classes_.tolist(),'vocabulary':{k:int(v) for k,v in vector.vocabulary_.items()},'idf':vector.idf_.tolist(),
       'coefficients':model.coef_.tolist(),'intercepts':model.intercept_.tolist(),
       'dataset':'zeroshot/twitter-financial-news-sentiment, original train split','training_source_hash':hashlib.sha256((ROOT/'data/public/sent_train.csv').read_bytes()).hexdigest()}
    (ROOT/'data/sentiment_model.json').write_text(json.dumps(out,separators=(',',':')),encoding='utf-8')
    (ROOT/'docs/training-report.json').write_text(json.dumps({'candidates':candidates,'selected_C':best[2],'train_n':len(train),'development_n':len(dev),
       'excluded_conflicting_normalized_texts':len(conflicts),'limitation':'Source split uses financial tweets, not full articles. No evidence of generalization to another period. Linear vocabulary cannot fully resolve negation or target-specific tone.'},indent=2))
    print('Selected development-only model:',candidates)

if __name__=='__main__':main()
