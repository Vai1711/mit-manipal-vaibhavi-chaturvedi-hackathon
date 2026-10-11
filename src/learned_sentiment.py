"""Portable trained linear scorer. JSON parameters, no pickle or training at startup."""
import json,math,re,unicodedata
from pathlib import Path
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]


class LearnedSentiment:
    def __init__(self):
        import numpy as np
        self.np=np
        model=json.loads((ROOT/'data/sentiment_model.json').read_text(encoding='utf-8'))
        self.vocabulary=model['vocabulary'];self.idf=model['idf'];self.coefficients=np.asarray(model['coefficients']);self.intercepts=np.asarray(model['intercepts']);self.classes=model['classes']
        self.version=__import__('hashlib').sha256((ROOT/'data/sentiment_model.json').read_bytes()).hexdigest()

    def predict(self,text):
        text=re.sub(r'https?://\S+','',text).lower()
        text=''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))
        tokens=re.findall(r'(?u)\b\w\w+\b',text)
        terms=tokens+[' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        counts={}
        for term in terms:
            index=self.vocabulary.get(term)
            if index is not None:counts[index]=counts.get(index,0)+1
        weights={i:(1+math.log(n))*self.idf[i] for i,n in counts.items()}
        norm=math.sqrt(sum(w*w for w in weights.values()))
        logits=self.intercepts.copy()
        if norm:
            for i,w in weights.items():logits+=self.coefficients[:,i]*(w/norm)
        probabilities=self.np.exp(logits-self.np.max(logits));probabilities/=probabilities.sum()
        mapping={'0':'negative','1':'positive','2':'neutral'}
        return {mapping[c]:float(p) for c,p in zip(self.classes,probabilities)}


@lru_cache(maxsize=1)
def scorer():return LearnedSentiment()
