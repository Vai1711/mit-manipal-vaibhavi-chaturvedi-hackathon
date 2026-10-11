"""Local REST interface. Run with uvicorn api:app --host 127.0.0.1."""
from datetime import datetime,timezone
from functools import lru_cache
import os,threading
from typing import Literal
from fastapi import FastAPI,HTTPException,Query
from pydantic import BaseModel,Field,field_validator
from src.pipeline import ROOT,load_portfolio,run_pipeline
from src.risk_engine import RiskEngine
from src.store import SignalStore

app=FastAPI(title='Risk Radar',version='2.0',description='Analyst prototype: public text, hypothetical exposures, assumed shocks.')
lock=threading.Lock()


class AnalyzeRequest(BaseModel):
    text:str=Field(min_length=1,max_length=6000)
    source:str=Field(default='API analyst input',min_length=1,max_length=120)
    source_type:Literal['news','social','manual']='manual'
    timestamp:str|None=None
    mode:Literal['baseline','cached','hybrid','compact']='cached'

    @field_validator('text')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():raise ValueError('Text cannot be blank')
        return value.strip()


@lru_cache(maxsize=3)
def engine(mode):return RiskEngine(load_portfolio(),mode)


def store():return SignalStore(os.environ.get('RADAR_DB_PATH',str(ROOT/'outputs/signals.sqlite')))


@app.get('/health')
def health():return {'status':'ok','model_downloads_automatic':False,'portfolio':'synthetic','live_polling_automatic':False}


@app.post('/analyze')
def analyze(request:AnalyzeRequest):
    item={'text':request.text,'source':request.source,'source_type':request.source_type,
          'timestamp':request.timestamp or datetime.now(timezone.utc).isoformat(),'synthetic':False,'timestamp_quality':'analyst supplied' if request.timestamp else 'ingestion time, not publication time'}
    try:
        with lock:result=run_pipeline(request.mode,[item],engine(request.mode))
        store().save(result['signals'])
        return result
    except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
    except (RuntimeError,ImportError,OSError) as exc:raise HTTPException(503,str(exc)) from exc


@app.get('/signals')
def signals(limit:int=Query(default=100,ge=1,le=500)):
    return {'signals':store().list(limit),'note':'Persisted observations; observed_at is ingestion time, not a market-event timestamp.'}
