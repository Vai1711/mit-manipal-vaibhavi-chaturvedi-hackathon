"""Bounded local SQLite signal history with idempotent content/version keys."""
import hashlib,json,sqlite3
from datetime import datetime,timezone
from pathlib import Path


class SignalStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS signals (key TEXT PRIMARY KEY, observed_at TEXT NOT NULL, payload TEXT NOT NULL)')

    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA journal_mode=WAL');return db

    @staticmethod
    def identity(signal):
        fields={k:signal.get(k) for k in ('text','timestamp','source','actual_nlp_backend','model_revisions','sentiment_model_version','impact_method','risk_engine_version','risk_factor','assertion_status')}
        return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()

    def save(self,signals):
        now=datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self.connect() as db:
            for signal in signals:
                db.execute('INSERT OR IGNORE INTO signals VALUES (?,?,?)',(self.identity(signal),now,json.dumps(signal)))
            db.execute('DELETE FROM signals WHERE key NOT IN (SELECT key FROM signals ORDER BY observed_at DESC,key DESC LIMIT 5000)')

    def list(self,limit=100):
        limit=max(1,min(500,int(limit)))
        with self.connect() as db:
            rows=db.execute('SELECT key,observed_at,payload FROM signals ORDER BY observed_at DESC,key DESC LIMIT ?',(limit,)).fetchall()
        return [{'storage_id':key,'observed_at':when,'signal':json.loads(payload)} for key,when,payload in rows]
