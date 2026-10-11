from fastapi.testclient import TestClient
from api import app


def test_api_analysis_persists_required_fields(tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_DB_PATH',str(tmp_path/'api.sqlite'))
    with TestClient(app) as client:
        assert client.get('/health').json()['status']=='ok'
        response=client.post('/analyze',json={'text':'Federal Reserve raises interest rates.','mode':'baseline'})
        assert response.status_code==200
        signal=response.json()['signals'][0]
        assert -1<=signal['sentiment_score']<=1 and 1<=signal['impact_score']<=10
        assert signal['event_classification']=='Macroeconomic'
        assert len(client.get('/signals').json()['signals'])==1


def test_api_rejects_invalid_text_and_timestamp(tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_DB_PATH',str(tmp_path/'api.sqlite'))
    with TestClient(app) as client:
        assert client.post('/analyze',json={'text':' '}).status_code==422
        assert client.post('/analyze',json={'text':'Fed raises rates.','timestamp':'not a date','mode':'baseline'}).status_code==422
        assert client.get('/signals?limit=501').status_code==422
