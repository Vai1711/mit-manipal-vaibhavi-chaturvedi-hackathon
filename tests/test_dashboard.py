from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_dashboard_event_selection_and_signed_shock():
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30).run()
    assert not at.exception
    assert {w.label:w.value for w in at.metric}['Net market impact']=='$21.95M loss'
    next(w for w in at.slider if w.label=='Rate shift · basis points').set_value(-100).run()
    assert not at.exception
    assert {w.label:w.value for w in at.metric}['Net market impact']=='$0.00M'
    next(w for w in at.checkbox if w.label=='Run manual what-if scenario').check().run()
    assert {w.label:w.value for w in at.metric}['Net market impact']=='$21.95M gain'
    next(w for w in at.selectbox if w.label=='Select an event').select('event-rite-aid').run()
    assert not at.exception
    metrics={w.label:w.value for w in at.metric}
    assert metrics['Stressed assets']=='2'
    assert metrics['Expected credit loss increase']=='$2.70M'
    next(w for w in at.button if w.label=='2 · Rate cut').click().run()
    assert not at.exception
    assert {w.label:w.value for w in at.metric}['Net market impact']=='$21.95M gain'


def test_live_mode_waits_for_explicit_fetch_and_shows_snapshot():
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30).run()
    next(w for w in at.radio if w.label=='Source mode').set_value('Live RSS Feed').run()
    assert not at.exception
    assert any('Live fetching is off' in w.value for w in at.info)
    assert not next(w for w in at.checkbox if w.label=='Refresh every 5 minutes').value
