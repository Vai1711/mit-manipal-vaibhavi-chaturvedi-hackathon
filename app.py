"""Risk Radar: compact visual overview, explicit provenance and on-demand inference."""
import html,json
from datetime import datetime,timezone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.pipeline import ROOT,load_portfolio,run_pipeline
from src.risk_engine import RiskEngine
from src.stress_engine import run_scenario
from src.source_manager import collect_sources
from src.store import SignalStore
from src.comparison import compare_scenarios

st.set_page_config(page_title='Risk Radar',page_icon='◉',layout='wide')
st.markdown('''<style>
.block-container{max-width:1450px;padding-top:3rem;}
.hero{font-size:34px;font-weight:700;color:#e5edf8;letter-spacing:-.6px;}
.subtitle{color:#9baec8;margin:5px 0 18px;}
div[data-testid="stMetric"]{background:#121e32;border:1px solid #283a55;padding:15px;border-radius:10px;}
.evidence{background:#121e32;border-left:3px solid #56d9c5;padding:15px;border-radius:5px;line-height:1.65;}
</style>''',unsafe_allow_html=True)
portfolio=load_portfolio()


@st.cache_resource
def get_engine(mode):return RiskEngine(portfolio,mode)


@st.cache_data(max_entries=12,show_spinner=False)
def analyze(mode,payload):return run_pipeline(mode,json.loads(payload),get_engine(mode))


def money(value):return f'${abs(value)/1e6:,.2f}M'


def impact(value):return '$0.00M' if abs(value)<.005 else money(value)+(' gain' if value>0 else ' loss')


def style(fig,height=280):
    fig.update_layout(template='plotly_dark',height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20,r=20,t=20,b=25),font_color='#dce8f7')
    return fig


def refresh_live():
    batch=collect_sources();st.session_state['live_batch']=batch
    previous=st.session_state.get('last_success',{})
    for status in batch['source_status']:
        if status['last_successful_fetch']:previous[status['source_type']]=status['last_successful_fetch']
    st.session_state['last_success']=previous


with st.sidebar:
    st.title('◉ Risk Radar')
    source_mode=st.radio('Source mode',['Historical public replay','Snapshot Replay','Live RSS Feed','Synthetic replay'])
    mode_label=st.selectbox('Analysis engine',['Lightweight AI · recommended','Local hybrid AI · on demand','Rules baseline'])
    mode={'Lightweight AI · recommended':'cached','Local hybrid AI · on demand':'hybrid','Rules baseline':'baseline'}[mode_label]
    if mode=='cached':st.caption('Trained sentiment plus recorded semantic event predictions. New text uses labelled evidence rules; ambiguous or unverified reports need review.')
    elif mode=='hybrid':st.caption('Optional local models, 2 CPU threads. No training or automatic downloads.')
    if source_mode=='Live RSS Feed':
        if st.button('Refresh live sources',width='stretch'):
            with st.spinner('Fetching up to five reports per source…'):refresh_live()
        periodic=st.checkbox('Refresh every 5 minutes',value=False)
        if periodic:
            @st.fragment(run_every=300)
            def poll():
                refresh_live();st.rerun()
            poll()
        if 'live_batch' in st.session_state:
            batch=st.session_state['live_batch']
            for status in batch['source_status']:
                st.caption(f"{status['source_type'].title()}: {status['status']} ({status['count']} reports)")
                previous=st.session_state.get('last_success',{}).get(status['source_type'])
                if previous:st.caption('Last success: '+previous)
                if status['error']:st.warning('Live fetch failed; saved snapshot in use. '+status['error'])
    with st.expander('Analyze your own text'):
        with st.form('text_form'):
            text=st.text_area('Financial report',placeholder='Mention a company, sector or central-bank decision…',max_chars=6000)
            submitted=st.form_submit_button('Analyze text')
        if submitted and text.strip():
            st.session_state['manual_items']=[{'id':'manual','text':text.strip(),'source':'Analyst input','source_type':'manual',
                'timestamp':datetime.now(timezone.utc).isoformat(),'synthetic':False,'timestamp_quality':'Ingestion time, not publication time'}]
        if st.session_state.get('manual_items') and st.button('Clear custom text'):del st.session_state['manual_items']
    st.caption('CPU work stops when no new input is processed. Close the app with Ctrl+C.')

st.markdown('<div class="hero">Event-to-Exposure Risk Radar</div><div class="subtitle">Financial events, mapped exposures and explainable portfolio scenarios</div>',unsafe_allow_html=True)
files={'Historical public replay':'public_replay.json','Snapshot Replay':'public_feed_snapshot.json','Synthetic replay':None}
if source_mode=='Live RSS Feed':
    batch=st.session_state.get('live_batch')
    if not batch:
        st.info('Live fetching is off until you press Refresh live sources. Showing the saved snapshot.')
        items=json.loads((ROOT/'data/public_feed_snapshot.json').read_text(encoding='utf-8'))
        delivery='saved snapshot'
    else:items=batch['items'];delivery=batch['delivery_mode'].replace('_',' ')
elif source_mode=='Synthetic replay':
    items=[]
    for name in ['news_feed.json','social_feed.json']:items+=json.loads((ROOT/'data'/name).read_text(encoding='utf-8'))
    delivery='authored synthetic replay'
else:
    items=json.loads((ROOT/'data'/files[source_mode]).read_text(encoding='utf-8'));delivery=source_mode.lower()
if st.session_state.get('manual_items'):items+=st.session_state['manual_items']
try:
    with st.spinner('Analyzing new text…'):output=analyze(mode,json.dumps(items,sort_keys=True))
except Exception as exc:
    st.error(f'Analysis unavailable: {type(exc).__name__}: {exc}')
    st.info('Choose Recorded AI or Rules baseline, or follow the optional local-model setup in README.')
    st.stop()
SignalStore(ROOT/'outputs/signals.sqlite').save(output['signals'])
events=output['events']
backends={s['actual_nlp_backend'] for s in output['signals']}
backend_label='recorded AI predictions' if backends=={'learned_sentiment+recorded_event'} else 'local compact AI' if backends=={'learned_sentiment+local_event'} else 'rules' if backends=={'rules'} else 'learned sentiment / mixed event backends'
st.caption(f'{delivery.title()} · {backend_label} · hypothetical holdings · assumed shocks')
if not events:st.info('No usable reports.');st.stop()
st.sidebar.caption(f"{len(output['signals'])} reports / {len(events)} events / {sum(bool(e['review_reasons']) for e in events)} need review")


def label(event_id):
    e=next(e for e in events if e['id']==event_id)
    subject=', '.join(e['entities'] or e['sectors']) or {'rates_up':'Rate rise','rates_down':'Rate cut'}.get(e['risk_factor'],'General report')
    return f"{e['event_classification']} · {subject} · {e['impact_score']}/10"


if source_mode=='Historical public replay':
    with st.expander('Guided demo · four scenarios'):
        cols=st.columns(4)
        for col,(title,id,shock) in zip(cols,[('1 · Rate rise','fed-rise',100),('2 · Rate cut','fed-cut',-100),('3 · Borrower credit','rite-aid',100),('4 · Oil sanctions','oil-sanctions',100)]):
            if col.button(title):
                st.session_state['selected_event']='event-'+id;st.session_state['rate_shock']=shock
        st.caption('Historical releases applied to one hypothetical portfolio. Shock magnitudes are analyst assumptions.')
ids=[e['id'] for e in events]
if st.session_state.get('selected_event') not in ids:st.session_state['selected_event']='event-fed-rise' if 'event-fed-rise' in ids else ids[0]
event_id=st.selectbox('Select an event',ids,format_func=label,key='selected_event')
event=next(e for e in events if e['id']==event_id);report=event['signals'][0]
overview,lab,evidence=st.tabs(['Overview','Scenario Lab','Evidence & Review'])

with lab:
    st.subheader('Scenario assumptions')
    a,b=st.columns(2)
    with a:
        rate=st.slider('Rate shift · basis points',-300,300,100,25,key='rate_shock')
        pd_mult=st.slider('Probability of default multiplier',1.,10.,3.,.5,key='pd_shock')
    with b:
        spread=st.slider('Credit spread widening · bps',0,500,150,25,key='spread_shock')
        equity=st.slider('Exposed equity decline · %',0,50,10,5,key='equity_shock')/100
    manual=st.checkbox('Run manual what-if scenario',key=f'manual_{event_id}',help='Overrides the trigger gate, not source verification.')
    result=run_scenario(portfolio,event,rate,spread,pd_mult,equity,manual)
    if result['trigger_note']:st.warning(result['trigger_note'])
    st.caption(f"{result['status'].capitalize()}. One basis point = 0.01 percentage points.")
    a,b,c=st.columns(3)
    a.metric('Without hedge',impact(result['unhedged_market_pnl']));b.metric('Hedge contribution',impact(result['hedge_pnl']));c.metric('With hedge',impact(result['hedged_market_pnl']))
    fig=go.Figure(go.Waterfall(x=['Bond / equity impact','Swap contribution','Net market impact'],measure=['relative','relative','total'],
        y=[result['unhedged_market_pnl']/1e6,result['hedge_pnl']/1e6,0],decreasing={'marker':{'color':'#fb7185'}},increasing={'marker':{'color':'#56d9c5'}},totals={'marker':{'color':'#82a9ee'}}))
    fig.update_layout(yaxis_title='USD millions');st.plotly_chart(style(fig,320),width='stretch')
    with st.expander('Exposure heatmap across events'):
        counts=[]
        for e in events:
            r=run_scenario(portfolio,e,rate_bps=-100 if e['risk_factor']=='rates_down' else 100,manual=True)
            counts.append([sum(x['type']==t and x['exposure_reason']!='Not affected' for x in r['rows']) for t in ['bond','loan','equity','swap']])
        fig=go.Figure(go.Heatmap(z=counts,x=['Bonds','Loans','Equities','Swap'],y=[label(e['id']) for e in events],colorscale=[[0,'#121e32'],[1,'#56d9c5']],text=counts,texttemplate='%{text}',showscale=False))
        st.plotly_chart(style(fig,max(240,min(650,len(events)*35))),width='stretch')
        st.caption('Mapped asset counts under explicit what-if. This ignores trigger gates to display scope, not confirmed losses.')
    with st.expander('Compare with a blanket shock'):
        comp=compare_scenarios(portfolio,event,rate_bps=rate,spread_bps=spread,pd_multiplier=pd_mult,geopolitical_decline=equity,manual=manual)
        blank=comp['blanket']
        st.dataframe(pd.DataFrame([{'Approach':'Exposure-aware','Market impact':impact(result['hedged_market_pnl']),'ECL increase':money(result['ecl_increase']),'Stressed assets':result['affected_asset_count']},
          {'Approach':'Blanket counterfactual','Market impact':impact(blank['market_pnl']),'ECL increase':money(blank['ecl_increase']),'Stressed assets':blank['affected_asset_count']}]),hide_index=True,width='stretch')
        st.caption(comp['assumptions'])
        st.write('Assets changed only by blanket mapping: '+(', '.join(comp['blanket_only_asset_ids']) or 'None'))
    st.download_button('Download this scenario',json.dumps({'event':event,'result':result},indent=2),'risk_scenario.json','application/json')

with overview:
    st.write('**'+report['supporting_evidence']+'**')
    st.caption(f"{report['source']} · {report['timestamp'][:10]} · Severity {event['impact_score']}/10 · Text sentiment {event['sentiment_score']:+.2f} · Policy stance: {report['policy_stance']}")
    if event['review_reasons'] and not manual:st.warning('Automatic stress paused for review. Evidence & Review explains why. An explicit what-if remains available.')
    elif result['status']=='not triggered':st.info('No automatic stress: severity, supported factor or shock direction does not satisfy the trigger.')
    elif manual:st.info('Manual what-if active. This does not verify the event.')
    a,b,c=st.columns(3)
    a.metric('Net market impact',impact(result['hedged_market_pnl']));b.metric('Stressed assets',result['affected_asset_count']);c.metric('Expected credit loss increase',money(result['ecl_increase']))
    left,right=st.columns([1,1.25])
    with left:
        st.subheader('Market-valued portfolio')
        fig=go.Figure(go.Bar(x=['Before','After'],y=[result['market_value_before']/1e6,result['market_value_after']/1e6],marker_color=['#82a9ee','#56d9c5'],
            text=[money(result['market_value_before']),money(result['market_value_after'])],textposition='auto'))
        fig.update_layout(yaxis_title='USD millions');st.plotly_chart(style(fig),width='stretch')
    affected=[r for r in result['rows'] if r['exposure_reason']!='Not affected']
    with right:
        st.subheader('Per-asset market impact')
        if affected:
            fig=go.Figure(go.Bar(y=[r['asset_id']+' · '+r['entity'] for r in affected],x=[r['market_pnl']/1e6 for r in affected],orientation='h',marker_color=['#fb7185' if r['market_pnl']<0 else '#56d9c5' for r in affected]))
            fig.update_layout(xaxis_title='USD millions',yaxis={'autorange':'reversed'});st.plotly_chart(style(fig),width='stretch')
        else:st.info('No assets receive a shock for the current event and trigger status.')
    with st.expander('Affected holdings and separate credit losses'):
        if affected:st.dataframe(pd.DataFrame([{k:r[k] for k in ['asset_id','entity','type','market_pnl','ecl_increase','exposure_reason']} for r in affected]),hide_index=True,width='stretch')
        else:st.caption('No affected holdings.')
        st.write('One-year expected loss proxy: '+money(result['baseline_ecl'])+' before, '+money(result['stressed_ecl'])+' after.')
    with st.expander('Event timeline'):
        timeline=pd.DataFrame([{'date':e['timestamp'],'severity':e['impact_score'],'label':label(e['id']),'review':'Needs review' if e['review_reasons'] else 'No flag'} for e in events])
        fig=go.Figure(go.Scatter(x=timeline['date'],y=timeline['severity'],mode='markers',text=timeline['label'],marker={'size':13,'color':timeline['severity'],'colorscale':'Tealrose','cmin':1,'cmax':10},hovertemplate='%{text}<br>%{x}<extra></extra>'))
        fig.update_layout(yaxis={'range':[0,11],'title':'Severity rubric'},xaxis_title='Report publication / supplied date');st.plotly_chart(style(fig,230),width='stretch')
        st.caption('Historical replay spans multiple years. Publication timestamps do not establish independent confirmation.')
    with st.expander('Assumptions and model boundaries'):
        st.write('All holdings, valuations, sensitivities, PD and LGD are hypothetical. Market value excludes loan EAD and swap notional. Bond repricing uses first-order duration; swap response uses signed linear sensitivity. Loan ECL is a separate one-year EAD × PD × LGD proxy, not IFRS 9 staging.')
        st.write('Severity is an auditable scenario-priority rubric, not a learned market-move forecast. Text sentiment and hawkish/dovish stance answer different questions. Hybrid routing anchors explicit event evidence with deterministic controls. The trained financial sentiment scorer supplies probabilities, and the semantic event model resolves other categories.')
        st.write('Replay AI outputs are recorded model inference. New text uses clearly labelled evidence rules. Explicit supported actions can trigger scenarios; ambiguous reports need review. Model scores are uncalibrated. Social-only, speculative, denied and conflicting events pause automatic stress.')

with evidence:
    st.subheader('Evidence and decision trace')
    for signal in event['signals']:
        with st.expander(signal['source']+' · '+signal['timestamp'][:10],expanded=signal is report):
            st.markdown('<div class="evidence">'+html.escape(signal['text'])+'</div>',unsafe_allow_html=True)
            if signal.get('url','').startswith(('https://','http://')):st.link_button('Open source',signal['url'])
            st.caption(f"Backend: {signal['actual_nlp_backend']} / Route: {signal['classification_route']} / Assertion: {signal['assertion_status']}")
            st.json({k:signal.get(k) for k in ['entities','entity_matches','risk_factor','risk_currency','policy_stance','impact_breakdown','model_event_classification','event_model_score','sentiment_probabilities','review_reasons','delivery_mode']},expanded=False)
    with st.expander('Review queue and copies'):
        rows=[{'event':e['id'],'reasons':'; '.join(e['review_reasons'])} for e in events if e['review_reasons']]
        if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        else:st.caption('No current review flags.')
        st.json(output['duplicates'],expanded=False)
    with st.expander('Per-asset calculation audit'):
        st.dataframe(pd.DataFrame(result['rows']),hide_index=True,width='stretch')
        st.dataframe(pd.DataFrame(portfolio['assets']),hide_index=True,width='stretch')
    with st.expander('Measured evaluation'):
        path=ROOT/'docs/benchmark-results.json'
        if path.exists():
            benchmark=json.loads(path.read_text(encoding='utf-8'))
            for section in ['sentiment','events']:
                st.write('**'+section.title()+' benchmark**')
                st.dataframe(pd.DataFrame(benchmark[section]['summary']),hide_index=True,width='stretch')
                st.caption(benchmark[section]['limitation'])
            st.download_button('Download complete benchmark',path.read_bytes(),'benchmark-results.json','application/json')
        else:st.info('Run scripts/benchmark.py to create measured results.')
        st.caption('Benchmark execution is a developer action, so a dashboard click cannot start a long model workload.')
    with st.expander('Impact rubric versus historical SPY moves'):
        path=ROOT/'docs/impact-check.json'
        if path.exists():
            check=json.loads(path.read_text(encoding='utf-8'))
            if check.get('available'):
                fig=go.Figure(go.Scatter(x=[r['severity'] for r in check['rows']],y=[abs(r['spy_return_percent']) for r in check['rows']],text=[r['event_id'] for r in check['rows']],mode='markers',marker={'size':12,'color':'#82a9ee'},hovertemplate='%{text}<br>Score %{x}, |return| %{y:.2f}%<extra></extra>'))
                fig.update_layout(xaxis_title='Scenario-priority score',yaxis_title='Absolute SPY daily return (%)');st.plotly_chart(style(fig,260),width='stretch')
                st.caption(check['note'])
    with st.expander('Persisted signal history and REST API'):
        history=SignalStore(ROOT/'outputs/signals.sqlite').list(100)
        st.dataframe(pd.DataFrame([{'observed_at':r['observed_at'],'source':r['signal']['source'],'category':r['signal']['event_classification'],'sentiment':r['signal']['sentiment_score'],'severity':r['signal']['impact_score'],'backend':r['signal']['actual_nlp_backend']} for r in history]),hide_index=True,width='stretch')
        st.code('./.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000')
        st.caption('Optional local API: POST /analyze, GET /signals, GET /health. Interactive documentation at localhost:8000/docs. No server starts automatically.')
    st.download_button('Download structured signals',json.dumps(output,indent=2),'risk_signals.json','application/json')
