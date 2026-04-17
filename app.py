"""
한국주식 분석 시스템 Pro (v9 - Cloud 호환)
pykrx 없어도 yfinance로 동작, 있으면 KRX 데이터 활용
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings, os
warnings.filterwarnings('ignore')

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# pykrx 안전 import
try:
    from pykrx import stock as krx_stock
    HAS_KRX = True
except Exception:
    HAS_KRX = False

# Streamlit Cloud Secrets
try:
    if 'KRX_ID' in st.secrets:
        os.environ['KRX_ID'] = st.secrets['KRX_ID']
    if 'KRX_PW' in st.secrets:
        os.environ['KRX_PW'] = st.secrets['KRX_PW']
except Exception:
    pass

st.set_page_config(page_title="한국주식 분석 Pro", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header {font-size:2.5rem;font-weight:bold;text-align:center;
background:linear-gradient(90deg,#FF6B6B 0%,#4ECDC4 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1rem;}
.warning-box {background-color:#fff3cd;border-left:4px solid #ffc107;
padding:1rem;border-radius:5px;margin:1rem 0;color:#333;}
</style>
""", unsafe_allow_html=True)

POPULAR_STOCKS = {
    '삼성전자': ('005930', '005930.KS'),
    'SK하이닉스': ('000660', '000660.KS'),
    '카카오': ('035720', '035720.KS'),
    'NAVER': ('035420', '035420.KS'),
    'LG에너지솔루션': ('373220', '373220.KS'),
    '현대차': ('005380', '005380.KS'),
    '셀트리온': ('068270', '068270.KS'),
    '포스코홀딩스': ('005490', '005490.KS'),
    'KB금융': ('105560', '105560.KS'),
    '삼성바이오로직스': ('207940', '207940.KS'),
    '제주항공': ('089590', '089590.KS'),
    '알테오젠': ('196170', '196170.KQ'),
    'LG화학': ('051910', '051910.KS'),
    '삼성SDI': ('006400', '006400.KS'),
    '기아': ('000270', '000270.KS'),
}

SECTOR_GROUPS = {
    '반도체': ['005930.KS', '000660.KS'],
    'IT/플랫폼': ['035420.KS', '035720.KS'],
    '자동차': ['005380.KS', '000270.KS'],
    '바이오': ['068270.KS', '207940.KS', '196170.KQ'],
    '에너지/소재': ['373220.KS', '051910.KS', '006400.KS', '005490.KS'],
    '금융': ['105560.KS'],
    '항공': ['089590.KS'],
}

MACRO_SYMBOLS = {
    'USD/KRW': 'KRW=X', 'S&P500': '^GSPC', 'KOSPI': '^KS11',
    'WTI유가': 'CL=F', '금': 'GC=F', '미국10년국채': '^TNX', 'VIX': '^VIX',
}


def resolve_ticker(user_input):
    if user_input in POPULAR_STOCKS:
        krx, yf_sym = POPULAR_STOCKS[user_input]
        return krx, yf_sym, user_input
    if user_input.isdigit() and len(user_input) == 6:
        yf_sym = f"{user_input}.KS"
        try:
            t = yf.Ticker(yf_sym)
            if not t.history(period="5d").empty:
                return user_input, yf_sym, user_input
        except: pass
        yf_sym = f"{user_input}.KQ"
        try:
            t = yf.Ticker(yf_sym)
            if not t.history(period="5d").empty:
                return user_input, yf_sym, user_input
        except: pass
    return None, None, user_input


# === 주가 데이터: KRX 우선, 없으면 yfinance ===
@st.cache_data(ttl=3600, show_spinner=False)
def get_price(krx_code, yf_sym, days):
    # KRX 시도
    if HAS_KRX:
        try:
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            df = krx_stock.get_market_ohlcv(start, end, krx_code)
            if not df.empty:
                df.columns = ['Open','High','Low','Close','Volume','Change']
                return df, 'KRX'
        except: pass
    # yfinance 폴백
    try:
        t = yf.Ticker(yf_sym)
        df = t.history(start=datetime.now()-timedelta(days=days), end=datetime.now())
        if not df.empty:
            df = df[['Open','High','Low','Close','Volume']]
            return df, 'Yahoo'
    except: pass
    return pd.DataFrame(), 'N/A'


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_krx(krx_code):
    if not HAS_KRX: return {}
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        df = krx_stock.get_market_fundamental(start, end, krx_code)
        if not df.empty:
            v = df[df.sum(axis=1) > 0]
            if not v.empty: return v.iloc[-1].to_dict()
    except: pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_cap_krx(krx_code):
    if not HAS_KRX: return {}
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        df = krx_stock.get_market_cap(start, end, krx_code)
        if not df.empty:
            v = df[df['시가총액'] > 0]
            if not v.empty: return v.iloc[-1].to_dict()
    except: pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_investor_krx(krx_code):
    if not HAS_KRX: return pd.DataFrame()
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        df = krx_stock.get_market_trading_value_by_date(start, end, krx_code)
        if not df.empty: return df
    except: pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def yf_info(sym):
    try: return yf.Ticker(sym).info
    except: return {}


@st.cache_data(ttl=1800, show_spinner=False)
def yf_news(sym):
    pos_words = ['surge','soar','jump','rally','gain','rise','profit','growth',
                'beat','exceed','record','high','upgrade','buy','strong','boom',
                'breakthrough','innovation','success','expand','bullish','positive',
                'dividend','partnership','deal','approval','launch','recovery']
    neg_words = ['fall','drop','decline','loss','crash','plunge','tumble','miss',
                'fail','weak','downgrade','sell','risk','concern','warning','cut',
                'layoff','lawsuit','investigation','scandal','bearish','negative',
                'recession','debt','default','bankruptcy','fraud','penalty']
    try:
        raw = yf.Ticker(sym).news
        if not raw: return [], 0, "중립 🟡"
        nl = raw if isinstance(raw, list) else raw.get('news', raw.get('items', []))
        results = []; total = 0
        for item in nl[:10]:
            title = ''
            for k in ['title','headline','name']:
                title = item.get(k,'') or ''; 
                if title: break
            if not title:
                c = item.get('content',{})
                if isinstance(c,dict): title = c.get('title','') or ''
            if not title: title = '(제목없음)'
            pub = ''
            for k in ['publisher','source']:
                pub = item.get(k,'') or ''
                if pub: break
            if not pub:
                p = item.get('provider',{})
                if isinstance(p,dict): pub = p.get('displayName','') or p.get('name','') or ''
            if not pub:
                c = item.get('content',{})
                if isinstance(c,dict):
                    p2 = c.get('provider',{})
                    if isinstance(p2,dict): pub = p2.get('displayName','') or ''
            if not pub: pub = '알 수 없음'
            link = ''
            for k in ['link','url']:
                link = item.get(k,'') or ''
                if link: break
            if not link:
                cn = item.get('canonicalUrl',{})
                if isinstance(cn,dict): link = cn.get('url','') or ''
            if not link:
                c = item.get('content',{})
                if isinstance(c,dict):
                    cn2 = c.get('canonicalUrl',{})
                    if isinstance(cn2,dict): link = cn2.get('url','') or ''
            if not link: link = '#'
            dt = 'N/A'
            pt = item.get('providerPublishTime',0) or item.get('publishedAt',0)
            if not pt:
                c = item.get('content',{})
                if isinstance(c,dict): pt = c.get('pubDate','') or ''
            if isinstance(pt,(int,float)) and pt > 1e6:
                dt = datetime.fromtimestamp(pt).strftime('%Y-%m-%d %H:%M')
            elif isinstance(pt,str) and len(pt)>5: dt = pt[:16]
            tl = title.lower()
            ps = sum(1 for w in pos_words if w in tl)
            ns = sum(1 for w in neg_words if w in tl)
            sc = ps - ns; total += sc
            sent = "긍정 🟢" if sc>0 else ("부정 🔴" if sc<0 else "중립 🟡")
            results.append({'title':title,'publisher':pub,'date':dt,'link':link,'sentiment':sent,'score':sc})
        ov = "긍정적 🟢" if total>2 else ("부정적 🔴" if total<-2 else "중립 🟡")
        return results, total, ov
    except: return [], 0, "분석불가"


@st.cache_data(ttl=3600, show_spinner=False)
def yf_macro():
    macro = {}
    for name, sym in MACRO_SYMBOLS.items():
        try:
            t = yf.Ticker(sym)
            df = t.history(period="3mo")
            if not df.empty and 'Close' in df.columns:
                cur = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2]) if len(df)>1 else cur
                chg = ((cur-prev)/prev)*100 if prev else 0
                m0 = float(df['Close'].iloc[0])
                mchg = ((cur-m0)/m0)*100 if m0 else 0
                macro[name] = {'current':cur,'daily_change':round(chg,2),'monthly_change':round(mchg,2),'history':df['Close']}
        except: continue
    return macro


def technical_analysis(df):
    for w in [5,20,60,120]: df[f'MA{w}'] = df['Close'].rolling(w).mean()
    d = df['Close'].diff()
    g = d.where(d>0,0).rolling(14).mean()
    l = -d.where(d<0,0).rolling(14).mean()
    df['RSI'] = 100-(100/(1+g/l))
    df['EMA_short'] = df['Close'].ewm(span=12).mean()
    df['EMA_long'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_short']-df['EMA_long']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD']-df['MACD_signal']
    df['BB_mid'] = df['Close'].rolling(20).mean()
    s = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid']+(s*2)
    df['BB_lower'] = df['BB_mid']-(s*2)
    return df


def build_fundamental(kf, kc, yi):
    per = kf.get('PER',0) or yi.get('trailingPE',0) or yi.get('forwardPE',0) or 0
    pbr = kf.get('PBR',0) or yi.get('priceToBook',0) or 0
    eps = kf.get('EPS',0) or yi.get('trailingEps',0) or yi.get('epsCurrentYear',0) or 0
    bps = kf.get('BPS',0) or yi.get('bookValue',0) or 0
    if eps==0:
        ni=yi.get('netIncomeToCommon',0)or 0; sh=yi.get('sharesOutstanding',0)or 0
        if ni>0 and sh>0: eps=ni/sh
    if pbr==0 and bps>0:
        cp=yi.get('currentPrice',0)or yi.get('previousClose',0)or 0
        if cp>0: pbr=cp/bps
    dk = kf.get('DIV',0) or 0
    if dk>0: div=dk
    else:
        dr=yi.get('dividendYield',0)or 0; div=dr if dr>1 else dr*100
    rr=yi.get('returnOnEquity',0)or 0; roe=rr if rr>1 else rr*100
    if roe==0 and eps>0 and bps>0: roe=(eps/bps)*100
    cap=kc.get('시가총액',0)or yi.get('marketCap',0)or 0
    return {
        'PER':round(per,2)if per else 0,'PBR':round(pbr,2)if pbr else 0,
        'EPS':round(eps,0)if eps else 0,'BPS':round(bps,0)if bps else 0,
        'DIV':round(div,2),'ROE':round(roe,2),'시가총액':cap,
        '부채비율':round((yi.get('debtToEquity',0)or 0),2),
        '매출성장률':round((yi.get('revenueGrowth',0)or 0)*100,2),
        '이익률':round((yi.get('profitMargins',0)or 0)*100,2),
        'src_PER':'KRX' if kf.get('PER',0) else 'Yahoo',
        'src_PBR':'KRX' if kf.get('PBR',0) else 'Yahoo',
        '업종':yi.get('industry','N/A'),'섹터':yi.get('sector','N/A'),
        '회사명':yi.get('longName',yi.get('shortName','N/A')),
    }


def evaluate(fund, news_sc=0, inst_pct=0):
    score=50; a={}
    per=fund.get('PER',0)or 0; pbr=fund.get('PBR',0)or 0; roe=fund.get('ROE',0)or 0
    div=fund.get('DIV',0)or 0; eps=fund.get('EPS',0)or 0; bps=fund.get('BPS',0)or 0
    debt=fund.get('부채비율',0)or 0; growth=fund.get('매출성장률',0)or 0
    if 0<per<=10: score+=15; a['PER']='✅ 저평가'
    elif 10<per<=15: score+=8; a['PER']='🟢 적정'
    elif 15<per<=25: a['PER']='🟡 평균'
    elif per>25: score-=10; a['PER']='🔴 고평가'
    else: a['PER']='⚠️ N/A'
    if 0<pbr<=1: score+=15; a['PBR']='✅ 저평가'
    elif 1<pbr<=2: score+=5; a['PBR']='🟢 적정'
    elif 2<pbr<=3: a['PBR']='🟡 평균'
    elif pbr>3: score-=8; a['PBR']='🔴 고평가'
    else: a['PBR']='⚠️ N/A'
    if roe>=15: score+=15; a['ROE']='✅ 매우우수'
    elif roe>=10: score+=10; a['ROE']='🟢 우수'
    elif roe>=5: score+=3; a['ROE']='🟡 보통'
    elif roe>0: score-=5; a['ROE']='🔴 부진'
    if div>=4: score+=5; a['배당']='✅ 고배당'
    elif div>=2: score+=2; a['배당']='🟢 중배당'
    if 0<debt<=50: score+=3; a['부채']='✅ 안정'
    elif debt>100: score-=5; a['부채']='🔴 높음'
    if growth>=20: score+=5; a['성장']='✅ 고성장'
    elif growth>=10: score+=3; a['성장']='🟢 양호'
    elif growth<0: score-=3; a['성장']='🔴 역성장'
    if news_sc>2: score+=5; a['뉴스']='✅ 긍정'
    elif news_sc<-2: score-=5; a['뉴스']='🔴 부정'
    else: a['뉴스']='🟡 중립'
    if inst_pct>=30: score+=3; a['기관']='✅ 관심높음'
    if eps>0 and bps>0: a['적정주가']=f'{(22.5*eps*bps)**0.5:,.0f}원'
    return a, max(0,min(100,score))


def ml_predict(df):
    d=df.copy()
    d['Returns']=d['Close'].pct_change()
    d['Volatility']=d['Returns'].rolling(20).std()
    d['Price_Range']=(d['High']-d['Low'])/d['Close']
    d['Vol_Change']=d['Volume'].pct_change()
    d['Momentum']=d['Close']/d['Close'].shift(20)-1
    d['Target']=d['Close'].shift(-5)/d['Close']-1
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    feats=['MA5','MA20','MA60','RSI','MACD','MACD_signal','BB_upper','BB_lower',
           'Volume','Volatility','Price_Range','Vol_Change','Momentum']
    feats=[f for f in feats if f in d.columns]
    if len(d)<50: return None
    X=d[feats].values; y=d['Target'].values
    mask=np.isfinite(X).all(axis=1)&np.isfinite(y); X=X[mask]; y=y[mask]
    if len(X)<50: return None
    sc=StandardScaler(); Xs=sc.fit_transform(X); sp=int(len(Xs)*0.8)
    m=RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1)
    m.fit(Xs[:sp],y[:sp]); r2=m.score(Xs[sp:],y[sp:])
    pred=m.predict(sc.transform(d[feats].iloc[-1:].values))[0]
    cp=d['Close'].iloc[-1]
    imp=dict(sorted(zip(feats,m.feature_importances_),key=lambda x:x[1],reverse=True))
    return {'r2':r2,'ret':pred*100,'pred_price':cp*(1+pred),'base_price':cp,'importance':imp}


def make_chart(df, name, sym):
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=0.05,
                      row_heights=[0.5,0.15,0.175,0.175],
                      subplot_titles=(f'{name} ({sym})','거래량','RSI','MACD'))
    fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
                  low=df['Low'],close=df['Close'],name='주가',
                  increasing_line_color='#ff4444',decreasing_line_color='#4444ff'),row=1,col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['MA20'],name='MA20',line=dict(color='orange',width=1)),row=1,col=1)
    if 'MA60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['MA60'],name='MA60',line=dict(color='purple',width=1)),row=1,col=1)
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['BB_upper'],name='BB상단',line=dict(color='gray',width=1,dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['BB_lower'],name='BB하단',line=dict(color='gray',width=1,dash='dot'),fill='tonexty',fillcolor='rgba(128,128,128,0.1)'),row=1,col=1)
    cl=['red' if df['Close'].iloc[i]>=df['Open'].iloc[i] else 'blue' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name='거래량',marker_color=cl,opacity=0.7),row=2,col=1)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['RSI'],name='RSI',line=dict(color='purple',width=1.5)),row=3,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="red",row=3,col=1)
        fig.add_hline(y=30,line_dash="dash",line_color="green",row=3,col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD',line=dict(color='blue',width=1.5)),row=4,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD_signal'],name='Signal',line=dict(color='red',width=1.5)),row=4,col=1)
        mc=['green' if x>=0 else 'red' for x in df['MACD_hist']]
        fig.add_trace(go.Bar(x=df.index,y=df['MACD_hist'],name='Hist',marker_color=mc,opacity=0.5),row=4,col=1)
    fig.update_layout(height=900,showlegend=True,xaxis_rangeslider_visible=False,hovermode='x unified',template='plotly_white')
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def sector_compare(yf_sym):
    target=None
    for sec,syms in SECTOR_GROUPS.items():
        if yf_sym in syms: target=sec; break
    if not target or len(SECTOR_GROUPS[target])<2: return None,None
    prices={}
    for s in SECTOR_GROUPS[target]:
        try:
            df=yf.Ticker(s).history(period="6mo")
            if not df.empty:
                nm=s.split('.')[0]
                for k,v in POPULAR_STOCKS.items():
                    if v[1]==s: nm=k; break
                prices[nm]=df['Close'].pct_change().dropna()
        except: continue
    if len(prices)<2: return None,None
    dfr=pd.DataFrame(prices)
    return dfr.corr(), {n:round(((1+r).prod()-1)*100,2) for n,r in prices.items()}


def main():
    st.markdown('<h1 class="main-header">📈 한국주식 분석 Pro</h1>',unsafe_allow_html=True)
    mode = "KRX + Yahoo" if HAS_KRX else "Yahoo Finance"
    st.markdown(f"### {mode} 하이브리드 분석")
    st.markdown("""<div class="warning-box">⚠️ <b>교육/학습 목적</b>입니다. 실제 투자 결정은 <b>본인 책임</b>입니다.</div>""",unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 종목 선택")
        sel=st.selectbox("빠른 선택",['직접 입력']+list(POPULAR_STOCKS.keys()))
        if sel=='직접 입력':
            inp=st.text_input("회사명 또는 종목코드",value="삼성전자")
        else:
            inp=sel; st.info(f"{inp} ({POPULAR_STOCKS[inp][0]})")
        days=st.slider("분석 기간",30,730,365,30)
        go_btn=st.button("🚀 분석 시작",type="primary",use_container_width=True)
        st.markdown("---")
        if HAS_KRX: st.success("✅ KRX 연결됨")
        else: st.warning("⚠️ KRX 미연결 (Yahoo만 사용)")

    if go_btn or inp:
        try:
            krx_code,yf_sym,display=resolve_ticker(inp)
            if krx_code is None:
                st.error(f"❌ '{inp}' 종목을 찾을 수 없습니다."); return

            with st.spinner(f"'{display}' 분석 중..."):
                df,price_src=get_price(krx_code,yf_sym,days)
                if df.empty: st.error("❌ 주가 데이터 없음"); return
                kf=get_fundamental_krx(krx_code)
                kc=get_cap_krx(krx_code)
                inv_df=get_investor_krx(krx_code)
                yi=yf_info(yf_sym)
                news,news_sc,news_ov=yf_news(yf_sym)
                macro=yf_macro()
                fund=build_fundamental(kf,kc,yi)
                df=technical_analysis(df)
                corr,sec_perf=sector_compare(yf_sym)
                inst_pct=round((yi.get('heldPercentInstitutions',0)or 0)*100,2)
                val,val_sc=evaluate(fund,news_sc,inst_pct)
                pred=ml_predict(df)
                earnings={}
                ts=yi.get('earningsTimestamp',0)
                if ts: earnings['실적발표일']=datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                eq=yi.get('earningsQuarterlyGrowth',0)or 0; earnings['분기성장률']=f"{eq*100:+.1f}%"
                rg=yi.get('revenueGrowth',0)or 0; earnings['매출성장률']=f"{rg*100:+.1f}%"
                peg=yi.get('trailingPegRatio',0)or 0
                if peg>0: earnings['PEG']=f"{peg:.2f}"; earnings['PEG해석']='✅ 저평가' if peg<1 else ('🟡 적정' if peg<2 else '🔴 고평가')

            eng=fund.get('회사명','')
            st.success(f"✅ **{display}** ({krx_code}) | 데이터: {price_src}" + (f" | {eng}" if eng!='N/A' else ""))
            if fund.get('섹터')!='N/A': st.caption(f"🏢 {fund.get('섹터')} > {fund.get('업종')}")

            c1,c2,c3,c4=st.columns(4)
            cp=df['Close'].iloc[-1]; pp=df['Close'].iloc[-2] if len(df)>1 else cp
            chg=cp-pp; chg_p=(chg/pp)*100 if pp else 0
            with c1: st.metric("현재가",f"{cp:,.0f}원",f"{chg:+,.0f}원 ({chg_p:+.2f}%)")
            with c2: st.metric("52주 최고",f"{df['High'].max():,.0f}원")
            with c3: st.metric("52주 최저",f"{df['Low'].min():,.0f}원")
            with c4:
                cap=fund.get('시가총액',0)
                if cap>=1e12: st.metric("시가총액",f"{cap/1e12:,.1f}조원")
                elif cap>0: st.metric("시가총액",f"{cap/1e8:,.0f}억원")
                else: st.metric("시가총액","N/A")

            st.markdown("---")
            st.subheader("🎯 종합 투자 분석 점수")
            rsi=df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            macd_v=df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
            macd_s=df['MACD_signal'].iloc[-1] if not pd.isna(df['MACD_signal'].iloc[-1]) else 0
            ts2=50
            if rsi<30: ts2+=15
            elif rsi>70: ts2-=15
            if macd_v>macd_s: ts2+=10
            else: ts2-=5
            try:
                if df['Close'].iloc[-1]>df['MA20'].iloc[-1]>df['MA60'].iloc[-1]: ts2+=10
            except: pass
            ts2=max(0,min(100,ts2))
            ms=50
            if pred:
                pr=pred['ret']
                if pr>2: ms+=20
                elif pr>0: ms+=10
                elif pr<-2: ms-=20
                else: ms-=10
            total=val_sc*0.4+ts2*0.3+ms*0.3
            c1,c2,c3,c4=st.columns(4)
            with c1: st.metric("💼 기업가치",f"{val_sc:.0f}/100")
            with c2: st.metric("📈 기술적",f"{ts2:.0f}/100")
            with c3: st.metric("🤖 ML",f"{ms:.0f}/100")
            with c4: st.metric("🎯 종합",f"{total:.1f}/100")
            if total>=70: st.success(f"### 🟢 매수 고려 | 뉴스: {news_ov}")
            elif total>=55: st.info(f"### 🟡 관심 종목 | 뉴스: {news_ov}")
            elif total>=40: st.warning(f"### ⚪ 관망 | 뉴스: {news_ov}")
            else: st.error(f"### 🔴 매수 비추천 | 뉴스: {news_ov}")

            st.markdown("---")
            t1,t2,t3,t4,t5,t6,t7=st.tabs(["📊 차트","💼 기업가치","📰 뉴스","🌍 거시경제","🏢 외국인/기관","🔗 업종비교","🤖 ML/실적"])

            with t1: st.plotly_chart(make_chart(df,display,krx_code),use_container_width=True)

            with t2:
                c1,c2=st.columns(2)
                with c1:
                    st.subheader("📊 재무 지표")
                    st.caption(f"PER: {fund.get('src_PER')} | PBR: {fund.get('src_PBR')}")
                    for k in ['PER','PBR','EPS','BPS','DIV','ROE','부채비율','매출성장률','이익률']:
                        v=fund.get(k,0)
                        if k in ['EPS','BPS']: st.write(f"**{k}**: {v:,.0f}원" if v else f"**{k}**: N/A")
                        elif k in ['DIV','ROE','부채비율','매출성장률','이익률']: st.write(f"**{k}**: {v}%" if v else f"**{k}**: N/A")
                        else: st.write(f"**{k}**: {v}" if v else f"**{k}**: N/A")
                with c2:
                    st.subheader("🔍 평가"); 
                    for k,v in val.items(): st.write(f"**{k}**: {v}")

            with t3:
                st.subheader(f"📰 뉴스 감성 | {news_ov} (점수: {news_sc:+d})")
                if news:
                    for item in news:
                        with st.expander(f"{item['sentiment']} {item['title'][:80]}"):
                            st.write(f"**출처**: {item['publisher']} | **날짜**: {item['date']}")
                            st.write(f"**감성**: {item['score']:+d}")
                            if item['link']!='#': st.markdown(f"[기사 원문]({item['link']})")
                else: st.info("뉴스 없음")

            with t4:
                st.subheader("🌍 거시경제")
                if macro:
                    cols=st.columns(3)
                    for i,(n,d) in enumerate(macro.items()):
                        with cols[i%3]:
                            vs=f"${d['current']:,.2f}" if n in ['WTI유가','금'] else f"{d['current']:,.2f}"
                            st.metric(n,vs,f"{d['daily_change']:+.2f}%")
                    st.markdown("---")
                    sel_m=st.selectbox("추이",list(macro.keys()))
                    if sel_m in macro:
                        fig=go.Figure(); fig.add_trace(go.Scatter(x=macro[sel_m]['history'].index,y=macro[sel_m]['history'].values,mode='lines',line=dict(color='#4ECDC4',width=2)))
                        fig.update_layout(height=300,template='plotly_white',title=f"{sel_m} 3개월")
                        st.plotly_chart(fig,use_container_width=True)
                else: st.warning("거시경제 데이터 로딩 실패")

            with t5:
                st.subheader("🏢 외국인/기관")
                if not inv_df.empty:
                    cs=[c for c in inv_df.columns if any(k in c for k in ['외국인','기관','개인'])]
                    if cs:
                        fig=go.Figure()
                        cm={'외국인':'#FF6B6B','기관':'#4ECDC4','개인':'#FFD93D'}
                        for col in cs[:4]:
                            clr='#999'
                            for k,c in cm.items():
                                if k in col: clr=c; break
                            fig.add_trace(go.Bar(x=inv_df.index,y=inv_df[col],name=col,marker_color=clr,opacity=0.8))
                        fig.update_layout(height=400,template='plotly_white',barmode='group')
                        st.plotly_chart(fig,use_container_width=True)
                else:
                    if not HAS_KRX: st.info("KRX 미연결 → Yahoo 데이터만 표시")
                    else: st.warning("KRX 매매 데이터 없음")
                st.markdown("---")
                st.write(f"**기관 보유율**: {inst_pct}%")
                na=yi.get('numberOfAnalystOpinions',0)or 0
                rec=yi.get('recommendationKey','N/A')
                rm={'buy':'🟢 매수','strong_buy':'🟢 적극매수','hold':'🟡 보유','sell':'🔴 매도'}
                st.write(f"**애널리스트 {na}명**: {rm.get(rec,rec)}")
                tgt=yi.get('targetMeanPrice',0)or 0
                if tgt>0: st.write(f"**목표가**: {tgt:,.0f}원 ({((tgt-cp)/cp)*100:+.1f}%)")

            with t6:
                st.subheader("🔗 업종 비교")
                if corr is not None and sec_perf is not None:
                    fig=go.Figure(go.Bar(x=list(sec_perf.keys()),y=list(sec_perf.values()),
                        marker_color=['green' if v>0 else 'red' for v in sec_perf.values()],
                        text=[f"{v:+.1f}%" for v in sec_perf.values()],textposition='outside'))
                    fig.update_layout(height=350,template='plotly_white',title="6개월 수익률")
                    st.plotly_chart(fig,use_container_width=True)
                    st.dataframe(corr.round(3),use_container_width=True)
                else: st.info("업종 비교 데이터 없음")

            with t7:
                c1,c2=st.columns(2)
                with c1:
                    st.subheader("🤖 ML 예측")
                    if pred:
                        st.metric("ML 기준일",f"{pred['base_price']:,.0f}원")
                        st.metric("5일 후",f"{pred['pred_price']:,.0f}원",f"{pred['ret']:+.2f}%")
                        st.metric("R²",f"{pred['r2']:.4f}")
                        imp=pred.get('importance',{})
                        if imp:
                            idf=pd.DataFrame.from_dict(imp,orient='index',columns=['중요도']).sort_values('중요도')
                            fig=go.Figure(go.Bar(x=idf['중요도'],y=idf.index,orientation='h',marker_color='#4ECDC4'))
                            fig.update_layout(height=400,template='plotly_white')
                            st.plotly_chart(fig,use_container_width=True)
                    else: st.warning("데이터 부족")
                with c2:
                    st.subheader("📅 실적")
                    for k,v in earnings.items(): st.write(f"**{k}**: {v}")
                    tgt=yi.get('targetMeanPrice',0)or 0
                    if tgt>0:
                        st.markdown("---"); st.subheader("🎯 목표가")
                        st.metric("평균",f"{tgt:,.0f}원")
                        st.metric("최고",f"{yi.get('targetHighPrice',0):,.0f}원")
                        st.metric("최저",f"{yi.get('targetLowPrice',0):,.0f}원")
        except Exception as e:
            st.error(f"❌ 오류: {str(e)}")
            with st.expander("상세"):
                import traceback; st.code(traceback.format_exc())

if __name__=="__main__": main()