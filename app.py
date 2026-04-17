"""
한국주식 분석 시스템 Pro (v8 - 뉴스/거시경제 수정)
"""

import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os
# Streamlit Cloud Secrets 지원
if hasattr(st, 'secrets'):
    if 'KRX_ID' in st.secrets:
        os.environ['KRX_ID'] = st.secrets['KRX_ID']
    if 'KRX_PW' in st.secrets:
        os.environ['KRX_PW'] = st.secrets['KRX_PW']
        
import yfinance as yf
from pykrx import stock
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
    '반도체': ['005930', '000660'],
    'IT/플랫폼': ['035420', '035720'],
    '자동차': ['005380', '000270'],
    '바이오': ['068270', '207940', '196170'],
    '에너지/소재': ['373220', '051910', '006400', '005490'],
    '금융': ['105560'],
    '항공': ['089590'],
}

MACRO_SYMBOLS = {
    'USD/KRW': 'KRW=X',
    'S&P500': '^GSPC',
    'KOSPI': '^KS11',
    'WTI유가': 'CL=F',
    '금': 'GC=F',
    '미국10년국채': '^TNX',
    'VIX': '^VIX',
}


def get_recent_business_day():
    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        try:
            test = stock.get_market_ohlcv(date, date, "005930")
            if not test.empty:
                return date
        except Exception:
            continue
    return (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')


def resolve_ticker(user_input):
    if user_input in POPULAR_STOCKS:
        krx, yf_sym = POPULAR_STOCKS[user_input]
        return krx, yf_sym, user_input
    if user_input.isdigit() and len(user_input) == 6:
        try:
            name = stock.get_market_ticker_name(user_input)
            if name:
                yf_sym = f"{user_input}.KS"
                try:
                    t = yf.Ticker(yf_sym)
                    if t.history(period="5d").empty:
                        yf_sym = f"{user_input}.KQ"
                except Exception:
                    yf_sym = f"{user_input}.KQ"
                return user_input, yf_sym, name
        except Exception:
            pass
    try:
        recent = get_recent_business_day()
        tickers = stock.get_market_ticker_list(recent, market="ALL")
        for t in tickers:
            try:
                name = stock.get_market_ticker_name(t)
                if name == user_input or user_input in name:
                    return t, f"{t}.KS", name
            except Exception:
                continue
    except Exception:
        pass
    return None, None, user_input


# === KRX 데이터 ===
@st.cache_data(ttl=3600, show_spinner=False)
def krx_price(ticker, days):
    end = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    try:
        df = stock.get_market_ohlcv(start, end, ticker)
        if not df.empty:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def krx_fundamental(ticker):
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        df = stock.get_market_fundamental(start, end, ticker)
        if not df.empty:
            df_valid = df[df.sum(axis=1) > 0]
            if not df_valid.empty:
                return df_valid.iloc[-1].to_dict()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def krx_market_cap(ticker):
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        df = stock.get_market_cap(start, end, ticker)
        if not df.empty:
            df_valid = df[df['시가총액'] > 0]
            if not df_valid.empty:
                return df_valid.iloc[-1].to_dict()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def krx_investor(ticker, days=60):
    end = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    try:
        df = stock.get_market_trading_value_by_date(start, end, ticker)
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        df = stock.get_market_trading_volume_by_date(start, end, ticker)
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


# === yfinance 데이터 ===
@st.cache_data(ttl=3600, show_spinner=False)
def yf_info(symbol):
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def yf_news(symbol):
    positive = ['surge','soar','jump','rally','gain','rise','profit','growth',
                'beat','exceed','record','high','upgrade','buy','strong','boom',
                'breakthrough','innovation','success','expand','bullish','positive',
                'dividend','partnership','deal','approval','launch','recovery']
    negative = ['fall','drop','decline','loss','crash','plunge','tumble','miss',
                'fail','weak','downgrade','sell','risk','concern','warning','cut',
                'layoff','lawsuit','investigation','scandal','bearish','negative',
                'recession','debt','default','bankruptcy','fraud','penalty']
    try:
        ticker = yf.Ticker(symbol)
        news_raw = ticker.news
        if not news_raw:
            return [], 0, "중립 🟡"

        # yfinance 버전별 구조 대응
        if isinstance(news_raw, dict):
            news_list = news_raw.get('news', news_raw.get('items', []))
        elif isinstance(news_raw, list):
            news_list = news_raw
        else:
            return [], 0, "중립 🟡"

        results = []
        total = 0
        for item in news_list[:10]:
            # 제목: 여러 경로 시도
            title = ''
            for key in ['title', 'headline', 'name']:
                title = item.get(key, '') or ''
                if title:
                    break
            if not title:
                content = item.get('content', {})
                if isinstance(content, dict):
                    title = content.get('title', '') or ''
            if not title:
                title = '(제목 없음)'

            # 출처: 여러 경로 시도
            publisher = ''
            for key in ['publisher', 'source']:
                publisher = item.get(key, '') or ''
                if publisher:
                    break
            if not publisher:
                prov = item.get('provider', {})
                if isinstance(prov, dict):
                    publisher = prov.get('displayName', '') or prov.get('name', '') or ''
            if not publisher:
                content = item.get('content', {})
                if isinstance(content, dict):
                    prov2 = content.get('provider', {})
                    if isinstance(prov2, dict):
                        publisher = prov2.get('displayName', '') or prov2.get('name', '') or ''
            if not publisher:
                publisher = '알 수 없음'

            # 링크: 여러 경로 시도
            link = ''
            for key in ['link', 'url']:
                link = item.get(key, '') or ''
                if link:
                    break
            if not link:
                canon = item.get('canonicalUrl', {})
                if isinstance(canon, dict):
                    link = canon.get('url', '') or ''
            if not link:
                content = item.get('content', {})
                if isinstance(content, dict):
                    canon2 = content.get('canonicalUrl', {})
                    if isinstance(canon2, dict):
                        link = canon2.get('url', '') or ''
            if not link:
                link = '#'

            # 날짜: 여러 경로 시도
            date_str = 'N/A'
            pub_time = item.get('providerPublishTime', 0) or item.get('publishedAt', 0)
            if not pub_time:
                content = item.get('content', {})
                if isinstance(content, dict):
                    pub_time = content.get('pubDate', '') or ''
            if isinstance(pub_time, (int, float)) and pub_time > 1000000:
                date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
            elif isinstance(pub_time, str) and len(pub_time) > 5:
                date_str = pub_time[:16]

            # 감성 분석
            t_lower = title.lower()
            pos = sum(1 for w in positive if w in t_lower)
            neg = sum(1 for w in negative if w in t_lower)
            sc = pos - neg
            total += sc
            sent = "긍정 🟢" if sc > 0 else ("부정 🔴" if sc < 0 else "중립 🟡")

            results.append({'title': title, 'publisher': publisher,
                           'date': date_str, 'link': link,
                           'sentiment': sent, 'score': sc})

        overall = "긍정적 🟢" if total > 2 else ("부정적 🔴" if total < -2 else "중립 🟡")
        return results, total, overall
    except Exception:
        return [], 0, "분석 불가"


@st.cache_data(ttl=3600, show_spinner=False)
def yf_macro(days=90):
    """거시경제 데이터 - 개별 다운로드"""
    macro = {}
    end = datetime.now()
    start = end - timedelta(days=days)

    for name, sym in MACRO_SYMBOLS.items():
        try:
            t = yf.Ticker(sym)
            df = t.history(start=start, end=end)
            if not df.empty and 'Close' in df.columns:
                cur = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2]) if len(df) > 1 else cur
                chg = ((cur - prev) / prev) * 100 if prev else 0
                m_ago = float(df['Close'].iloc[0])
                m_chg = ((cur - m_ago) / m_ago) * 100 if m_ago else 0
                macro[name] = {
                    'current': cur,
                    'daily_change': round(chg, 2),
                    'monthly_change': round(m_chg, 2),
                    'history': df['Close']
                }
        except Exception:
            continue
    return macro


# === 분석 함수들 ===
def technical_analysis(df):
    for w in [5, 20, 60, 120]:
        df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_short'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + (std * 2)
    df['BB_lower'] = df['BB_mid'] - (std * 2)
    return df


def build_fundamental(kf, kc, yi):
    per = kf.get('PER', 0) or yi.get('trailingPE', 0) or yi.get('forwardPE', 0) or 0
    pbr = kf.get('PBR', 0) or yi.get('priceToBook', 0) or 0
    eps = kf.get('EPS', 0) or yi.get('trailingEps', 0) or yi.get('epsCurrentYear', 0) or 0
    bps = kf.get('BPS', 0) or yi.get('bookValue', 0) or 0
    if eps == 0:
        ni = yi.get('netIncomeToCommon', 0) or 0
        sh = yi.get('sharesOutstanding', 0) or 0
        if ni > 0 and sh > 0:
            eps = ni / sh
    if pbr == 0 and bps > 0:
        cp = yi.get('currentPrice', 0) or yi.get('previousClose', 0) or 0
        if cp > 0:
            pbr = cp / bps

    div_krx = kf.get('DIV', 0) or 0
    if div_krx > 0:
        div = div_krx
    else:
        dr = yi.get('dividendYield', 0) or 0
        div = dr if dr > 1 else dr * 100

    roe_r = yi.get('returnOnEquity', 0) or 0
    roe = roe_r if roe_r > 1 else roe_r * 100
    if roe == 0 and eps > 0 and bps > 0:
        roe = (eps / bps) * 100

    cap = kc.get('시가총액', 0) or yi.get('marketCap', 0) or 0

    return {
        'PER': round(per, 2) if per else 0,
        'PBR': round(pbr, 2) if pbr else 0,
        'EPS': round(eps, 0) if eps else 0,
        'BPS': round(bps, 0) if bps else 0,
        'DIV': round(div, 2),
        'ROE': round(roe, 2),
        '시가총액': cap,
        '부채비율': round((yi.get('debtToEquity', 0) or 0), 2),
        '매출성장률': round((yi.get('revenueGrowth', 0) or 0) * 100, 2),
        '이익률': round((yi.get('profitMargins', 0) or 0) * 100, 2),
        'src_PER': 'KRX' if kf.get('PER', 0) else 'Yahoo',
        'src_PBR': 'KRX' if kf.get('PBR', 0) else 'Yahoo',
        '업종': yi.get('industry', 'N/A'),
        '섹터': yi.get('sector', 'N/A'),
        '회사명': yi.get('longName', yi.get('shortName', 'N/A')),
    }


def evaluate(fund, news_sc=0, inst_pct=0):
    score = 50; a = {}
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

    if roe>=15: score+=15; a['ROE']='✅ 매우 우수'
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
    if eps>0 and bps>0:
        fair=(22.5*eps*bps)**0.5; a['적정주가']=f'{fair:,.0f}원'
    return a, max(0,min(100,score))


def ml_predict(df):
    d=df.copy()
    d['Returns']=d['Close'].pct_change()
    d['Volatility']=d['Returns'].rolling(20).std()
    d['Price_Range']=(d['High']-d['Low'])/d['Close']
    d['Vol_Change']=d['Volume'].pct_change()
    d['Momentum']=d['Close']/d['Close'].shift(20)-1
    d['Target']=d['Close'].shift(-5)/d['Close']-1
    # 무한대, NaN 제거
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna()
    feats=['MA5','MA20','MA60','RSI','MACD','MACD_signal','BB_upper','BB_lower',
           'Volume','Volatility','Price_Range','Vol_Change','Momentum']
    feats=[f for f in feats if f in d.columns]
    if len(d)<50: return None
    X=d[feats].values; y=d['Target'].values
    # 혹시 남은 무한대 한번 더 체크
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X=X[mask]; y=y[mask]
    if len(X)<50: return None
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    sp=int(len(Xs)*0.8)
    m=RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1)
    m.fit(Xs[:sp],y[:sp]); r2=m.score(Xs[sp:],y[sp:])
    pred=m.predict(sc.transform(d[feats].iloc[-1:].values))[0]
    cp=d['Close'].iloc[-1]
    imp=dict(sorted(zip(feats,m.feature_importances_),key=lambda x:x[1],reverse=True))
    return {'r2':r2,'ret':pred*100,'pred_price':cp*(1+pred),'base_price':cp,'importance':imp}

def make_chart(df, name, ticker):
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=0.05,
                      row_heights=[0.5,0.15,0.175,0.175],
                      subplot_titles=(f'{name} ({ticker})','거래량','RSI','MACD'))
    fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
                                  low=df['Low'],close=df['Close'],name='주가',
                                  increasing_line_color='#ff4444',decreasing_line_color='#4444ff'),row=1,col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['MA20'],name='MA20',
                                  line=dict(color='orange',width=1)),row=1,col=1)
    if 'MA60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['MA60'],name='MA60',
                                  line=dict(color='purple',width=1)),row=1,col=1)
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['BB_upper'],name='BB상단',
                                  line=dict(color='gray',width=1,dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['BB_lower'],name='BB하단',
                                  line=dict(color='gray',width=1,dash='dot'),
                                  fill='tonexty',fillcolor='rgba(128,128,128,0.1)'),row=1,col=1)
    colors=['red' if df['Close'].iloc[i]>=df['Open'].iloc[i] else 'blue' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name='거래량',marker_color=colors,opacity=0.7),row=2,col=1)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['RSI'],name='RSI',
                                  line=dict(color='purple',width=1.5)),row=3,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="red",row=3,col=1)
        fig.add_hline(y=30,line_dash="dash",line_color="green",row=3,col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD'],name='MACD',
                                  line=dict(color='blue',width=1.5)),row=4,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df['MACD_signal'],name='Signal',
                                  line=dict(color='red',width=1.5)),row=4,col=1)
        mc=['green' if x>=0 else 'red' for x in df['MACD_hist']]
        fig.add_trace(go.Bar(x=df.index,y=df['MACD_hist'],name='Hist',
                              marker_color=mc,opacity=0.5),row=4,col=1)
    fig.update_layout(height=900,showlegend=True,xaxis_rangeslider_visible=False,
                      hovermode='x unified',template='plotly_white')
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def sector_compare(ticker, days=180):
    target=None
    for sec,tks in SECTOR_GROUPS.items():
        if ticker in tks: target=sec; break
    if not target or len(SECTOR_GROUPS[target])<2: return None,None
    end=datetime.now().strftime('%Y%m%d')
    start=(datetime.now()-timedelta(days=days)).strftime('%Y%m%d')
    prices={}
    for t in SECTOR_GROUPS[target]:
        try:
            df=stock.get_market_ohlcv(start,end,t)
            if not df.empty:
                name=t
                for k,v in POPULAR_STOCKS.items():
                    if v[0]==t: name=k; break
                prices[name]=df['종가'].pct_change().dropna()
        except Exception:
            continue
    if len(prices)<2: return None,None
    df_r=pd.DataFrame(prices)
    return df_r.corr(), {n:round(((1+r).prod()-1)*100,2) for n,r in prices.items()}


# === 메인 ===
def main():
    st.markdown('<h1 class="main-header">📈 한국주식 분석 Pro</h1>',unsafe_allow_html=True)
    st.markdown("### KRX + Yahoo Finance 하이브리드 분석")
    st.markdown("""<div class="warning-box">
    ⚠️ <b>교육/학습 목적</b>입니다. 실제 투자 결정은 <b>본인 책임</b>입니다.
    </div>""",unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 종목 선택")
        sel=st.selectbox("빠른 선택",['직접 입력']+list(POPULAR_STOCKS.keys()))
        if sel=='직접 입력':
            inp=st.text_input("회사명 또는 종목코드",value="삼성전자")
        else:
            inp=sel; st.info(f"{inp} ({POPULAR_STOCKS[inp][0]})")
        days=st.slider("분석 기간 (일)",30,730,365,30)
        go_btn=st.button("🚀 분석 시작",type="primary",use_container_width=True)
        st.markdown("---")
        st.markdown("""### 📊 분석 7가지
1. 📈 주가 차트
2. 💼 기업가치 (KRX)
3. 📰 뉴스 감성
4. 🌍 거시경제
5. 🏢 외국인/기관 (KRX)
6. 🔗 업종 비교 (KRX)
7. 🤖 ML 예측 + 실적""")

    if go_btn or inp:
        try:
            krx_code,yf_sym,display=resolve_ticker(inp)
            if krx_code is None:
                st.error(f"❌ '{inp}' 종목을 찾을 수 없습니다."); return

            with st.spinner(f"'{display}' 종합 분석 중... (15~30초)"):
                df=krx_price(krx_code,days)
                if df.empty:
                    st.error("❌ 주가 데이터를 가져올 수 없습니다."); return
                kf=krx_fundamental(krx_code)
                kc=krx_market_cap(krx_code)
                inv_df=krx_investor(krx_code,60)
                yi=yf_info(yf_sym)
                news,news_sc,news_ov=yf_news(yf_sym)
                macro=yf_macro(90)
                fund=build_fundamental(kf,kc,yi)
                df=technical_analysis(df)
                corr,sec_perf=sector_compare(krx_code,180)
                inst_pct=round((yi.get('heldPercentInstitutions',0)or 0)*100,2)
                val,val_sc=evaluate(fund,news_sc,inst_pct)
                pred=ml_predict(df)
                earnings={}
                ts=yi.get('earningsTimestamp',0)
                if ts: earnings['실적발표일']=datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                eq=yi.get('earningsQuarterlyGrowth',0)or 0
                earnings['분기성장률']=f"{eq*100:+.1f}%"
                rg=yi.get('revenueGrowth',0)or 0
                earnings['매출성장률']=f"{rg*100:+.1f}%"
                peg=yi.get('trailingPegRatio',0)or 0
                if peg>0:
                    earnings['PEG']=f"{peg:.2f}"
                    earnings['PEG해석']='✅ 저평가' if peg<1 else ('🟡 적정' if peg<2 else '🔴 고평가')

            eng=fund.get('회사명','')
            st.success(f"✅ **{display}** ({krx_code})" + (f" | {eng}" if eng!='N/A' else ""))
            if fund.get('섹터')!='N/A':
                st.caption(f"🏢 {fund.get('섹터')} > {fund.get('업종')}")

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

            with t1:
                st.plotly_chart(make_chart(df,display,krx_code),use_container_width=True)

            with t2:
                c1,c2=st.columns(2)
                with c1:
                    st.subheader("📊 재무 지표")
                    st.caption(f"PER: {fund.get('src_PER')} | PBR: {fund.get('src_PBR')}")
                    for k in ['PER','PBR','EPS','BPS','DIV','ROE','부채비율','매출성장률','이익률']:
                        v=fund.get(k,0)
                        if k in ['EPS','BPS']:
                            st.write(f"**{k}**: {v:,.0f}원" if v else f"**{k}**: N/A")
                        elif k in ['DIV','ROE','부채비율','매출성장률','이익률']:
                            st.write(f"**{k}**: {v}%" if v else f"**{k}**: N/A")
                        else:
                            st.write(f"**{k}**: {v}" if v else f"**{k}**: N/A")
                with c2:
                    st.subheader("🔍 평가 결과")
                    for k,v in val.items(): st.write(f"**{k}**: {v}")

            with t3:
                st.subheader(f"📰 뉴스 감성 | 종합: {news_ov} (점수: {news_sc:+d})")
                if news:
                    for item in news:
                        with st.expander(f"{item['sentiment']} {item['title'][:80]}"):
                            st.write(f"**출처**: {item['publisher']} | **날짜**: {item['date']}")
                            st.write(f"**감성 점수**: {item['score']:+d}")
                            if item['link'] != '#':
                                st.markdown(f"[기사 원문 보기]({item['link']})")
                else:
                    st.info("뉴스 데이터가 없습니다.")

            with t4:
                st.subheader("🌍 거시경제 지표")
                if macro:
                    cols=st.columns(3)
                    for i,(name,data) in enumerate(macro.items()):
                        with cols[i%3]:
                            val_str = f"{data['current']:,.2f}"
                            if name=='WTI유가' or name=='금':
                                val_str = f"${data['current']:,.2f}"
                            st.metric(name, val_str, f"{data['daily_change']:+.2f}% (일간)")
                    st.markdown("---")
                    sel_m=st.selectbox("추이 보기",list(macro.keys()))
                    if sel_m in macro:
                        fig=go.Figure()
                        fig.add_trace(go.Scatter(x=macro[sel_m]['history'].index,
                                                  y=macro[sel_m]['history'].values,
                                                  mode='lines',line=dict(color='#4ECDC4',width=2)))
                        fig.update_layout(height=300,template='plotly_white',title=f"{sel_m} 3개월")
                        st.plotly_chart(fig,use_container_width=True)
                else:
                    st.warning("거시경제 데이터를 불러올 수 없습니다. 인터넷 연결을 확인하세요.")

            with t5:
                st.subheader("🏢 외국인/기관 매매 동향 (KRX)")
                if not inv_df.empty:
                    cols_show=[c for c in inv_df.columns if any(k in c for k in ['외국인','기관','개인'])]
                    if cols_show:
                        fig=go.Figure()
                        cmap={'외국인':'#FF6B6B','기관':'#4ECDC4','개인':'#FFD93D'}
                        for col in cols_show[:4]:
                            clr='#999'
                            for k,c in cmap.items():
                                if k in col: clr=c; break
                            fig.add_trace(go.Bar(x=inv_df.index,y=inv_df[col],name=col,marker_color=clr,opacity=0.8))
                        fig.update_layout(height=400,template='plotly_white',barmode='group',title="투자자별 순매매")
                        st.plotly_chart(fig,use_container_width=True)
                        st.subheader("📊 누적 순매매")
                        summary=inv_df[cols_show].sum()
                        cs=st.columns(min(len(cols_show),4))
                        for i,col in enumerate(cols_show[:4]):
                            with cs[i]:
                                v=summary[col]
                                if abs(v)>=1e8: st.metric(col,f"{v/1e8:,.0f}억원")
                                else: st.metric(col,f"{v/1e4:,.0f}만원")
                    else:
                        st.dataframe(inv_df.tail(20),use_container_width=True)
                else:
                    st.warning("외국인/기관 매매 데이터를 가져올 수 없습니다.")
                st.markdown("---")
                st.subheader("📋 기관 보유 (Yahoo)")
                c1,c2=st.columns(2)
                with c1:
                    st.write(f"**기관 보유율**: {inst_pct}%")
                    st.write(f"**내부자 보유율**: {round((yi.get('heldPercentInsiders',0)or 0)*100,2)}%")
                with c2:
                    n_a=yi.get('numberOfAnalystOpinions',0)or 0
                    rec=yi.get('recommendationKey','N/A')
                    rm={'buy':'🟢 매수','strong_buy':'🟢 적극매수','hold':'🟡 보유','sell':'🔴 매도'}
                    st.write(f"**애널리스트 {n_a}명**: {rm.get(rec,rec)}")
                    tgt=yi.get('targetMeanPrice',0)or 0
                    if tgt>0:
                        up=((tgt-cp)/cp)*100
                        st.write(f"**목표가**: {tgt:,.0f}원 ({up:+.1f}%)")

            with t6:
                st.subheader("🔗 업종 내 비교 (KRX)")
                if corr is not None and sec_perf is not None:
                    fig=go.Figure(go.Bar(x=list(sec_perf.keys()),y=list(sec_perf.values()),
                                          marker_color=['green' if v>0 else 'red' for v in sec_perf.values()],
                                          text=[f"{v:+.1f}%" for v in sec_perf.values()],textposition='outside'))
                    fig.update_layout(height=350,template='plotly_white',title="6개월 수익률")
                    st.plotly_chart(fig,use_container_width=True)
                    st.write("**상관관계**")
                    st.dataframe(corr.round(3),use_container_width=True)
                else:
                    st.info("업종 비교 데이터가 없습니다.")

            with t7:
                c1,c2=st.columns(2)
                with c1:
                    st.subheader("🤖 ML 예측")
                    if pred:
                        st.metric("ML 기준일",f"{pred['base_price']:,.0f}원")
                        st.metric("5일 후 예측",f"{pred['pred_price']:,.0f}원",f"{pred['ret']:+.2f}%")
                        st.metric("R² 점수",f"{pred['r2']:.4f}")
                        st.markdown("---")
                        imp=pred.get('importance',{})
                        if imp:
                            idf=pd.DataFrame.from_dict(imp,orient='index',columns=['중요도']).sort_values('중요도')
                            fig=go.Figure(go.Bar(x=idf['중요도'],y=idf.index,orientation='h',marker_color='#4ECDC4'))
                            fig.update_layout(height=400,template='plotly_white',title="피처 중요도")
                            st.plotly_chart(fig,use_container_width=True)
                    else:
                        st.warning("데이터 부족")
                with c2:
                    st.subheader("📅 실적 정보")
                    for k,v in earnings.items(): st.write(f"**{k}**: {v}")
                    tgt=yi.get('targetMeanPrice',0)or 0
                    if tgt>0:
                        st.markdown("---")
                        st.subheader("🎯 목표가")
                        st.metric("평균",f"{tgt:,.0f}원")
                        st.metric("최고",f"{yi.get('targetHighPrice',0):,.0f}원")
                        st.metric("최저",f"{yi.get('targetLowPrice',0):,.0f}원")

        except Exception as e:
            st.error(f"❌ 오류: {str(e)}")
            with st.expander("상세"):
                import traceback
                st.code(traceback.format_exc())

if __name__=="__main__":
    main()