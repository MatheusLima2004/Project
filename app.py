import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time

# --- 1. SOVEREIGN THEME & MULTI-SOURCE CONFIG ---
st.set_page_config(page_title="Medallion Sovereign Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1217; color: #e1e4e8; }
    .stMetric { background-color: #161c23; border: 1px solid #30363d; padding: 15px; border-radius: 4px; }
    .news-card { background-color: #161c23; padding: 12px; border-radius: 4px; border-left: 4px solid #58a6ff; margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DECENTRALIZED DATA TOOLS ---

def get_marketwatch_sentiment(ticker):
    """Pulls news from MarketWatch to offload Yahoo traffic"""
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker.split('.')[0]}/morenews"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.content, 'html.parser')
        headlines = [h.text.strip() for h in soup.find_all('h3', class_='article__headline')[:2]]
        return headlines if headlines else ["MarketWatch: No recent headlines."]
    except: return ["News Source: MarketWatch Offline."]

def run_medallion_math(df, ticker):
    """Wall Street Algorithms: Z-Score, Kelly, TOC Squeeze, and ML"""
    try:
        if df.empty or len(df) < 60: return None
        
        # Z-Score Mean Reversion
        ma = df['Close'].rolling(60).mean()
        std = df['Close'].rolling(60).std()
        curr = df['Close'].iloc[-1]
        z_score = (curr - ma.iloc[-1]) / std.iloc[-1]
        
        # Kelly Criterion
        rets = df['Close'].pct_change().dropna()
        win_rate = len(rets[rets > 0]) / len(rets)
        win_loss = rets[rets > 0].mean() / abs(rets[rets < 0].mean()) if not rets[rets < 0].empty else 1
        kelly = (win_rate * (win_loss + 1) - 1) / win_loss
        
        # ML Trend Forecast
        y = df['Close'].tail(90).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 30]])[0]
        
        # Options PoP (Probability of Profit)
        sigma = rets.std() * np.sqrt(252)
        days_to_expiry = 30 / 365
        d2 = (np.log(curr / forecast) + (0.045 - 0.5 * sigma**2) * days_to_expiry) / (sigma * np.sqrt(days_to_expiry))
        pop = si.norm.cdf(abs(d2)) * 100
        
        # TOC Squeeze (Bollinger/Keltner Compression)
        atr = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        squeeze = "🔒 SQUEEZE" if (2 * std.iloc[-1] < 1.5 * atr) else "🌊 EXPANSION"

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": z_score, 
            "Kelly %": max(0, kelly * 100), "TOC State": squeeze, 
            "ML Target": forecast, "PoP %": pop
        }
    except: return None

# --- 3. 15-MINUTE DEEP SCAN FRAGMENT ---

@st.fragment(run_every=900)
def sovereign_render(ticker_list):
    st.markdown("<h2 style='color:#58a6ff;'>Sovereign Medallion Command | Decentralized Edition</h2>", unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Cycle Interval", "15:00 MIN", "MULTI-SOURCE SYNC")
    col_stat2.metric("Shield Status", "DECENTRALIZED", "ACTIVE")
    col_stat3.metric("Assets Analyzed", len(ticker_list))

    with st.spinner("Spreading Data Calls Across Global Exchanges..."):
        # Batching historic data (Yahoo) while news is offloaded (MarketWatch)
        df_mega = yf.download(ticker_list, period="2y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        results, news_feed = [], []
        for t in ticker_list:
            res = run_medallion_math(df_mega[t], t)
            if res: 
                results.append(res)
                # News from MarketWatch instead of Yahoo news API
                news_headlines = get_marketwatch_sentiment(t)
                for h in news_headlines:
                    news_feed.append({"symbol": t, "title": h})

    col_main, col_news = st.columns([3, 1])
    
    with col_main:
        if results:
            df_res = pd.DataFrame(results)
            st.dataframe(
                df_res.style.background_gradient(subset=['PoP %'], cmap='RdYlGn')
                .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "Kelly %": "{:.1f}%", "ML Target": "${:.2f}", "PoP %": "{:.1f}%"}),
                use_container_width=True, hide_index=True, height=500
            )
        else:
            st.error("Protocol Failure: Multi-source validation failed. Check Ticker List.")
        
        # Live Chart Feed
        st.components.v1.html(f"""
            <div id="tv-chart" style="height:450px;"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>
            new TradingView.widget({{"width": "100%", "height": 450, "symbol": "{ticker_list[0]}", "interval": "D", "theme": "dark", "style": "1", "locale": "en", "enable_publishing": false, "allow_symbol_change": true, "container_id": "tv-chart"}});
            </script>
        """, height=450)

    with col_news:
        st.subheader("📰 MarketWatch Feed")
        for news in news_feed:
            st.markdown(f"""<div class='news-card'><b>{news['symbol']}</b>: {news['title']}</div>""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Global Asset Universe")
    default_list = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, AMD, PLTR, VALE3.SA, PETR4.SA, ITUB4.SA, SPY, QQQ"
    raw_tickers = st.text_area("Ticker Universe (Comma Separated)", default_list, height=300)
    ticker_list = [x.strip() for x in raw_tickers.split(',') if x.strip()]

sovereign_render(ticker_list)
