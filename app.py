import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# --- 1. THE BRAIN: MULTI-SOURCE DATA ---

def get_nasdaq_data(symbol):
    """Fetches key data from NASDAQ's internal API"""
    headers = {
        'accept': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        url = f'https://api.nasdaq.com/api/quote/{symbol}/summary?assetclass=stocks'
        resp = requests.get(url, headers=headers, timeout=5).json()
        return resp['data']['summaryData']
    except: return {}

def get_finviz_news(symbol):
    """Scrapes news headlines from FinViz"""
    headers = {'user-agent': 'Mozilla/5.0'}
    try:
        url = f'https://finviz.com/quote.ashx?t={symbol}'
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.content, 'html.parser')
        news_table = soup.find(id='news-table')
        rows = news_table.findAll('tr')
        return [r.a.get_text() for r in rows[:3]] # Top 3 headlines
    except: return ["News currently unavailable"]

def get_medallion_metrics(ticker):
    """The core algorithmic engine"""
    try:
        # 1. Fetch Price Data (Yahoo)
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d", interval="1h")
        if df.empty: return None

        # 2. VWAP & TWAP
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        twap = df['Close'].mean()

        # 3. TOC Squeeze (Theory of Constraints)
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        atr = (df['High'] - df['Low']).rolling(20).mean()
        squeeze = "🔒 SQUEEZE" if (sma - (2*std) > sma - (1.5*atr)).iloc[-1] else "🌊 EXPANSION"

        # 4. ML Forecast (Linear Regression)
        y = df['Close'].values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 1]])[0]

        # 5. Options POP (Probability of Profit)
        # Using Normal Distribution: POP = N(d2)
        sigma = df['Close'].pct_change().std() * np.sqrt(252 * 7)
        dist = np.log(forecast / df['Close'].iloc[-1])
        pop = si.norm.cdf(dist / (sigma * np.sqrt(1/12))) * 100

        # 6. Fundamental Spread (NASDAQ)
        nasdaq = get_nasdaq_data(ticker)
        news = get_finviz_news(ticker)

        return {
            "Ticker": ticker,
            "Price": df['Close'].iloc[-1],
            "VWAP": df['vwap'].iloc[-1],
            "TWAP": twap,
            "TOC State": squeeze,
            "AI Forecast": forecast,
            "Success %": pop,
            "News": news,
            "Market Cap": nasdaq.get('MarketCap', {}).get('value', 'N/A')
        }
    except Exception as e:
        return None

# --- 2. THE UI: FIDELITY ELITE THEME ---

st.set_page_config(page_title="Fidelity Elite Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1217; color: #e1e4e8; }
    .stMetric { background-color: #161c23; border: 1px solid #30363d; padding: 15px; border-radius: 4px; }
    .stDataFrame { border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("💹 Fidelity Elite Terminal | Multi-Source AI")

# AUTO-REFRESH FRAGMENT
@st.fragment(run_every=65)
def render_terminal(ticker_list):
    st.write(f"🕒 **Last Terminal Sync:** {datetime.now().strftime('%H:%M:%S')}")
    
    results = []
    # Using Threads to speed up multi-source calls
    for t in ticker_list:
        res = get_medallion_metrics(t)
        if res: results.append(res)
        time.sleep(0.5) # Anti-ban throttle
    
    if results:
        df_final = pd.DataFrame(results)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Assets Analyzed", len(df_final))
        c2.metric("Squeeze Alerts", len(df_final[df_final['TOC State'] == "🔒 SQUEEZE"]))
        c3.metric("Avg Success Rate", f"{df_final['Success %'].mean():.1f}%")

        st.subheader("📊 Fidelity Pro Scanner")
        st.dataframe(
            df_final[['Ticker', 'Price', 'VWAP', 'TWAP', 'TOC State', 'AI Forecast', 'Success %', 'Market Cap']]
            .style.background_gradient(subset=['Success %'], cmap='RdYlGn')
            .format({"Price": "${:.2f}", "VWAP": "${:.2f}", "TWAP": "${:.2f}", "AI Forecast": "${:.2f}", "Success %": "{:.1f}%"}),
            use_container_width=True, hide_index=True
        )

        st.subheader("📰 Market Intelligence News (FinViz)")
        for _, row in df_final.iterrows():
            with st.expander(f"Top Headlines: {row['Ticker']}"):
                for h in row['News']:
                    st.write(f"• {h}")

# --- 3. MAIN INTERFACE ---
with st.sidebar:
    st.header("Watchlist Management")
    # Defaulting to a spread of US and Brazil stocks
    tickers = st.text_area("Symbols (Comma Separated)", "AAPL, TSLA, NVDA, PETR4.SA, VALE3.SA, SPY, QQQ", height=150)
    ticker_list = [x.strip() for x in tickers.split(',') if x.strip()]
    st.divider()
    st.info("Terminal updates every 65 seconds.")

render_terminal(ticker_list)
