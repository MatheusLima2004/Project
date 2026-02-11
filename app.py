import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# --- 1. SYSTEM SETUP ---
st.set_page_config(page_title="Sovereign Options Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e11; color: #e1e4e8; }
    .stMetric { background-color: #15191e; border: 1px solid #2d333b; padding: 15px; border-radius: 4px; }
    .strategy-card { background-color: #1c2128; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .call-signal { color: #00c805; font-weight: bold; }
    .put-signal { color: #ff3b3b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE OPTIONS ML ENGINE ---

def analyze_options_strategy(df, ticker):
    try:
        if df.empty or len(df) < 60: return None
        curr = df['Close'].iloc[-1]
        
        # 1. Z-Score (Mean Reversion)
        ma, std = df['Close'].rolling(50).mean(), df['Close'].rolling(50).std()
        z = (curr - ma.iloc[-1]) / std.iloc[-1]
        
        # 2. Volatility (Sigma) for Black-Scholes
        rets = df['Close'].pct_change().dropna()
        sigma = rets.std() * np.sqrt(252)
        
        # 3. ML Target & Probability of Profit (PoP)
        y = df['Close'].tail(60).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 30]])[0] # 30-day outlook
        
        # POP Calculation
        t_expiry = 30 / 365
        d2 = (np.log(curr / forecast) + (0.045 - 0.5 * sigma**2) * t_expiry) / (sigma * np.sqrt(t_expiry))
        pop = si.norm.cdf(abs(d2)) * 100
        
        # 4. TOC Squeeze (Timing)
        atr = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        squeeze = "🔒 SQUEEZE" if (2 * std.iloc[-1] < 1.5 * atr) else "🌊 EXPANSION"
        
        # 5. STRATEGY SELECTION LOGIC
        if z < -1.5 and pop > 60:
            strat = "BULLISH CALL"
            strike = round(curr * 1.02, 2)
            risk = "Low (Oversold)"
        elif z > 1.5 and pop < 40:
            strat = "BEARISH PUT"
            strike = round(curr * 0.98, 2)
            risk = "Moderate (Overbought)"
        else:
            strat = "NEUTRAL / HOLD"
            strike = "N/A"
            risk = "Wait for Signal"

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": round(z, 2),
            "PoP %": round(pop, 1), "Target": round(forecast, 2),
            "Strategy": strat, "Rec. Strike": strike, "Risk Level": risk, "State": squeeze
        }
    except: return None

# --- 3. 15-MINUTE COMMAND FRAGMENT ---

@st.fragment(run_every=900)
def options_terminal_render(ticker_list):
    st.markdown("<h2 style='color:#58a6ff;'>Sovereign Medallion | Options Alpha Engine</h2>", unsafe_allow_html=True)
    
    with st.spinner("Processing Multi-Node Options Intelligence..."):
        results = []
        # Multi-node download to protect Yahoo IP
        df_mega = yf.download(ticker_list, period="2y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        for t in ticker_list:
            analysis = analyze_options_strategy(df_mega[t], t)
            if analysis: results.append(analysis)
            time.sleep(0.5)

    col_grid, col_rec = st.columns([2, 1])

    with col_grid:
        st.subheader("📊 Strategy Matrix")
        df_res = pd.DataFrame(results)
        st.dataframe(
            df_res.style.applymap(lambda x: 'color: #00c805' if x == 'BULLISH CALL' else ('color: #ff3b3b' if x == 'BEARISH PUT' else ''), subset=['Strategy']),
            use_container_width=True, hide_index=True, height=500
        )
        
        st.subheader(f"📈 Execution Feed: {ticker_list[0]}")
        st.components.v1.html(f'<iframe src="https://s3.tradingview.com/tv.js" style="display:none;"></iframe><div id="tv_chart"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"width": "100%", "height": 400, "symbol": "{ticker_list[0]}", "interval": "D", "theme": "dark", "style": "1", "container_id": "tv_chart"}});</script>', height=400)

    with col_rec:
        st.subheader("🎯 Active Trade Alerts")
        for res in results[:10]: # Top 10 alerts
            if res['Strategy'] != "NEUTRAL / HOLD":
                color = "#00c805" if "CALL" in res['Strategy'] else "#ff3b3b"
                st.markdown(f"""
                <div class='strategy-card' style='border-left-color: {color};'>
                    <b>{res['Symbol']} - {res['Strategy']}</b><br>
                    Target: ${res['Target']} | Strike: ${res['Rec. Strike']}<br>
                    PoP: {res['PoP %']}% | Risk: {res['Risk Level']}
                </div>
                """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e3/Fidelity_Investments_logo.svg", width=120)
    st.header("Asset Universe")
    st.write("Scan Interval: **15:00 Minutes**")
    tickers = st.text_area("Ticker List", "AAPL, TSLA, NVDA, AMZN, META, NFLX, AMD, VALE3.SA, PETR4.SA, SPY, QQQ", height=250)
    ticker_list = [x.strip() for x in tickers.split(',') if x.strip()]

options_terminal_render(ticker_list)
