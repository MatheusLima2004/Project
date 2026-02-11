import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- 1. SOVEREIGN THEME & LAYOUT ---
st.set_page_config(page_title="Medallion Sovereign Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1217; color: #e1e4e8; }
    .stMetric { background-color: #161c23; border: 1px solid #30363d; padding: 15px; border-radius: 4px; }
    .news-card { background-color: #161c23; padding: 12px; border-radius: 4px; border-left: 4px solid #58a6ff; margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DEEP OPTIONS & MEDALLION ML ENGINE ---

def run_sovereign_logic(df, ticker):
    try:
        if df is None or len(df) < 50: return None
        
        # Z-Score (Mean Reversion)
        ma_50 = df['Close'].rolling(50).mean()
        std_50 = df['Close'].rolling(50).std()
        curr = df['Close'].iloc[-1]
        z_score = (curr - ma_50.iloc[-1]) / std_50.iloc[-1]
        
        # Kelly Criterion (Sizing)
        rets = df['Close'].pct_change().dropna()
        win_rate = len(rets[rets > 0]) / len(rets)
        win_loss = rets[rets > 0].mean() / abs(rets[rets < 0].mean()) if not rets[rets < 0].empty else 1
        kelly = (win_rate * (win_loss + 1) - 1) / win_loss
        
        # ML Price Forecast (30-Day Projection)
        y = df['Close'].tail(60).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        forecast = model.predict([[len(y) + 30]])[0]
        
        # Options ML: Probability of Success (PoP)
        sigma = rets.std() * np.sqrt(252)
        days_to_expiry = 30 / 365
        # d2 calculation for probability stock is above forecast
        d2 = (np.log(curr / forecast) + (0.045 - 0.5 * sigma**2) * days_to_expiry) / (sigma * np.sqrt(days_to_expiry))
        pop = si.norm.cdf(abs(d2)) * 100
        
        # TOC Squeeze
        atr = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        squeeze = "🔒 SQUEEZE" if (2 * std_50.iloc[-1] < 1.5 * atr) else "🌊 EXPANSION"

        return {
            "Symbol": ticker, "Price": curr, "Z-Score": z_score, 
            "Kelly %": max(0, kelly * 100), "TOC State": squeeze, 
            "ML Target": forecast, "PoP %": pop, "Volatility": sigma
        }
    except: return None

# --- 3. 12-MINUTE DEEP SCAN FRAGMENT ---

@st.fragment(run_every=720)
def sovereign_render(ticker_list):
    st.markdown("<h2 style='color:#58a6ff;'>Sovereign Medallion Command</h2>", unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Cycle Interval", "12:00 MIN", "DEEP ML SCAN")
    col_stat2.metric("Sync Status", "LIVE", f"T+{datetime.now().strftime('%S')}s")
    col_stat3.metric("Assets Analyzed", len(ticker_list))

    with st.spinner("Executing Deep ML Intelligence..."):
        df_mega = yf.download(ticker_list, period="2y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        results, news_feed = [], []
        for t in ticker_list:
            res = run_sovereign_logic(df_mega[t], t)
            if res: 
                results.append(res)
                try:
                    n = yf.Ticker(t).news[:1]
                    for item in n: news_feed.append({"symbol": t, "title": item['title'], "link": item['link']})
                except: pass

    col_main, col_news = st.columns([3, 1])
    
    with col_main:
        if results:
            df_res = pd.DataFrame(results)
            
            # --- THE FIX: Conditional Styling ---
            # This ensures the code doesn't crash if 'PoP %' is missing
            styled_df = df_res.style
            if 'PoP %' in df_res.columns:
                styled_df = styled_df.background_gradient(subset=['PoP %'], cmap='RdYlGn')
            
            st.dataframe(
                styled_df.format({
                    "Price": "${:.2f}", "Z-Score": "{:.2f}", "Kelly %": "{:.1f}%", 
                    "ML Target": "${:.2f}", "PoP %": "{:.1f}%", "Volatility": "{:.1%}"
                }),
                use_container_width=True, hide_index=True, height=500
            )
        else:
            st.error("No data could be analyzed. Check your internet connection or ticker symbols.")
        
        # TradingView Chart
        st.subheader(f"📈 Real-Time Feed: {ticker_list[0]}")
        st.components.v1.html(f"""
            <div id="tv-chart" style="height:450px;"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>
            new TradingView.widget({{"width": "100%", "height": 450, "symbol": "{ticker_list[0]}", "interval": "D", "theme": "dark", "style": "1", "locale": "en", "enable_publishing": false, "allow_symbol_change": true, "container_id": "tv-chart"}});
            </script>
        """, height=450)

    with col_news:
        st.subheader("📰 Watchlist Intel")
        for news in news_feed:
            st.markdown(f"""<div class='news-card'><b>{news['symbol']}</b>: <a href='{news['link']}' target='_blank' style='color:white;text-decoration:none;'>{news['title']}</a></div>""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Asset Universe")
    default_list = "
