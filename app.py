import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ProTrader AI Terminal",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom "Fidelity-Dark" CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .stMetric { background-color: #1e2127; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .stDataFrame { border: 1px solid #333; }
    div[data-testid="stExpander"] { background-color: #1e2127; border-radius: 8px; }
    a { color: #4DA6FF; text-decoration: none; }
    a:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)

st.title("📺 ProTrader AI Terminal | Live TV & Algo Scanner")

# --- 2. MATH & AI ENGINE ---

def generate_market_resume(df):
    """Generates a text summary of the trading day based on data."""
    if df.empty: return "Waiting for data..."
    
    # 1. Market Sentiment (Based on SPY)
    spy = df[df['Ticker'] == 'SPY']
    if not spy.empty:
        spy_change = spy['Change %'].values[0]
        if spy_change > 0.5: sentiment = "BULLISH 🐂"
        elif spy_change < -0.5: sentiment = "BEARISH 🐻"
        else: sentiment = "NEUTRAL ⚖️"
    else:
        sentiment = "UNKNOWN"
        spy_change = 0

    # 2. Top Movers
    top_gainer = df.sort_values(by="Change %", ascending=False).iloc[0]
    top_loser = df.sort_values(by="Change %", ascending=True).iloc[0]
    
    # 3. Sector Rotation (Using Tech vs Energy proxy if available)
    # Simple logic: If Tech (QQQ) > Bonds (TLT), it's "Risk On"
    
    summary = f"""
    ### 📝 Executive Market Resume
    * **Market Mood:** The S&P 500 is **{sentiment}** today ({spy_change:+.2f}%).
    * **Top Performer:** **{top_gainer['Ticker']}** is leading the pack, up **{top_gainer['Change %']:+.2f}%**.
    * **Laggard:** **{top_loser['Ticker']}** is dragging, down **{top_loser['Change %']:+.2f}%**.
    * **Institutional Action:** Algorithms are detecting **{len(df[df['Algo Signal'].str.contains('ICEBERG')])}** hidden accumulation zones (Icebergs) right now.
    """
    return summary

def calculate_rsi(series, period=14):
    if len(series) < period: return 50
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_vwap(df):
    try:
        v = df['Volume'].values
        p = df['Close'].values
        return (p * v).cumsum() / v.cumsum()
    except: return pd.Series([0]*len(df))

def detect_stealth_algo(df):
    try:
        recent = df.tail(5)
        avg_vol = df['Volume'].mean()
        curr_vol = recent['Volume'].mean()
        price_change = abs(recent['Close'].pct_change().mean()) * 100
        
        if curr_vol > (avg_vol * 1.5) and price_change < 0.1: return "🧊 ICEBERG (Accumulation)"
        elif curr_vol > (avg_vol * 2.0) and price_change > 1.5: return "🌊 MOMENTUM SURGE"
        else: return "---"
    except: return "---"

def ai_analyst(row):
    rsi = row['RSI']
    price = row['Price']
    vwap = row['VWAP']
    algo = row['Algo Signal']
    
    if algo == "🧊 ICEBERG (Accumulation)" and rsi < 50: return "🔥 STRONG BUY: Whale detected."
    if rsi < 30 and price < vwap: return "✅ BUY THE DIP: Oversold."
    if rsi > 60 and price > vwap and algo == "🌊 MOMENTUM SURGE": return "🚀 RIDE TREND: Breakout."
    if rsi > 75: return "⚠️ SELL/TRIM: Overbought."
    return "😴 HOLD/WAIT"

def calculate_graham_value(eps, book_value):
    try:
        if eps is not None and book_value is not None and eps > 0 and book_value > 0:
            return math.sqrt(22.5 * eps * book_value)
    except: pass
    return 0

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    live_mode = st.toggle("🔴 LIVE MODE (Auto-Refresh)", value=False)
    if live_mode: st.caption("Refreshing every 60s...")
    st.divider()
    
    default_tickers = "NVDA, TSLA, AAPL, AMD, MSFT, AMZN, GOOGL, META, PLTR, SPY, QQQ, IWM"
    ticker_input = st.text_area("Watchlist", default_tickers, height=150)
    stock_list = [t.strip() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    st.divider()
    rf_rate = st.number_input("Risk Free Rate (%)", value=4.5) / 100

# --- 4. DATA ENGINE ---
@st.cache_data(ttl=60 if live_mode else 3600)
def fetch_market_data(stocks):
    data_points = []
    for ticker in stocks:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo", interval="1d")
            intraday = stock.history(period="5d", interval="60m")
            if hist.empty: continue
            
            curr_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else curr_price
            pct_change = ((curr_price - prev_close) / prev_close) * 100
            
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            vwap = calculate_vwap(intraday).iloc[-1] if not intraday.empty else curr_price
            algo_signal = detect_stealth_algo(intraday) if not intraday.empty else "---"
            
            info = stock.info
            target_price = info.get('targetMeanPrice', 0)
            if target_price and target_price > 0: upside = ((target_price - curr_price) / curr_price) * 100
            else: upside = 0
            fair_val = calculate_graham_value(info.get('trailingEps', 0), info.get('bookValue', 0))
            
            news_items = info.get('news', [])
            top_news = news_items[0] if news_items else {}
            
            row_data = {
                "Ticker": ticker,
                "Price": curr_price,
                "Change %": pct_change,
                "RSI": rsi,
                "VWAP": vwap,
                "Algo Signal": algo_signal,
                "Target Price": target_price,
                "Upside %": upside,
                "Fair Value": fair_val,
                "Headline": top_news.get('title', 'No recent news'),
                "Link": top_news.get('link', '#'),
                "Publisher": top_news.get('publisher', 'Unknown')
            }
            row_data["🤖 AI Tip"] = ai_analyst(row_data)
            data_points.append(row_data)
        except Exception: continue
    return pd.DataFrame(data_points)

# --- 5. RENDER DASHBOARD ---
placeholder = st.empty()

while True:
    with placeholder.container():
        df = fetch_market_data(stock_list)
        
        if df.empty:
            st.warning("⚠️ No data loaded yet. Markets might be closed or connection is slow.")
        else:
            # --- MARKET RESUME SECTION (NEW) ---
            st.info(generate_market_resume(df))
            
            # METRICS ROW
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📅 Date", datetime.now().strftime("%Y-%m-%d"))
            c2.metric("🕒 Time", datetime.now().strftime("%H:%M:%S"))
            c3.metric("🦅 Risk Free Rate", f"{rf_rate*100:.2f}%")
            
            market_proxy = df[df['Ticker'] == 'SPY']
            if not market_proxy.empty:
                spy_price = market_proxy['Price'].values[0]
                spy_change = market_proxy['Change %'].values[0]
                c4.metric("🇺🇸 SPY Market", f"${spy_price:.2f}", f"{spy_change:.2f}%")
            else:
                c4.metric("Active Assets", len(df))

            # TABS
            tab_tv, tab_main, tab_edu = st.tabs(["📺 Live TV & News", "🚀 AI Scanner", "📘 Education"])

            with tab_tv:
                col_video, col_news = st.columns([2, 1])
                with col_video:
                    st.subheader("🔴 Bloomberg Global Financial News (Live)")
                    st.video("https://www.youtube.com/watch?v=dp8PhLsUcFE")
                    st.caption("Stream provided via YouTube. Content subject to broadcaster availability.")

                with col_news:
                    st.subheader("📰 Breaking Headlines")
                    st.markdown("---")
                    for i, row in df.iterrows():
                        with st.container():
                            st.markdown(f"**{row['Ticker']}**")
                            st.markdown(f"[{row['Headline']}]({row['Link']})")
                            st.caption(f"Source: {row['Publisher']}")
                            st.divider()

            with tab_main:
                st.subheader("🤖 Artificial Intelligence Insights")
                st.dataframe(
                    df.style.background_gradient(subset=['Upside %'], cmap='RdYlGn', vmin=-10, vmax=30)
                    .format({
                        "Price": "${:.2f}", 
                        "Change %": "{:+.2f}%", 
                        "RSI": "{:.0f}", 
                        "VWAP": "${:.2f}", 
                        "Target Price": "${:.2f}", 
                        "Upside %": "{:+.1f}%",
                        "Fair Value": "${:.2f}"
                    }),
                    column_config={
                        "Algo Signal": st.column_config.TextColumn("Inst. Footprint"),
                        "🤖 AI Tip": st.column_config.TextColumn("AI Action Plan", width="medium"),
                        "Headline": st.column_config.TextColumn("Top Story", width="large"),
                        "Link": st.column_config.LinkColumn("Read")
                    },
                    use_container_width=True, hide_index=True, height=600
                )

            with tab_edu:
                st.header("📘 Options Trading Academy")
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("🟢 The Basics: Calls vs. Puts", expanded=True):
                        st.write("**Call:** You think price goes UP.")
                        st.write("**Put:** You think price goes DOWN.")
                with col2:
                    with st.expander("📐 The 'Greeks'", expanded=True):
                        st.write("**Delta:** Sensitivity to Price.")
                        st.write("**Theta:** Sensitivity to Time (Decay).")

    if not live_mode:
        break
    time.sleep(60)
