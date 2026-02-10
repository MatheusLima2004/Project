import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Master Terminal + TOC",
    page_icon="💎",
    layout="wide"
)

st.title("💎 The Master Terminal (TOC Edition)")
st.markdown("### Constraints & Momentum Scanner")

# --- 2. SIDEBAR SETUP ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_tickers = """MELI, NU, PBR, BSBR, BBD, VALE, ITUB
NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, META
V, MA, JPM, KO, PEP, COST, MCD, DIS
AMD, PLTR, SOFI, UBER, ABNB, SHOP, NET
WEGE3.SA, PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA
O, SCHD, JEPI, T, VZ, MO"""

    ticker_input = st.text_area("Watchlist", default_tickers, height=300)
    tickers = [t.strip() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    st.info(f"Scanning {len(tickers)} assets.")

# --- 3. HELPER FUNCTIONS ---
def generate_sparkline(series):
    bar_chars = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    if series.empty: return ""
    series = series.tail(30)
    min_val, max_val = series.min(), series.max()
    if max_val == min_val: return "▇" * 10
    spark = ""
    for price in series:
        idx = int((price - min_val) / (max_val - min_val) * 7)
        spark += bar_chars[idx]
    return spark

@st.cache_data(ttl=3600)
def scan_market(tickers):
    progress_text = "Analyzing Constraints (TOC)..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        history = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Extract Data
            df = history[ticker] if len(tickers) > 1 else history
            if df.empty or len(df) < 50: continue
            
            close = df['Close'].dropna()
            current_price = close.iloc[-1]
            
            # --- TOC METRICS (The New Logic) ---
            
            # 1. Calculate Bollinger Bands (The Constraint Boundaries)
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            # 2. Band Width (How tight is the constraint?)
            # Tighter = Higher Potential Energy
            bandwidth = ((upper_band - lower_band) / sma_20) * 100
            current_bw = bandwidth.iloc[-1]
            avg_bw = bandwidth.rolling(window=50).mean().iloc[-1]
            
            # 3. Throughput (Volume Flow)
            # Is money flowing IN to break the constraint?
            # Note: We need Volume data, usually yfinance sends it. 
            # If unavailable, we skip volume check.
            try:
                volume = df['Volume'].dropna()
                vol_sma = volume.rolling(window=20).mean().iloc[-1]
                current_vol = volume.iloc[-1]
                throughput_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0
            except:
                throughput_ratio = 1.0

            # --- STANDARD METRICS ---
            ma_50 = close.rolling(window=50).mean().iloc[-1]
            std_50 = close.rolling(window=50).std().iloc[-1]
            z_score = (current_price - ma_50) / std_50 if std_50 > 0 else 0
            trend = generate_sparkline(close)
            
            # Fundamentals
            info = yf.Ticker(ticker).info
            roe = info.get('returnOnEquity', 0)
            target = info.get('targetMeanPrice', current_price)
            roe_fmt = roe * 100 if roe else 0
            upside = ((target - current_price) / current_price) * 100
            
            # --- TOC STATUS LOGIC ---
            # If Bandwidth is < 50% of its average, it's a "Squeeze" (Constraint)
            # If Throughput (Volume) is > 1.5x average, it's "Flowing"
            
            toc_status = "Neutral"
            
            if current_bw < (avg_bw * 0.7):
                toc_status = "🔒 CONSTRAINED" # Squeeze
            elif throughput_ratio > 1.5 and z_score > 0:
                toc_status = "🌊 FLOWING" # Breakout
            elif z_score < -2.0:
                toc_status = "📉 OVERSOLD"

            # Score Update
            score = 50
            if toc_status == "🔒 CONSTRAINED": score += 20 # Potential explosion
            if toc_status == "🌊 FLOWING": score += 15 # Moving now
            if roe_fmt > 15: score += 15
            if z_score < -1.5: score += 10
            
            signal = "HOLD"
            if score >= 80: signal = "💎 STRONG BUY"
            elif score <= 40: signal = "⚠️ AVOID"
            
            results.append({
                "Ticker": ticker,
                "Price": current_price,
                "Trend": trend,
                "TOC Status": toc_status,
                "Bandwidth %": current_bw,
                "Throughput (Vol)": throughput_ratio,
                "Score": score,
                "Signal": signal
            })
            
        except Exception:
            continue
            
        my_bar.progress((i + 1) / len(tickers))
        
    my_bar.empty()
    return pd.DataFrame(results)

# --- 4. MAIN APP ---
if st.button("🔄 Run TOC Scanner"):
    st.cache_data.clear()

df = scan_market(tickers)

if not df.empty:
    df = df.sort_values(by="Score", ascending=False)
    
    st.dataframe(
        df.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=30, vmax=90)
          .format({
              "Price": "${:.2f}", 
              "Bandwidth %": "{:.2f}%", 
              "Throughput (Vol)": "{:.1f}x"
          }),
        column_config={
            "Trend": st.column_config.TextColumn("30d Trend"),
            "TOC Status": st.column_config.TextColumn("Constraint State"),
        },
        height=800,
        use_container_width=True,
        hide_index=True
    )
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "toc_scan.csv", "text/csv")

else:
    st.warning("Click 'Run TOC Scanner' to start.")

with st.expander("📖 TOC Glossary"):
    st.markdown("""
    * **🔒 CONSTRAINED (The Squeeze):** Volatility is abnormally low. The price is being compressed. A large move is imminent (Release of energy).
    * **🌊 FLOWING (Throughput):** High volume is pushing the price. The constraint has broken.
    * **Bandwidth %:** The width of the 'river'. Lower is tighter (more constrained).
    * **Throughput:** Volume multiplier. `2.0x` means double the normal money flow.
    """)
