import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Master Terminal",
    page_icon="💎",
    layout="wide"
)

st.title("💎 The Master Terminal")
st.markdown("### Real-Time Value & Momentum Scanner")

# --- 2. SIDEBAR SETUP ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Default Watchlist
    default_tickers = """MELI, NU, PBR, BSBR, BBD, VALE, ITUB
NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, META
V, MA, JPM, KO, PEP, COST, MCD, DIS
AMD, PLTR, SOFI, UBER, ABNB, SHOP, NET
WEGE3.SA, PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA
O, SCHD, JEPI, T, VZ, MO"""

    ticker_input = st.text_area("Watchlist (Comma Separated)", default_tickers, height=300)
    
    # Process the input string into a list
    tickers = [t.strip() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    
    st.info(f"Loaded {len(tickers)} stocks.")

# --- 3. HELPER FUNCTIONS ---
def generate_sparkline(series):
    """Generates a text-based sparkline graph"""
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
    progress_text = "Scanning Global Markets... Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # Bulk Download
        history = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        st.error(f"Error connecting to Yahoo Finance: {e}")
        return pd.DataFrame()

    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Extract Data
            df = history[ticker] if len(tickers) > 1 else history
            if df.empty or len(df) < 50: continue
            
            close = df['Close'].dropna()
            current_price = close.iloc[-1]
            
            # --- TECHNICALS (Broken into steps to avoid copy errors) ---
            
            # 1. Moving Average (50 day)
            ma_50 = close.rolling(window=50).mean().iloc[-1]
            
            # 2. Standard Deviation (50 day)
            std_50 = close.rolling(window=50).std().iloc[-1]
            
            # 3. Z-Score Calculation
            if std_50 > 0:
                z_score = (current_price - ma_50) / std_50
            else:
                z_score = 0
            
            trend = generate_sparkline(close)
            
            # --- FUNDAMENTALS ---
            info = yf.Ticker(ticker).info
            roe = info.get('returnOnEquity', 0)
            debt_eq = info.get('debtToEquity', 0)
            target = info.get('targetMeanPrice', current_price)
            
            # Formats
            roe_fmt = roe * 100 if roe else 0
            upside = ((target - current_price) / current_price) * 100
            
            # --- SCORING ---
            score = 50
            if roe_fmt > 15: score += 15
            if debt_eq < 100 and debt_eq > 0: score += 10
            if z_score < -1.5: score += 10
            if upside > 20: score += 15
            
            # Signal
            signal = "HOLD"
            if score >= 80: signal = "💎 STRONG BUY"
            elif score <= 40: signal = "⚠️ AVOID"
            elif z_score <= -2.0: signal = "🛒 OVERSOLD"
            
            results.append({
                "Ticker": ticker,
                "Price": current_price,
                "Trend": trend,
                "Z-Score": z_score,
                "ROE %": roe_fmt,
                "Upside %": upside,
                "Score": score,
                "Signal": signal
            })
            
        except Exception:
            continue
            
        # Update Progress
        my_bar.progress((i + 1) / len(tickers))
        
    my_bar.empty()
    return pd.DataFrame(results)

# --- 4. MAIN APP LOGIC ---
if st.button("🔄 Run Scanner"):
    st.cache_data.clear()

# Run the scan
df = scan_market(tickers)

if not df.empty:
    # Sort by Score
    df = df.sort_values(by="Score", ascending=False)
    
    # Display Interactive Table
    st.dataframe(
        df.style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=30, vmax=90)
          .format({"Price": "${:.2f}", "Z-Score": "{:+.2f}", "ROE %": "{:.1f}%", "Upside %": "{:+.1f}%"}),
        column_config={
            "Trend": st.column_config.TextColumn("30d Trend", help="Visual price history"),
            "Signal": st.column_config.TextColumn("Verdict"),
        },
        height=800,
        use_container_width=True,
        hide_index=True
    )
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results (CSV)", csv, "market_scan.csv", "text/csv")

else:
    st.warning("Click 'Run Scanner' to start.")

# --- 5. GLOSSARY ---
with st.expander("📖 Trader's Glossary (Click to Expand)"):
    st.markdown("""
    * **Z-Score:** How 'stretched' the price is. `<-2.0` is Cheap. `>+2.0` is Expensive.
    * **Trend:** A 30-day mini-chart. Left is old, Right is new.
    * **ROE %:** Return on Equity. `>15%` indicates a high-quality company.
    * **Score:** Our custom 0-100 rating combining value, safety, and momentum.
    """)
