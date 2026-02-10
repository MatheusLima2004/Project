import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Categorized Terminal", page_icon="💎", layout="wide")
st.title("💎 The Master Terminal (Categorized)")
st.markdown("### 📊 Market Scanner: ETFs vs. Stocks vs. Moonshots")

# --- 2. MATH FORMULAS ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        return max(price, 0.0)
    except:
        return 0.0

def kelly_criterion(win_prob, win_loss_ratio):
    return max(0, (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio)

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

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_tickers = """SPY, QQQ, IWM, DIA, GLD, SLV, TLT, XLE, XLF
NVDA, TSLA, AAPL, AMD, PLTR, MSFT, AMZN, GOOGL, META
COIN, MSTR, MARA, HOOD, SOFI
MELI, NU, PBR, VALE"""

    ticker_input = st.text_area("Watchlist", default_tickers, height=300)
    tickers = [t.strip() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=4.5) / 100
    account_size = st.number_input("Account Size ($)", value=10000)

# --- 4. ENGINE ---
@st.cache_data(ttl=3600)
def scan_market(tickers):
    data_list = []
    progress_text = "Scanning..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if len(hist) < 50: continue
            
            current_price = hist['Close'].iloc[-1]
            daily_returns = hist['Close'].pct_change().dropna()
            hist_vol = daily_returns.std() * np.sqrt(252)
            spark = generate_sparkline(hist['Close'])
            
            # Asset Classification
            if hist_vol < 0.20:
                asset_type = "🛡️ Safe ETF"
            elif hist_vol > 0.60:
                asset_type = "🚀 Moonshot"
            else:
                asset_type = "🏢 Stock"

            # Options Logic
            best_contract = "No Data"
            edge_percent = 0.0
            kelly_cash = 0.0
            
            if current_price > 5:
                try:
                    exps = stock.options
                    if exps:
                        target_date = exps[min(4, len(exps)-1)]
                        days = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.now()).days
                        T = days / 365
                        chain = stock.option_chain(target_date).calls
                        chain = chain[(chain['strike'] > current_price * 0.95) & (chain['strike'] < current_price * 1.05)]
                        
                        if not chain.empty:
                            row = chain.iloc[0]
                            bs = black_scholes(current_price, row['strike'], T, risk_free_rate, hist_vol, "call")
                            mkt = row['lastPrice']
                            if mkt > 0:
                                edge_percent = ((bs - mkt) / mkt) * 100
                                best_contract = f"${row['strike']} Call ({target_date})"
                                if edge_percent > 0:
                                    kelly_pct = kelly_criterion(0.55, 2.0)
                                    kelly_cash = account_size * kelly_pct
                except: pass

            data_list.append({
                "Ticker": ticker,
                "Type": asset_type,
                "Price": current_price,
                "Trend": spark,
                "Volatility": hist_vol,
                "Best Option": best_contract,
                "Edge %": edge_percent,
                "Kelly Bet ($)": kelly_cash
            })
            
        except: continue
        my_bar.progress((i + 1) / len(tickers))
        
    my_bar.empty()
    return pd.DataFrame(data_list)

# --- 5. RENDER SECTIONS ---
if st.button("🔄 Run Categorized Scan"):
    st.cache_data.clear()

df = scan_market(tickers)

if not df.empty:
    # CREATE TABS
    tab1, tab2, tab3 = st.tabs(["🛡️ Safe ETFs (The Shield)", "🏢 Growth Stocks (The Core)", "🚀 Moonshots (The Spear)"])
    
    # Define Column Config (Reusable)
    col_config = {
        "Trend": st.column_config.TextColumn("30d Trend"),
        "Kelly Bet ($)": st.column_config.NumberColumn("Bet Size", format="$%.2f"),
        "Edge %": st.column_config.NumberColumn("Edge", format="%.1f%%"),
        "Volatility": st.column_config.NumberColumn("Risk (Vol)", format="%.1%")
    }

    # SECTION 1: ETFs
    with tab1:
        st.subheader("🛡️ Low Risk / Wealth Preservation")
        df_etf = df[df["Type"] == "🛡️ Safe ETF"].sort_values(by="Edge %", ascending=False)
        if not df_etf.empty:
            st.dataframe(df_etf.style.background_gradient(subset=['Edge %'], cmap='RdYlGn', vmin=-10, vmax=30), 
                         column_config=col_config, use_container_width=True, hide_index=True)
        else:
            st.info("No Safe ETFs found in watchlist.")

    # SECTION 2: STOCKS
    with tab2:
        st.subheader("🏢 Quality Growth Stocks")
        df_stock = df[df["Type"] == "🏢 Stock"].sort_values(by="Edge %", ascending=False)
        if not df_stock.empty:
            st.dataframe(df_stock.style.background_gradient(subset=['Edge %'], cmap='RdYlGn', vmin=-10, vmax=30), 
                         column_config=col_config, use_container_width=True, hide_index=True)
        else:
            st.info("No Stocks found.")

    # SECTION 3: MOONSHOTS
    with tab3:
        st.subheader("🚀 High Risk / High Reward")
        st.warning("⚠️ These assets have >60% Volatility. Use strict risk management.")
        df_moon = df[df["Type"] == "🚀 Moonshot"].sort_values(by="Edge %", ascending=False)
        if not df_moon.empty:
            st.dataframe(df_moon.style.background_gradient(subset=['Edge %'], cmap='RdYlGn', vmin=-10, vmax=30), 
                         column_config=col_config, use_container_width=True, hide_index=True)
        else:
            st.info("No Moonshots found.")

else:
    st.info("Click 'Run Categorized Scan' to start.")
