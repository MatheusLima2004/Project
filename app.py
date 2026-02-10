import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hedge Fund Terminal", page_icon="🏦", layout="wide")
st.title("🏦 The Hedge Fund Terminal (Stocks + Options)")

# --- 2. MATH FORMULAS ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates theoretical fair value of an option."""
    # S: Spot price, K: Strike, T: Time to expiry (years), r: Risk-free rate, sigma: Volatility
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        return max(price, 0.0) # Price cannot be negative
    except:
        return 0.0

def kelly_criterion(win_prob, win_loss_ratio):
    """Calculates optimal bet size %."""
    # f = (p(b+1) - 1) / b
    return max(0, (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    tickers_input = st.text_area("Tickers", "NVDA, TSLA, AAPL, AMD, PLTR, SPY, QQQ", height=100)
    tickers = [t.strip() for t in tickers_input.split(',')]
    
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=4.5) / 100
    account_size = st.number_input("Account Size ($)", value=10000)

# --- 4. STOCK SCANNER ENGINE ---
@st.cache_data(ttl=3600)
def get_data(tickers):
    data_list = []
    
    # Progress bar for UX
    my_bar = st.progress(0, text="Scanning Market...")
    
    for i, ticker in enumerate(tickers):
        try:
            # Get Stock Data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            
            if len(hist) < 50: continue
            
            current_price = hist['Close'].iloc[-1]
            
            # 1. Volatility (Annualized)
            daily_returns = hist['Close'].pct_change().dropna()
            hist_vol = daily_returns.std() * np.sqrt(252)
            
            # 2. Trend (SMA)
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            trend = "BULLISH 🐂" if current_price > sma_50 else "BEARISH 🐻"
            
            # 3. Momentum (RSI)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            data_list.append({
                "Ticker": ticker,
                "Price": current_price,
                "Trend": trend,
                "RSI": rsi,
                "Volatility": hist_vol,
                "SMA_50": sma_50
            })
            
        except: continue
        my_bar.progress((i + 1) / len(tickers))
        
    my_bar.empty()
    return pd.DataFrame(data_list)

# --- 5. OPTIONS ENGINE ---
def analyze_options(ticker, current_price, hist_vol, r):
    stock = yf.Ticker(ticker)
    
    # Get Next Expiration Date
    try:
        exps = stock.options
        if not exps: return None
        # Pick an expiration ~30 days out (Optimal for swing trading)
        target_date = exps[min(4, len(exps)-1)] 
    except: return None
    
    # Get Option Chain
    chain = stock.option_chain(target_date)
    calls = chain.calls
    
    # Filter for "Near the Money" (Strike within 10% of price)
    calls = calls[(calls['strike'] > current_price * 0.95) & (calls['strike'] < current_price * 1.05)]
    
    if calls.empty: return None
    
    # Calculate Black-Scholes for each
    opportunities = []
    days_to_exp = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.now()).days
    T = days_to_exp / 365
    
    for index, row in calls.iterrows():
        bs_price = black_scholes(current_price, row['strike'], T, r, hist_vol, "call")
        market_price = row['lastPrice']
        
        # "Edge" = (Value - Price) / Price
        edge = ((bs_price - market_price) / market_price) * 100 if market_price > 0 else 0
        
        opportunities.append({
            "Contract": f"{ticker} ${row['strike']} Call",
            "Exp Date": target_date,
            "Strike": row['strike'],
            "Market Price": market_price,
            "Fair Value (BS)": bs_price,
            "Edge %": edge,
            "Implied Vol": row['impliedVolatility']
        })
        
    return pd.DataFrame(opportunities).sort_values(by="Edge %", ascending=False).head(1)

# --- 6. RENDER DASHBOARD ---
df_stocks = get_data(tickers)

if not df_stocks.empty:
    # TAB 1: STOCKS
    st.subheader("📊 Stock Overview")
    st.dataframe(df_stocks.style.format({"Price": "${:.2f}", "RSI": "{:.1f}", "Volatility": "{:.1%}"}))

    # TAB 2: OPTIONS LAB
    st.markdown("---")
    st.subheader("🎯 Options Sniper (Black-Scholes Model)")
    st.write("Searching for undervalued contracts...")
    
    for index, row in df_stocks.iterrows():
        if row['Trend'] == "BULLISH 🐂": # Only look for calls on Bullish stocks
            opt_df = analyze_options(row['Ticker'], row['Price'], row['Volatility'], risk_free_rate)
            
            if opt_df is not None and not opt_df.empty:
                best_opt = opt_df.iloc[0]
                
                # Kelly Bet Logic (Simplified)
                # Assuming 55% Win Rate, 2:1 Reward/Risk
                kelly_pct = kelly_criterion(0.55, 2.0) * 100 
                kelly_cash = account_size * (kelly_pct / 100)
                
                with st.expander(f"🚀 {row['Ticker']} Opportunity found!"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Contract", best_opt['Contract'])
                    c2.metric("Market Price", f"${best_opt['Market Price']:.2f}")
                    c3.metric("Fair Value", f"${best_opt['Fair Value (BS)']:.2f}", 
                              delta=f"{best_opt['Edge %']:.1f}% Undervalued")
                    
                    st.info(f"💰 **Kelly Bet Sizing:** The math suggests betting **{kelly_pct:.1f}%** of your account (${kelly_cash:.0f}) on this trade.")
                    st.dataframe(opt_df)
else:
    st.warning("No data found.")
