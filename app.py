import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import feedparser
import requests
# --- 1. System Config & CSS (系統配置與樣式) ---
st.set_page_config(page_title="FinData AI Terminal", page_icon="📊", layout="wide")
st.markdown("""
<style>
    /* 頂部新聞滾動條 */
    .news-ticker {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #2980b9;
        padding: 8px; margin-bottom: 5px; border-radius: 4px;
        font-family: 'Roboto Mono', monospace; font-size: 13px; color: #2c3e50;
    }
    /* 作者資訊欄 */
    .author-line {
        font-size: 14px; color: #57606f; margin-bottom: 20px; border-bottom: 1px solid #dfe4ea; padding-bottom: 10px;
    }
    /* 深度解讀框 */
    .insight-box {
        background-color: #f1f8e9;
        border-left: 4px solid #7cb342;
        padding: 12px;
        border-radius: 4px;
        font-size: 14px;
        margin-top: 10px;
        color: #2d3436;
        line-height: 1.5;
    }
    /* 警示解讀框 */
    .insight-box-warn {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 12px; border-radius: 4px; font-size: 14px; margin-top: 10px; color: #2d3436;
    }
</style>
""", unsafe_allow_html=True)
# --- 2. Title & Author Info (標題與作者資訊) ---
st.title("📊 Integrated Data Science Dashboard: Quantitative Analysis of Crypto & Macro Assets")
st.title("  基于多源数据的金融资产量化分析与可视化看板")
st.markdown("""
<div class="author-line">
    <b>Author:</b> Fan Xing (樊星) | <b>ID:</b> MC566736 | <b>Institution:</b> University of Macau | <b>Course:</b> CISC7201 Data Science Programming
</div>
""", unsafe_allow_html=True)
with st.expander("ℹ️ Project Background & Motivation (項目背景與動機)"):
    st.markdown("""
    **Motivation（動機）:**
    In the volatile cryptocurrency market, retail investors often lack professional tools to analyze the correlation between crypto assets and macroeconomic factors.
    <br>(在波動劇烈的加密貨幣市場中，散戶投資者往往缺乏專業工具來分析加密資產與宏觀經濟因素之間的相關性。)
    
    **Objective（目標）:**
    This dashboard implements an **end-to-end data science pipeline** (Collection $\\rightarrow$ Cleaning $\\rightarrow$ Modeling $\\rightarrow$ Visualization) to provide:
    <br>（这个仪表板实现了一个端到端的数据科学管道（收集→清洗→建模→可视化）来提供：）
    1. **Real-time Monitoring:** Price action and Sentiment analysis.
    <br>(即時監測：價格走勢與情緒分析。)
    2. **Risk Assessment:** Volatility, Sharpe Ratio, and Max Drawdown.
    <br>(風險評估：波動率、夏普比率與最大回撤。)
    3. **Predictive Modeling:** Monte Carlo simulations for future price paths.
    <br>(預測建模：蒙特卡洛模擬未來價格路徑。)
    """, unsafe_allow_html=True)
# --- 3. Sidebar Control ---
st.sidebar.header("🎛️ Analysis Controls (分析控制台)")
# 3.1 Asset Selection
ticker_map = {
    'Bitcoin (BTC)': 'BTC-USD', 'Ethereum (ETH)': 'ETH-USD',
    'Nasdaq 100 (QQQ)': 'QQQ', 'S&P 500 (SPY)': 'SPY',
    'NVIDIA (NVDA)': 'NVDA', 'Tesla (TSLA)': 'TSLA',
    'Gold (GLD)': 'GLD'
}
macro_tickers = {
    'Gold (黃金)': 'GLD', 'US 10Y Bond (美債)': '^TNX',
    'Dollar Index (美元DXY)': 'DX-Y.NYB', 'VIX (恐慌指數)': '^VIX'
}
selected_label = st.sidebar.selectbox("🎯 Target Asset (核心標的)", list(ticker_map.keys()))
selected_ticker = ticker_map[selected_label]
compare_label = st.sidebar.selectbox("⚖️ Benchmark (對比基準)", ['S&P 500 (SPY)', 'Nasdaq 100 (QQQ)'], index=1)
compare_ticker = ticker_map.get(compare_label, 'QQQ')
# 3.2 Time Window
st.sidebar.subheader("⏱️ Time Window (時間週期)")
if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=365), datetime.now())
def update_dates():
    selection = st.session_state.quick_select
    end = datetime.now()
    if selection == "⚡ Live (今日實盤)": start = end
    elif selection == "1M (近1月)": start = end - timedelta(days=30)
    elif selection == "3M (近3月)": start = end - timedelta(days=90)
    elif selection == "1Y (近1年)": start = end - timedelta(days=365)
    elif selection == "YTD (今年以來)": start = datetime(end.year, 1, 1)
    else: start = datetime(2023, 1, 1)
   
    if selection != "⚡ Live (今日實盤)":
        st.session_state.date_range = (start, end)
time_filter = st.sidebar.radio(
    "Quick Select (快速選擇)",
    ["⚡ Live (今日實盤)", "1M (近1月)", "3M (近3月)", "1Y (近1年)", "YTD (今年以來)", "All (全部)"],
    index=3, key='quick_select', on_change=update_dates
)
if time_filter != "⚡ Live (今日實盤)":
    select_dates = st.sidebar.date_input("📅 Custom Range (自訂範圍)", value=st.session_state.date_range, max_value=datetime.now())
    start_date, end_date = select_dates if isinstance(select_dates, tuple) and len(select_dates)==2 else st.session_state.date_range
    interval_setting = "1d"
else:
    start_date = datetime.now() - timedelta(days=5); end_date = datetime.now()
    interval_setting = "15m"
    st.sidebar.success("⚡ High-Frequency Mode (15分鐘高頻模式)")
prediction_days = st.sidebar.slider("🔮 Forecast Horizon (預測步長)", 7, 60, 30)
# Data Source Info
st.sidebar.markdown("---")
st.sidebar.caption("**Data Provenance:** Yahoo Finance API, Alternative.me API, RSS Feeds. ")
# --- 4. Data Engine ---
@st.cache_data(ttl=300)
def load_data(ticker, start, end, interval):
    try:
        period = "5d" if interval == "15m" else None
        df = yf.download(ticker, start=start if not period else None, end=end if not period else None,
                         period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except: return pd.DataFrame()
@st.cache_data(ttl=3600)
def get_fng_index():
    try:
        url = "https://api.alternative.me/fng/?limit=2"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            data = r.json()['data']
            today_val = int(data[0]['value'])
            today_label = data[0]['value_classification']
            yesterday_val = int(data[1]['value'])
            
            change_val = today_val - yesterday_val
            return today_val, today_label, change_val
    except: pass
    return None, "N/A", 0
with st.spinner('🚀 Establishing Data Pipeline... (建立數據管道...)'):
    main_df = load_data(selected_ticker, start_date, end_date, interval_setting)
    bench_df = load_data(compare_ticker, start_date, end_date, interval_setting)
   
    macro_data = {}
    if time_filter != "⚡ Live (今日實盤)":
        for n, t in macro_tickers.items():
            d = load_data(t, start_date, end_date, interval_setting)
            if not d.empty: macro_data[n] = d['Close']
   
    fng_val, fng_label, fng_change = get_fng_index()
if main_df.empty: st.error("⚠️ Data connection failed. Please adjust filters. (數據連接失敗。請調整篩選條件。)"); st.stop()
# --- 5. News Ticker (新聞流) ---
def get_rss():
    try:
        url = "https://finance.yahoo.com/news/rssindex"
        if "BTC" in selected_ticker: url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        f = feedparser.parse(url)
        return " | ".join([f"📰 {e.title}" for e in f.entries[:5]])
    except: return "Initializing Global News Stream... (初始化全球新聞流...)"
st.markdown(f'<div class="news-ticker"><marquee>{get_rss()}</marquee></div>', unsafe_allow_html=True)
# --- 6. Unified KPI Board (統一指標看板) ---
fng_val, fng_label, fng_change = get_fng_index()
c1, c2, c3, c4, c5 = st.columns(5)

curr_p = main_df['Close'].iloc[-1]
prev_p = main_df['Close'].iloc[-2]
ret_pct = (curr_p - prev_p) / prev_p * 100
total_ret = (curr_p / main_df['Close'].iloc[0] - 1) * 100
returns = main_df['Close'].pct_change().dropna()
sharpe = (returns.mean() - 0.04/252) / returns.std() * np.sqrt(252)
volatility = returns.std() * np.sqrt(252) * 100

time_label = "15m" if interval_setting == "15m" else "Day"
# Column 1: Price
c1.metric(
    "Price（現價）", 
    f"${curr_p:,.2f}", 
    f"{ret_pct:+.2f}% vs Prev. {time_label}" 
)
# Column 2: Total Return
c2.metric(
    "Return (累計回報)", 
    f"{total_ret:+.2f}%",
    help="Return since the start of the selected date range"
)
# Column 3: Sharpe
c3.metric("Sharpe (夏普比率)", f"{sharpe:.2f}")
# Column 4: Volatility
c4.metric("Volatility (年化波動)", f"{volatility:.1f}%", delta_color="inverse")
# Column 5: Fear & Greed
with c5:
    if fng_val is not None:
        st.metric(
            f"F&G Index（恐慌指數）)", 
            f"{fng_val}/100  ({fng_label}", 
            f"{fng_change:+d} vs Yest.", 
            delta_color="off"
        )
    else:
        st.metric("Sentiment", "N/A", "API Error")
# --- 7. Main Tabs (核心功能區) ---
tabs = st.tabs(["🕯️ Market Overview (市場概覽)", "📈 Advanced Analytics (深度量化)", "🎲 Monte Carlo (隨機模擬)"])
# === Tab 1: Market Overview ===
with tabs[0]:
    # K-Line Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.2, 0.7])
    fig.add_trace(go.Candlestick(x=main_df.index, open=main_df['Open'], high=main_df['High'], low=main_df['Low'], close=main_df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Bar(x=main_df.index, y=main_df['Volume'], marker_color='rgba(0, 150, 136, 0.5)', name='Vol'), row=2, col=1)
   
    # MA50
    ma50 = main_df['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=main_df.index, y=ma50, line=dict(color='blue', width=1), name='MA50'), row=1, col=1)
   
    # Smart Annotation (High/Low)
    hi_idx = main_df['High'].idxmax(); hi_val = main_df['High'].max()
    lo_idx = main_df['Low'].idxmin(); lo_val = main_df['Low'].min()
    fig.add_annotation(x=hi_idx, y=hi_val, text=f"High: {hi_val:,.0f}", showarrow=True, arrowhead=1, row=1, col=1)
    fig.add_annotation(x=lo_idx, y=lo_val, text=f"Low: {lo_val:,.0f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1)
   
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, title=f"{selected_label} Price Action Analysis (價格走勢分析)", margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    vol_ratio = main_df['Volume'].iloc[-1] / main_df['Volume'].mean()
    price_pos = "above" if curr_p > ma50.iloc[-1] else "below"
    trend_cn = "多頭 (Bullish)" if price_pos == "above" else "空頭 (Bearish)"
   
    st.markdown(f"""
    <div class="insight-box">
        <b>💡 Technical Analysis Insight (技術面深度解讀):</b><br>
        1. <b>Trend Structure:</b> The asset is currently trading <b>{price_pos}</b> its 50-period Moving Average, suggesting a <b>{trend_cn}</b> medium-term trend.<br>
        (1. <b>趨勢結構：</b> 資產目前交易於其50期移動平均線<b>{price_pos}</b>，暗示中期的<b>{trend_cn}</b>趨勢。)<br>
        2. <b>Volume Profile:</b> Today's trading volume is <b>{vol_ratio:.2f}x</b> the average. { 'High volume confirms the trend strength.' if vol_ratio > 1.2 else 'Low volume indicates market consolidation.' }<br>
        (2. <b>成交量概況：</b> 今日成交量為平均水平的<b>{vol_ratio:.2f}倍</b>。{ '高成交量確認趨勢強度。' if vol_ratio > 1.2 else '低成交量顯示市場盤整。' })<br>
        3. <b>Range:</b> The price fluctuated between <b>${lo_val:,.0f}</b> and <b>${hi_val:,.0f}</b> within the selected period.<br>
        (3. <b>範圍：</b> 價格在選定期間內波動於<b>${lo_val:,.0f}</b>至<b>${hi_val:,.0f}</b>之間。)<br>
    </div>
    """, unsafe_allow_html=True)
# === Tab 2: Advanced Analytics ===
with tabs[1]:
    st.subheader("📊 Quantitative Factor Analysis (量化因子分析)")
   
    # Row 1: Drawdown & Correlation
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**1. Max Drawdown (最大回撤深度)**")
        roll_max = main_df['Close'].cummax()
        dd = (main_df['Close'] / roll_max - 1)
        fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd, fill='tozeroy', line=dict(color='#e74c3c'), name='Drawdown'))
        fig_dd.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), yaxis_title="% from Peak (% 從峰值)")
        st.plotly_chart(fig_dd, use_container_width=True)
       
        # Enhanced Interpretation
        current_dd = dd.iloc[-1]*100
        risk_level = "High" if current_dd < -20 else "Moderate"
        st.markdown(f"""
        <div class="insight-box-warn">
            <b>📉 Risk Insight (風險解讀):</b><br>
            The asset is currently <b>{current_dd:.2f}%</b> below its historical peak.
            A drawdown of this magnitude indicates a <b>{risk_level}</b> risk profile. Investors should monitor if support levels hold.
            <br>(資產目前低於歷史峰值<b>{current_dd:.2f}%</b>。
            此等規模的回撤顯示<b>{risk_level}</b>風險輪廓。投資者應監測支撐位是否守住。)
        </div>
        """, unsafe_allow_html=True)
       
    with r1c2:
        st.markdown("**2. Macro Correlation (宏觀相關性)**")
        if macro_data:
            df_m = pd.DataFrame(macro_data); df_m[selected_label] = main_df['Close']
            corr = df_m.pct_change().corr()
            # Find highest correlation
            high_corr_factor = corr[selected_label].drop(selected_label).idxmax()
            high_corr_val = corr[selected_label].drop(selected_label).max()
           
            fig_hm = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            fig_hm.update_layout(height=350)
            st.plotly_chart(fig_hm, use_container_width=True)
           
            st.markdown(f"""
            <div class="insight-box">
            <b>🔗 Correlation Insight (相關性解讀):</b><br>
            The asset shows the strongest correlation (<b>{high_corr_val:.2f}</b>) with <b>{high_corr_factor}</b>.<br>
            • Positive (>0.5): Moves together (Risk of contagion).
            <br>(• 正相關 (>0.5)：同向移動 (傳染風險)。)<br>
            • Negative (<-0.5): Moves opposite (Good for hedging).
            <br>(• 負相關 (<-0.5)：反向移動 (適合對沖)。)
            </div>
            """, unsafe_allow_html=True)
        else: st.warning("Correlation requires historical data (Select 1M/3M/1Y). (相關性需歷史數據 (選擇1M/3M/1Y)。)")
    st.divider()
   
    # Row 2: Distribution & Seasonality
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**3. Return Distribution (收益分佈)**")
        fig_dist = plt.figure(figsize=(8, 4))
        sns.histplot(returns, kde=True, color="#3498db", stat="density")
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        st.pyplot(fig_dist)
       
        skew = returns.skew()
        st.markdown(f"""
        <div class="insight-box">
        <b>📊 Statistical Insight (統計分佈解讀):</b><br>
        Skewness is <b>{skew:.2f}</b>. { 'Negative skew suggests frequent small gains but rare extreme losses (Crash Risk).' if skew < 0 else 'Positive skew suggests frequent small losses but rare massive gains (Moonshot).' }
        <br>(偏度為<b>{skew:.2f}</b>。{ '負偏度暗示頻繁小幅獲利但罕見極端損失 (崩盤風險)。' if skew < 0 else '正偏度暗示頻繁小幅損失但罕見巨額獲利 (月球射擊)。' })
        </div>
        """, unsafe_allow_html=True)
    with r2c2:
        st.markdown("**4. Seasonality (月度日曆效應)**")
        if len(main_df) > 300:
            m_ret = main_df['Close'].resample('M').apply(lambda x: (x.iloc[-1]/x.iloc[0]-1)*100)
            m_ret.index = pd.to_datetime(m_ret.index)
            piv = pd.pivot_table(pd.DataFrame({'Y':m_ret.index.year, 'M':m_ret.index.month, 'V':m_ret.values}), values='V', index='Y', columns='M')
            fig_sea = plt.figure(figsize=(8, 4))
            sns.heatmap(piv, cmap='RdYlGn', center=0, annot=True, fmt=".1f", cbar=False)
            st.pyplot(fig_sea)
            st.markdown("""
            <div class="insight-box">
            <b>🗓️ Calendar Effect (日曆效應解讀):</b><br>
            Green cells indicate historically profitable months. Look for vertical patterns to identify specific months (e.g., "September Effect") that consistently underperform or outperform.
            <br>(綠色格子顯示歷史盈利月份。尋找垂直模式以識別特定月份 (如「九月效應」) 的持續低迷或超額表現。)
            </div>
            """, unsafe_allow_html=True)
        else: st.info("Requires >1 year of data for seasonality analysis. (需超過1年數據進行季節性分析。)")
# === Tab 3: Monte Carlo ===
with tabs[2]:
    st.subheader("🎲 Monte Carlo Stochastic Model (蒙特卡洛隨機模型)")
   
    col_sim, col_res = st.columns([3, 1])
   
    days_pred = 30
    sims = 100
    last_price = main_df['Close'].iloc[-1]
    log_ret = np.log(1 + main_df['Close'].pct_change())
    drift = log_ret.mean() - (0.5 * log_ret.var())
    sigma = log_ret.std()
   
    future_dates = [main_df.index[-1] + timedelta(days=x) for x in range(1, days_pred + 1)]
    paths = np.zeros((days_pred, sims))
    paths[0] = last_price
   
    for t in range(1, days_pred):
        shock = drift + sigma * np.random.normal(0, 1, sims)
        paths[t] = paths[t-1] * np.exp(shock)
   
    with col_sim:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=main_df.index[-60:], y=main_df['Close'].iloc[-60:], name='History (歷史)', line=dict(color='black')))
        for i in range(min(50, sims)):
            fig_mc.add_trace(go.Scatter(x=future_dates, y=paths[:, i], mode='lines', line=dict(color='rgba(46, 134, 222, 0.1)'), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=future_dates, y=paths.mean(axis=1), name='Mean Path (平均路徑)', line=dict(color='red', width=3)))
        fig_mc.update_layout(height=500, title=f"30-Day Forward Simulation (30天前瞻模擬)", margin=dict(t=30))
        st.plotly_chart(fig_mc, use_container_width=True)
       
    with col_res:
        st.markdown("### 📊 Forecast Stats (預測統計)")
        exp_price = paths.mean(axis=1)[-1]
        exp_ret_mc = (exp_price - last_price)/last_price*100
       
        st.metric("Expected Price (預期價格)", f"${exp_price:,.2f}")
        st.metric("Exp. Return (預期回報)", f"{exp_ret_mc:+.2f}%")
        st.markdown(f"""
        <div class="insight-box">
        <b>Logic (預測邏輯):</b><br>
        Based on Geometric Brownian Motion (GBM).<br>
        (基於幾何布朗運動 (GBM)。)<br>
        • <b>Drift:</b> {drift:.5f}<br>
        (• <b>漂移：</b> {drift:.5f})<br>
        • <b>Volatility:</b> {sigma:.5f}<br>
        (• <b>波動率：</b> {sigma:.5f})<br>
        The red line represents the statistical average of {sims} simulated future paths.
        <br>(紅線代表{sims}條模擬未來路徑的統計平均。)
        </div>
        """, unsafe_allow_html=True)
# --- Footer ---
st.markdown("---")
st.caption(f"**CISC7201 Final Project** | Data Points: {len(main_df)*len(main_df.columns):,} | Model: Monte Carlo (GBM)")
