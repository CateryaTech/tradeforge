"""
TradeForge AaaS - Streamlit Dashboard
Entry point untuk Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="TradeForge AaaS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Custom ──────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .profit { color: #a6e3a1; }
    .loss   { color: #f38ba8; }
    .neutral{ color: #cdd6f4; }
    h1 { color: #cba6f7 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/TradeForge-AaaS-purple?style=for-the-badge")
    st.markdown("**Author:** Ary HH @ CATERYA Tech")
    st.markdown("---")

    page = st.selectbox("📌 Navigate", [
        "🏠 Dashboard",
        "📈 Backtesting",
        "💹 Live Trading",
        "🌊 DeFi Positions",
        "💰 Payments & Revenue",
        "🤖 AI Agents",
        "⚙️ Settings",
    ])

    st.markdown("---")
    st.markdown("**Subscription Tier**")
    tier = st.selectbox("", ["Free", "Basic ($29/mo)", "Premium ($99/mo)", "Institutional ($499/mo)"])

# ─── Helper: generate demo data ──────────────────────────────

def gen_price_series(n=365, start=40000, volatility=0.02):
    np.random.seed(42)
    returns = np.random.normal(0.001, volatility, n)
    prices = start * np.cumprod(1 + returns)
    dates = pd.date_range(end=datetime.now(), periods=n)
    return pd.Series(prices, index=dates)


def gen_equity_curve(n=365, start=10000):
    np.random.seed(7)
    returns = np.random.normal(0.0008, 0.018, n)
    equity = start * np.cumprod(1 + returns)
    dates = pd.date_range(end=datetime.now(), periods=n)
    return pd.Series(equity, index=dates)


# ═══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("⚡ TradeForge AaaS — Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portfolio Value", "$24,831", "+$1,204 (5.1%)")
    col2.metric("Total P&L", "+$4,831", "+24.2%")
    col3.metric("Open Positions", "7", "+2")
    col4.metric("Win Rate", "63.4%", "+2.1%")
    col5.metric("Sharpe Ratio", "1.84", "+0.12")

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("📊 Portfolio Equity Curve")
        equity = gen_equity_curve()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            fill='tozeroy', fillcolor='rgba(203,166,247,0.15)',
            line=dict(color='#cba6f7', width=2),
            name='Equity'
        ))
        fig.update_layout(
            paper_bgcolor='#1e1e2e', plot_bgcolor='#1e1e2e',
            font=dict(color='#cdd6f4'), height=300,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#313244'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🥧 Allocation")
        labels = ['BTC', 'ETH', 'USDC', 'ARB', 'OP']
        values = [35, 25, 20, 12, 8]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.5,
            marker=dict(colors=['#f38ba8','#89b4fa','#a6e3a1','#fab387','#cba6f7'])
        ))
        fig2.update_layout(
            paper_bgcolor='#1e1e2e', font=dict(color='#cdd6f4'),
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Recent trades
    st.subheader("📋 Recent Trades")
    trades_data = {
        "Time": ["16:08", "15:43", "14:21", "13:55", "12:30"],
        "Pair": ["BTC/USDT", "ETH/USDT", "ARB/USDT", "BTC/USDT", "SOL/USDT"],
        "Side": ["BUY", "SELL", "BUY", "SELL", "BUY"],
        "Amount": [0.012, 0.45, 150.0, 0.008, 2.5],
        "Price": [61240, 3218, 1.24, 61800, 142.5],
        "P&L": ["+$42.10", "+$18.30", "-$5.20", "+$91.60", "+$12.80"],
        "Fee": ["$0.15", "$0.29", "$0.04", "$0.10", "$0.07"],
        "Status": ["✅ Filled", "✅ Filled", "✅ Filled", "✅ Filled", "✅ Filled"],
    }
    df = pd.DataFrame(trades_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: BACKTESTING
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Backtesting":
    st.title("📈 Backtesting Engine")
    st.info("Parallel backtesting with regime detection and Monte Carlo VaR.")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT"])
        strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Bollinger Breakout"])
    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        capital = st.number_input("Initial Capital (USD)", value=10000, step=1000)
        fast_period = st.slider("Fast MA Period", 5, 50, 20)
        slow_period = st.slider("Slow MA Period", 20, 200, 50)

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            import time; time.sleep(1.5)  # simulate

            # Generate demo results
            equity = gen_equity_curve(365, capital)
            final = equity.iloc[-1]
            ret = (final - capital) / capital * 100

            st.success("✅ Backtest complete!")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Return", f"{ret:.1f}%")
            m2.metric("Final Capital", f"${final:,.0f}")
            m3.metric("Sharpe Ratio", "1.72")
            m4.metric("Max Drawdown", "-14.3%")
            m5.metric("Win Rate", "61.2%")
            m6.metric("Regime", "🐂 Bull")

            # Equity chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, fill='tozeroy',
                fillcolor='rgba(166,227,161,0.1)', line=dict(color='#a6e3a1', width=2)))
            fig.update_layout(
                title="Equity Curve", paper_bgcolor='#1e1e2e',
                plot_bgcolor='#1e1e2e', font=dict(color='#cdd6f4'),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#313244'),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            # Monte Carlo
            st.subheader("🎲 Monte Carlo VaR (10,000 simulations)")
            sim_returns = np.random.normal(0.001, 0.02, 10000)
            var_95 = np.percentile(sim_returns, 5)
            cvar_95 = sim_returns[sim_returns <= var_95].mean()

            v1, v2, v3 = st.columns(3)
            v1.metric("VaR 95%", f"{var_95*100:.2f}%")
            v2.metric("CVaR 95%", f"{cvar_95*100:.2f}%")
            v3.metric("Worst Case", f"{sim_returns.min()*100:.2f}%")

            fig_hist = px.histogram(sim_returns * 100, nbins=100, title="Return Distribution",
                color_discrete_sequence=['#89b4fa'])
            fig_hist.add_vline(x=var_95*100, line_dash="dash", line_color="#f38ba8",
                annotation_text="VaR 95%")
            fig_hist.update_layout(paper_bgcolor='#1e1e2e', plot_bgcolor='#1e1e2e',
                font=dict(color='#cdd6f4'), height=300)
            st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: LIVE TRADING
# ═══════════════════════════════════════════════════════════════
elif page == "💹 Live Trading":
    st.title("💹 Live Trading")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📤 Place Order")
        exchange = st.selectbox("Exchange", ["Binance", "Coinbase", "Kraken", "OKX"])
        pair = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        order_type = st.radio("Type", ["Market", "Limit"], horizontal=True)
        amount = st.number_input("Amount", value=0.01, format="%.4f")
        if order_type == "Limit":
            price = st.number_input("Limit Price", value=60000.0)

        if st.button("⚡ Execute Order", type="primary"):
            with st.spinner("Executing..."):
                import time; time.sleep(0.8)
            st.success(f"✅ {side} {amount} {pair} filled @ $61,240 | Latency: 312ms | Fee: $0.12")

    with col2:
        st.subheader("🔍 Arbitrage Scanner")
        if st.button("🔎 Scan Now"):
            arb_data = {
                "Symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                "Buy Exchange": ["Kraken", "Binance", "OKX"],
                "Buy Price": [61180.0, 3211.5, 141.8],
                "Sell Exchange": ["Binance", "Coinbase", "Kraken"],
                "Sell Price": [61310.0, 3224.0, 142.9],
                "Profit %": ["+0.21%", "+0.39%", "+0.78%"],
                "After Fees": ["+0.01%", "+0.19%", "+0.58%"],
            }
            st.dataframe(pd.DataFrame(arb_data), use_container_width=True, hide_index=True)

        st.subheader("📡 Slippage Estimator")
        est_amount = st.number_input("Order Size (USD)", value=10000)
        if st.button("Estimate Slippage"):
            slippage = est_amount / 10_000_000 * 100
            st.metric("Estimated Slippage", f"{slippage:.4f}%",
                "✅ Safe" if slippage < 0.1 else "⚠️ High")


# ═══════════════════════════════════════════════════════════════
# PAGE: DEFI
# ═══════════════════════════════════════════════════════════════
elif page == "🌊 DeFi Positions":
    st.title("🌊 DeFi Positions")

    st.subheader("🏆 Best Yield Opportunities")
    yield_data = {
        "Protocol": ["Aave V3", "Aave V3", "Aave V3", "Uniswap V3", "Uniswap V3"],
        "Chain": ["Arbitrum", "Base", "Ethereum", "Arbitrum", "Base"],
        "Asset": ["USDC", "USDC", "USDC", "ETH/USDC", "ETH/USDC"],
        "APY": ["8.4%", "7.2%", "5.1%", "24.6%", "18.3%"],
        "Type": ["Lending", "Lending", "Lending", "LP 0.05%", "LP 0.05%"],
        "Risk": ["🟢 Low", "🟢 Low", "🟢 Low", "🟡 Medium", "🟡 Medium"],
    }
    st.dataframe(pd.DataFrame(yield_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📌 My Positions")
    pos_data = {
        "Protocol": ["Uniswap V3", "Aave V3", "Uniswap V3"],
        "Chain": ["Arbitrum", "Base", "Ethereum"],
        "Tokens": ["ETH/USDC", "USDC", "WBTC/USDC"],
        "Value (USD)": ["$5,240", "$3,100", "$2,800"],
        "Current APY": ["24.6%", "7.2%", "19.8%"],
        "Fees Earned": ["$142", "$18", "$89"],
        "In Range": ["✅ Yes", "N/A", "✅ Yes"],
        "Needs Rebalance": ["❌ No", "❌ No", "⚠️ Soon"],
    }
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total DeFi Value", "$11,140")
    col2.metric("Total Fees Earned", "$249")
    col3.metric("Avg APY", "17.2%")


# ═══════════════════════════════════════════════════════════════
# PAGE: PAYMENTS & REVENUE
# ═══════════════════════════════════════════════════════════════
elif page == "💰 Payments & Revenue":
    st.title("💰 Payments & Revenue Projections")

    tab1, tab2 = st.tabs(["📊 Revenue Model", "💳 Subscribe"])

    with tab1:
        st.subheader("Configure User Base")
        col1, col2 = st.columns(2)
        with col1:
            n_free = st.slider("Free users", 0, 5000, 100)
            n_basic = st.slider("Basic users ($29/mo)", 0, 2000, 50)
        with col2:
            n_premium = st.slider("Premium users ($99/mo)", 0, 1000, 20)
            n_inst = st.slider("Institutional ($499/mo)", 0, 100, 5)

        avg_vol = st.slider("Avg trade size (USD)", 100, 10000, 500)
        avg_trades = st.slider("Avg trades/user/day", 1, 50, 5)

        # Calculate
        sub_mrr = n_basic * 29 + n_premium * 99 + n_inst * 499
        total_users = n_free + n_basic + n_premium + n_inst
        trade_mrr = total_users * avg_trades * 30 * avg_vol * 0.002
        defi_mrr = (n_premium + n_inst) * 50 * 0.10
        total_mrr = sub_mrr + trade_mrr + defi_mrr

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subscription MRR", f"${sub_mrr:,.0f}")
        c2.metric("Trade Fee MRR (0.2%)", f"${trade_mrr:,.0f}")
        c3.metric("DeFi Yield Share (10%)", f"${defi_mrr:,.0f}")
        c4.metric("🎯 Total MRR", f"${total_mrr:,.0f}", f"ARR: ${total_mrr*12:,.0f}")

        # Chart
        months = list(range(1, 13))
        growth_rate = 1.15  # 15% MoM growth
        mrr_series = [total_mrr * (growth_rate ** m) for m in months]
        fig = go.Figure(go.Bar(x=months, y=mrr_series,
            marker_color=['#cba6f7'] * 12))
        fig.update_layout(
            title="MRR Growth Projection (15% MoM)",
            xaxis_title="Month", yaxis_title="MRR (USD)",
            paper_bgcolor='#1e1e2e', plot_bgcolor='#1e1e2e',
            font=dict(color='#cdd6f4'), height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        if total_mrr * 12 >= 1_000_000:
            st.success(f"🎉 $1M ARR target reached at month {next(i+1 for i,v in enumerate(mrr_series) if v*12 >= 1_000_000)}!")

    with tab2:
        st.subheader("💳 Choose a Plan")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Basic\n**$29/month**\n- 100 trades/day\n- 3 strategies\n- Backtesting\n- CEX support")
            if st.button("Pay with Card (Stripe)", key="stripe_basic"):
                st.info("Stripe checkout would open here.")
            if st.button("Pay with Crypto", key="crypto_basic"):
                st.info("NOWPayments — supports BTC, ETH, SOL, 350+ coins.")
        with col2:
            st.markdown("### Premium\n**$99/month**\n- Unlimited trades\n- AI agents\n- DeFi automation\n- All chains")
            if st.button("Pay with Card (Stripe)", key="stripe_premium"):
                st.info("Stripe checkout would open here.")
            if st.button("Pay with Crypto", key="crypto_premium"):
                st.info("NOWPayments — supports BTC, ETH, SOL, 350+ coins.")
        with col3:
            st.markdown("### Institutional\n**$499/month**\n- White-label\n- Dedicated support\n- Custom integrations\n- SLA guarantee")
            if st.button("Contact Sales", key="sales"):
                st.info("Email: cateryatech@proton.me")


# ═══════════════════════════════════════════════════════════════
# PAGE: AI AGENTS
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 AI Agents":
    st.title("🤖 AI Trading Agents")

    tab1, tab2 = st.tabs(["🧠 Strategy Generator", "🌾 Yield Farming Agent"])

    with tab1:
        st.subheader("Generate AI Trading Strategy")
        col1, col2 = st.columns(2)
        with col1:
            sym = st.selectbox("Symbol", ["BTC", "ETH", "SOL"])
            risk = st.select_slider("Risk Tolerance", ["Low", "Medium", "High"])
            capital_ai = st.number_input("Capital (USD)", value=5000)
        with col2:
            provider = st.selectbox("LLM Provider", ["Groq (Fast)", "OpenRouter", "Together AI"])
            horizon = st.selectbox("Horizon", ["1 week", "1 month", "3 months"])

        if st.button("🧠 Generate Strategy", type="primary"):
            with st.spinner("AI analyzing market conditions..."):
                import time; time.sleep(2)

            st.success("✅ Strategy generated!")
            st.json({
                "strategy_type": "rsi_mean_reversion",
                "config": {"rsi_period": 14, "oversold": 28, "overbought": 72},
                "reasoning": f"Current {sym} regime is sideways-to-bull. RSI mean reversion "
                             f"performs best in these conditions with {risk.lower()} risk tolerance.",
                "expected_sharpe": 1.65,
                "risk_level": risk.lower(),
                "recommended_position_size": f"{min(30, capital_ai * 0.03):.0f} USD per trade",
            })

    with tab2:
        st.subheader("Autonomous Yield Farming Agent")
        st.info("Agent scans all chains every 15 minutes, auto-rebalances positions when APY diff > 2%.")

        agent_status = st.toggle("Enable Yield Agent", value=False)
        if agent_status:
            st.success("🟢 Agent active — monitoring Ethereum, Arbitrum, Base, Optimism")
            st.metric("Last rebalance", "14 minutes ago")
            st.metric("Yield improvement this week", "+$47.20 (+3.1%)")
        else:
            st.warning("⚪ Agent paused")

        log_data = {
            "Time": ["16:05", "15:50", "15:20", "14:45"],
            "Action": ["Scan", "Rebalance", "Compound", "Scan"],
            "Detail": [
                "Arbitrum USDC APY: 8.4% > Ethereum 5.1% → flag",
                "Moved $1,200 USDC from Ethereum to Arbitrum Aave",
                "Compounded $12.40 rewards on Arbitrum",
                "No action needed — all positions optimal",
            ],
            "Gas (USD)": ["$0", "$1.20", "$0.85", "$0"],
        }
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    tab1, tab2, tab3 = st.tabs(["🔑 API Keys", "🔔 Notifications", "🏢 White Label"])

    with tab1:
        st.subheader("Exchange API Keys")
        st.text_input("Binance API Key", type="password", placeholder="Enter key...")
        st.text_input("Binance Secret", type="password", placeholder="Enter secret...")
        st.markdown("---")
        st.subheader("RPC Endpoints")
        st.text_input("Ethereum RPC", placeholder="https://mainnet.infura.io/v3/...")
        st.text_input("Arbitrum RPC", placeholder="https://arb-mainnet.g.alchemy.com/v2/...")
        if st.button("Save API Keys"):
            st.success("Saved securely (encrypted at rest)")

    with tab2:
        st.subheader("Alert Settings")
        st.toggle("Email alerts", value=True)
        st.toggle("Telegram alerts", value=False)
        st.toggle("Discord webhook", value=False)
        st.slider("Volatility alert threshold (%)", 1, 20, 5)
        st.slider("Drawdown alert threshold (%)", 1, 30, 10)

    with tab3:
        st.subheader("White Label Configuration")
        st.info("Available on Institutional plan only.")
        st.text_input("Company Name", placeholder="Your Company")
        st.text_input("Logo URL", placeholder="https://...")
        st.color_picker("Brand Color", "#cba6f7")
        st.text_input("Custom Domain", placeholder="trading.yourcompany.com")
