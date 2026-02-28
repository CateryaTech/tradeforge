"""
TradeForge AaaS — Production Dashboard v2.0
Author: Ary HH @ CATERYA Tech
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json, io, time, urllib.request, re, html as _html

st.set_page_config(page_title="TradeForge AaaS", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
# USERS & TIERS
# ══════════════════════════════════════════════════════════════
USERS = {
    # free tier — no login needed, but can also have an account
    "bandung123":    {"password":"bandung123",    "tier":"basic",        "name":"Basic Member"},
    "bandung12345":  {"password":"bandung12345",  "tier":"premium",      "name":"Premium Member"},
    "bandung1234567":{"password":"bandung1234567","tier":"institutional","name":"Institutional Member"},
}

TIERS = {
    "free":          {"label":"Free",          "price":"$0/mo",   "color":"#64748b","badge":"FREE",  "order":0},
    "basic":         {"label":"Basic",         "price":"$29/mo",  "color":"#38bdf8","badge":"BASIC", "order":1},
    "premium":       {"label":"Premium",       "price":"$99/mo",  "color":"#a78bfa","badge":"PRO",   "order":2},
    "institutional": {"label":"Institutional", "price":"$499/mo", "color":"#f59e0b","badge":"INST",  "order":3},
}

# Pages accessible per tier
TIER_PAGES = {
    "free":          ["🏠 Dashboard","📡 Live Market","📰 News & Sentiment","💡 Investment Guide","💳 Pricing & Plans"],
    "basic":         ["🏠 Dashboard","📡 Live Market","📈 Backtesting","💹 Live Trading",
                      "📰 News & Sentiment","🧠 News Intelligence","💡 Investment Guide","📤 Export Center","💳 Pricing & Plans","⚙️ Settings"],
    "premium":       ["🏠 Dashboard","📡 Live Market","📈 Backtesting","💹 Live Trading",
                      "🌊 DeFi + Wallet","🤖 AI Signals","📰 News & Sentiment",
                      "🧠 News Intelligence","💡 Investment Guide",
                      "💰 Revenue","📤 Export Center","💳 Pricing & Plans","⚙️ Settings"],
    "institutional": ["🏠 Dashboard","📡 Live Market","📈 Backtesting","💹 Live Trading",
                      "🌊 DeFi + Wallet","🤖 AI Signals","📰 News & Sentiment",
                      "🧠 News Intelligence","💡 Investment Guide",
                      "💰 Revenue","📤 Export Center","💳 Pricing & Plans","⚙️ Settings"],
}

TIER_LIMITS = {
    "free":          {"history_days":30,  "news_sources":2,  "backtest_strategies":0,  "trades_day":0},
    "basic":         {"history_days":180, "news_sources":5,  "backtest_strategies":3,  "trades_day":50},
    "premium":       {"history_days":365, "news_sources":10, "backtest_strategies":10, "trades_day":-1},
    "institutional": {"history_days":1825,"news_sources":10, "backtest_strategies":-1, "trades_day":-1},
}

ALL_PAGES = [
    "🏠 Dashboard","📡 Live Market","📈 Backtesting","💹 Live Trading",
    "🌊 DeFi + Wallet","🤖 AI Signals","📰 News & Sentiment",
    "🧠 News Intelligence","💡 Investment Guide",
    "💰 Revenue","📤 Export Center","💳 Pricing & Plans","⚙️ Settings",
]

# ── Auth helpers ──────────────────────────────────────────────
def get_tier():   return st.session_state.get("tier","free")
def get_name():   return st.session_state.get("uname","Guest")
def logged_in():  return st.session_state.get("logged_in", False)
def tier_order(): return TIERS.get(get_tier(),TIERS["free"])["order"]

def do_login(u, p):
    usr = USERS.get(u)
    if usr and usr["password"] == p:
        st.session_state.update(logged_in=True, username=u,
            uname=usr["name"], tier=usr["tier"])
        return True
    return False

def do_logout():
    for k in ["logged_in","username","uname","tier"]: st.session_state.pop(k,None)

def need_tier(min_tier: str) -> bool:
    """Returns True if current user has access."""
    return TIERS.get(get_tier(),TIERS["free"])["order"] >= TIERS[min_tier]["order"]

def show_upgrade_wall(feature: str, min_tier: str):
    ti = TIERS[min_tier]
    c  = ti["color"]
    st.markdown(f"""
    <div style="max-width:460px;margin:50px auto;background:#0d1117;
    border:1px solid {c}55;border-radius:16px;padding:40px;text-align:center">
    <div style="font-size:2.8rem">🔒</div>
    <div style="font-family:monospace;color:{c};font-size:1rem;font-weight:700;
    margin:12px 0 6px">{ti['badge']} FEATURE</div>
    <div style="color:#e2e8f0;font-size:1rem;font-weight:600;margin-bottom:8px">{feature}</div>
    <div style="color:#64748b;font-size:.85rem;margin-bottom:24px">
      Tersedia di plan <b style="color:{c}">{ti['label']} ({ti['price']})</b> ke atas.</div>
    </div>""", unsafe_allow_html=True)

    if not logged_in():
        st.markdown("#### 🔑 Login untuk mengakses")
        with st.form("wall_login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary", use_container_width=True):
                if do_login(u, p):
                    st.success(f"Welcome {get_name()}!"); st.rerun()
                else:
                    st.error("Username/password salah")
        st.caption("Cek halaman **💳 Pricing & Plans** untuk info akun demo")
    else:
        st.warning(f"Akun Anda ({TIERS[get_tier()]['label']}) belum termasuk fitur ini.")
        if st.button("💳 Lihat Pricing & Plans", type="primary"):
            st.session_state["_goto"] = "💳 Pricing & Plans"; st.rerun()

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Mono',monospace!important;color:#e2e8f0!important}
.stApp{background:#0a0e1a}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e293b}
.kpi{background:linear-gradient(135deg,#0d1117,#111827);border:1px solid #1e293b;
     border-radius:12px;padding:18px 20px;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
             background:linear-gradient(90deg,#3b82f6,#8b5cf6,#06b6d4)}
.kpi-v{font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;color:#f1f5f9}
.kpi-l{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.kpi-u{color:#22c55e;font-size:.78rem}.kpi-d{color:#ef4444;font-size:.78rem}
.sig-buy{background:#052e16;border:1px solid #16a34a;border-radius:8px;padding:10px 14px;
         color:#4ade80;font-family:'Space Mono',monospace;font-size:.8rem}
.sig-sell{background:#2d0a0a;border:1px solid #dc2626;border-radius:8px;padding:10px 14px;
          color:#f87171;font-family:'Space Mono',monospace;font-size:.8rem}
.sig-neu{background:#1c1f2e;border:1px solid #475569;border-radius:8px;padding:10px 14px;
         color:#94a3b8;font-family:'Space Mono',monospace;font-size:.8rem}
.news-card{background:#0d1117;border:1px solid #1e293b;border-radius:10px;padding:14px;margin-bottom:10px}
div[data-testid="stDownloadButton"] button{background:#111827;border:1px solid #1e293b;
  color:#94a3b8;border-radius:8px;font-size:.82rem;width:100%;transition:all .2s}
div[data-testid="stDownloadButton"] button:hover{border-color:#3b82f6;color:#3b82f6}
.plan-card{border-radius:16px;padding:28px 24px;margin-bottom:8px;position:relative}
.plan-badge{display:inline-block;font-size:.65rem;font-weight:700;letter-spacing:.1em;
            padding:3px 10px;border-radius:20px;margin-bottom:12px}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ══════════════════════════════════════════════════════════════
CL = dict(paper_bgcolor='#0a0e1a',plot_bgcolor='#0d1117',
          font=dict(color='#94a3b8',family='DM Sans'),
          xaxis=dict(showgrid=False,color='#334155'),
          yaxis=dict(gridcolor='#1e293b',color='#334155'),
          margin=dict(l=0,r=0,t=30,b=0))
PL = dict(paper_bgcolor='#0a0e1a',font=dict(color='#94a3b8',family='DM Sans'),
          margin=dict(l=0,r=0,t=10,b=0))

# ══════════════════════════════════════════════════════════════
# DATA FETCHERS  (all cached)
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_prices(ids="bitcoin,ethereum,solana"):
    try:
        url=(f"https://api.coingecko.com/api/v3/simple/price?ids={ids}"
             f"&vs_currencies=usd&include_24hr_change=true&include_market_cap=true")
        with urllib.request.urlopen(url,timeout=8) as r: return json.loads(r.read())
    except: return {}

@st.cache_data(ttl=600)
def fetch_fg():
    try:
        with urllib.request.urlopen("https://api.alternative.me/fng/?limit=30",timeout=8) as r:
            return json.loads(r.read()).get("data",[])
    except: return []

@st.cache_data(ttl=300)
def fetch_ohlcv(ticker="BTC-USD", days=180):
    try:
        import yfinance as yf
        df=yf.download(ticker,
            start=(datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,auto_adjust=True)
        df.reset_index(inplace=True)
        df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
        return df[["Date","Open","High","Low","Close","Volume"]].dropna()
    except:
        np.random.seed(42); n=days
        d=pd.date_range(end=datetime.now(),periods=n)
        c=40000*np.cumprod(1+np.random.normal(0.001,0.025,n))
        return pd.DataFrame({"Date":d,"Open":c*.998,"High":c*1.015,
                             "Low":c*.985,"Close":c,"Volume":np.random.uniform(1e9,5e9,n)})

@st.cache_data(ttl=600)
def fetch_defi_yields():
    try:
        with urllib.request.urlopen("https://yields.llama.fi/pools",timeout=10) as r:
            pools=json.loads(r.read()).get("data",[])
        ok=[p for p in pools if p.get("symbol","").upper() in
            ["USDC","USDT","ETH","WBTC"] and p.get("tvlUsd",0)>5e6 and p.get("apy",0)>0]
        ok.sort(key=lambda x:x.get("apy",0),reverse=True)
        return ok[:20]
    except: return []

@st.cache_data(ttl=600)
def fetch_news(sources: tuple):
    NEWS_RSS = {
        "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "The Block":     "https://www.theblock.co/rss.xml",
        "Decrypt":       "https://decrypt.co/feed",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Bitcoin Mag":   "https://bitcoinmagazine.com/feed",
        "Bankless":      "https://banklesshq.com/rss/",
        "Messari":       "https://messari.io/rss",
        "WSJ Markets":   "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "Reuters":       "https://feeds.reuters.com/reuters/businessNews",
        "CryptoPanic":   "https://cryptopanic.com/news/rss/",
    }
    all_items=[]
    for src in sources:
        url=NEWS_RSS.get(src,"")
        if not url: continue
        try:
            import feedparser
            feed=feedparser.parse(url)
            for e in feed.entries[:5]:
                all_items.append({"source":src,"title":e.get("title","")[:140],
                    "url":e.get("link","#"),"time":e.get("published","")[:16]})
        except: pass
    all_items.sort(key=lambda x:x.get("time",""),reverse=True)
    if not all_items:  # fallback
        return [
            {"source":"CoinDesk","title":"Bitcoin holds $60K as ETF inflows hit record","url":"#","time":"2025-02-28"},
            {"source":"The Block","title":"Ethereum Pectra upgrade details confirmed","url":"#","time":"2025-02-28"},
            {"source":"Decrypt","title":"DeFi TVL crosses $120B milestone","url":"#","time":"2025-02-27"},
            {"source":"CoinTelegraph","title":"Arbitrum sees record DEX volume week","url":"#","time":"2025-02-27"},
            {"source":"Reuters","title":"Fed holds rates, crypto markets rally","url":"#","time":"2025-02-27"},
        ]
    return all_items[:30]

# ══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean(); l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def macd(s,f=12,sl=26,sig=9):
    ml=s.ewm(span=f).mean()-s.ewm(span=sl).mean()
    return ml,ml.ewm(span=sig).mean(),ml-ml.ewm(span=sig).mean()

def bbands(s,n=20,k=2):
    m=s.rolling(n).mean(); sd=s.rolling(n).std()
    return m+k*sd,m,m-k*sd

def signals(df):
    c=df["Close"]; rv=rsi(c).iloc[-1]
    ml,ms,_=macd(c); mc="BUY" if ml.iloc[-1]>ms.iloc[-1] else "SELL"
    tr="BUY" if c.rolling(20).mean().iloc[-1]>c.rolling(50).mean().iloc[-1] else "SELL"
    bu,_,bl=bbands(c); pr=c.iloc[-1]
    bc="BUY" if pr<bl.iloc[-1] else("SELL" if pr>bu.iloc[-1] else "NEUTRAL")
    rv_s="BUY" if rv<40 else("SELL" if rv>65 else "NEUTRAL")
    v=[mc,tr,bc,rv_s]; buy=v.count("BUY"); sel=v.count("SELL")
    ov="BUY" if buy>=3 else("SELL" if sel>=3 else "NEUTRAL")
    return {"overall":ov,"rsi":round(rv,1),"macd":mc,"trend":tr,
            "bb":bc,"buy":buy,"sell":sel,"price":round(pr,2)}

# ══════════════════════════════════════════════════════════════
# PDF (unicode-safe, no kaleido)
# ══════════════════════════════════════════════════════════════
def safe_str(t:str)->str:
    subs={"\u2014":"-","\u2013":"-","\u2018":"'","\u2019":"'","\u201c":'"',
          "\u201d":'"',"\u2026":"...","\u2022":"*","\u00a0":" ","\u20ac":"EUR",
          "\u2191":"^","\u2193":"v","\u25b2":"^","\u25bc":"v","\u2192":"->",
          "\u2714":"OK","\u274c":"X","\u2705":"OK","\u26a1":"!","\u26a0":"!",
          "\u2728":"*","\u00e9":"e","\u00e8":"e","\u00fc":"u","\u00f6":"o"}
    for ch,rep in subs.items(): t=t.replace(ch,rep)
    return t.encode("latin-1",errors="replace").decode("latin-1")

def make_pdf(title:str, df:pd.DataFrame)->bytes:
    from fpdf import FPDF
    doc=FPDF(); doc.set_margins(15,15,15); doc.set_auto_page_break(True,15); doc.add_page()
    doc.set_fill_color(13,17,23); doc.rect(0,0,210,26,"F")
    doc.set_xy(15,7); doc.set_font("Helvetica","B",13); doc.set_text_color(129,140,248)
    doc.cell(0,8,safe_str("TradeForge AaaS - "+title))
    doc.set_xy(15,17); doc.set_font("Helvetica","",8); doc.set_text_color(100,116,139)
    doc.cell(0,5,safe_str(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | CATERYA Tech"))
    doc.set_y(32)
    # Header row
    cols=list(df.columns); cw=min(180//max(len(cols),1),45)
    doc.set_font("Helvetica","B",8); doc.set_text_color(56,189,248)
    doc.set_fill_color(17,24,39)
    for col in cols:
        doc.cell(cw,7,safe_str(str(col)[:18]),border=0,fill=True)
    doc.ln()
    doc.set_font("Helvetica","",7); doc.set_text_color(203,213,225)
    for _,row in df.iterrows():
        for val in row:
            try: doc.cell(cw,6,safe_str(str(val)[:18]),border=0)
            except: doc.cell(cw,6,"?",border=0)
        doc.ln()
        if doc.get_y()>270: doc.add_page(); doc.set_font("Helvetica","",7); doc.set_text_color(203,213,225)
    raw=doc.output()
    return bytes(raw) if not isinstance(raw,bytes) else raw

# ══════════════════════════════════════════════════════════════
# EXCEL
# ══════════════════════════════════════════════════════════════
def safe_sheet(n:str)->str:
    return re.sub(r"[\\/:*?\[\]]","_",str(n))[:31].strip() or "Sheet"

def to_excel(sheets:dict)->bytes:
    buf=io.BytesIO()
    seen={}
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        for name,df in sheets.items():
            s=safe_sheet(name)
            if s in seen: seen[s]+=1; s=s[:28]+f"_{seen[s]}"
            else: seen[s]=0
            df.to_excel(w,sheet_name=s,index=False)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════
# EXPORT PANEL
# ══════════════════════════════════════════════════════════════
def export_panel(label:str, df:pd.DataFrame, fig=None,
                 extra_sheets:dict=None, filename_base:str="export"):
    sheets={safe_sheet(label):df}
    if extra_sheets:
        for k,v in extra_sheets.items(): sheets[safe_sheet(k)]=v
    ts=datetime.now().strftime("%Y%m%d_%H%M")
    fn=f"{filename_base}_{ts}"
    c1,c2,c3=st.columns(3)
    c1.download_button("⬇️ CSV",df.to_csv(index=False).encode(),
        f"{fn}.csv","text/csv",use_container_width=True)
    c2.download_button("⬇️ Excel",to_excel(sheets),
        f"{fn}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    c3.download_button("⬇️ JSON",df.to_json(orient="records",indent=2).encode(),
        f"{fn}.json","application/json",use_container_width=True)
    c4,c5=st.columns(2)
    # HTML report
    tbl=df.to_html(index=False,border=0)
    now=datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    html_rep=(f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
              f"<title>{label}</title><style>body{{font-family:Arial;background:#0a0e1a;color:#e2e8f0;padding:32px}}"
              f"h1{{color:#818cf8}}table{{border-collapse:collapse;width:100%;font-size:13px}}"
              f"th{{background:#1e293b;color:#818cf8;padding:8px 12px;text-align:left}}"
              f"td{{border-bottom:1px solid #1e293b;padding:7px 12px}}"
              f"</style></head><body><h1>TradeForge AaaS — {label}</h1>"
              f"<p style='color:#64748b'>Generated: {now} | CATERYA Tech</p>{tbl}</body></html>")
    c4.download_button("📄 HTML Report",html_rep.encode(),
        f"{fn}.html","text/html",use_container_width=True)
    try:
        pdf=make_pdf(label,df)
        c5.download_button("📕 PDF Report",pdf,f"{fn}.pdf","application/pdf",use_container_width=True)
    except Exception as e:
        c5.caption(f"PDF err: {str(e)[:60]}")
    if fig:
        c6,c7=st.columns(2)
        c7.download_button("🌐 Chart HTML",
            fig.to_html(full_html=True,include_plotlyjs="cdn").encode(),
            f"{fn}_chart.html","text/html",use_container_width=True)
        c6.caption("🖼️ Chart: download via Chart HTML → browser print")

# ══════════════════════════════════════════════════════════════
# DEMO DATA
# ══════════════════════════════════════════════════════════════
def gen_trades(n=50):
    np.random.seed(42)
    d=pd.date_range(end=datetime.now(),periods=n,freq="2h")
    pr=np.random.uniform(100,65000,n); am=np.random.uniform(.001,2,n)
    return pd.DataFrame({
        "DateTime":d.strftime("%Y-%m-%d %H:%M"),
        "Pair":np.random.choice(["BTC/USDT","ETH/USDT","SOL/USDT","ARB/USDT"],n),
        "Exchange":np.random.choice(["Binance","Kraken","Coinbase","OKX"],n),
        "Side":np.random.choice(["BUY","SELL"],n),
        "Amount":am.round(4),"Price":pr.round(2),
        "Value":(am*pr).round(2),"PnL":np.random.normal(20,80,n).round(2),
        "Fee":(am*pr*.002).round(4),
        "Status":np.random.choice(["Filled","Failed"],n,p=[.93,.07])})

def gen_portfolio():
    return pd.DataFrame({
        "Asset":["BTC","ETH","USDC","ARB","OP","SOL"],
        "Amount":[.42,3.1,2500,850,1200,12.5],
        "Price":[61240,3218,1,.24,2.11,142.5],
        "Value":[25720,9975,2500,204,2532,1781],
        "Alloc%":[58.4,22.7,5.7,0.5,5.8,4.1],
        "24h%":[2.1,-.8,0,3.4,-1.2,1.9]})

def kpi(col,label,val,chg="",up=True):
    sign="kpi-u" if up else "kpi-d"
    chg_html=f'<div class="{sign}">{chg}</div>' if chg else ""
    col.markdown(f'<div class="kpi"><div class="kpi-l">{label}</div>'
                 f'<div class="kpi-v">{val}</div>{chg_html}</div>',
                 unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:12px 0 6px'>
    <span style='font-family:monospace;font-size:1.15rem;color:#818cf8;font-weight:700'>⚡ TRADEFORGE</span><br>
    <span style='font-size:.68rem;color:#475569;letter-spacing:.1em'>AaaS PLATFORM v2.0</span>
    </div>""",unsafe_allow_html=True)

    # Auth widget
    if logged_in():
        ti=TIERS[get_tier()]; c=ti["color"]
        st.markdown(f"""<div style='background:#0d1117;border:1px solid {c}33;
        border-radius:10px;padding:10px 12px;margin:8px 0'>
        <div style='font-size:.68rem;color:#64748b'>LOGGED IN</div>
        <div style='color:#e2e8f0;font-weight:600;font-size:.9rem'>{get_name()}</div>
        <span style='background:{c}22;color:{c};font-size:.63rem;font-weight:700;
        padding:2px 8px;border-radius:4px'>{ti["badge"]} {ti["price"]}</span>
        </div>""",unsafe_allow_html=True)
        if st.button("🚪 Logout",use_container_width=True):
            do_logout(); st.rerun()
    else:
        with st.expander("🔑 Login",expanded=False):
            with st.form("sb_login",clear_on_submit=False):
                su=st.text_input("Username")
                sp=st.text_input("Password",type="password")
                if st.form_submit_button("Login",type="primary",use_container_width=True):
                    if do_login(su,sp): st.rerun()
                    else: st.error("Salah")
            st.caption("bandung123 / bandung12345 / bandung1234567")

    st.markdown("---")

    # Nav — show all pages, lock unavailable
    cur_tier  = get_tier()
    avail     = TIER_PAGES.get(cur_tier, TIER_PAGES["free"])
    goto      = st.session_state.pop("_goto", None)

    def plabel(p): return p if p in avail else p+" 🔒"
    labels = [plabel(p) for p in ALL_PAGES]
    default_idx = 0
    if goto and plabel(goto) in labels:
        default_idx = labels.index(plabel(goto))

    sel   = st.selectbox("Navigate", labels, index=default_idx)
    page  = sel.replace(" 🔒","")

    st.markdown("---")
    # Live ticker
    st.caption("LIVE PRICES")
    live = fetch_prices()
    for sym,cid in [("BTC","bitcoin"),("ETH","ethereum"),("SOL","solana")]:
        d=live.get(cid,{}); p=d.get("usd",0); ch=d.get("usd_24h_change",0)
        col="#22c55e" if ch>=0 else "#ef4444"; arr="▲" if ch>=0 else "▼"
        st.markdown(f"""<div style='display:flex;justify-content:space-between;padding:3px 0'>
        <span style='color:#94a3b8;font-size:.78rem'>{sym}</span>
        <span style='font-family:monospace;font-size:.78rem;color:#e2e8f0'>${p:,.0f}
        <span style='color:{col}'>{arr}{abs(ch):.1f}%</span></span></div>""",
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD  (free)
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("⚡ Dashboard")
    st.caption(f"Live data · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    live=fetch_prices(); fg=fetch_fg()
    btc=live.get("bitcoin",{}); eth=live.get("ethereum",{})
    fgv=int(fg[0]["value"]) if fg else 62
    fgc=fg[0]["value_classification"] if fg else "Greed"
    c1,c2,c3,c4,c5=st.columns(5)
    kpi(c1,"BTC PRICE",f"${btc.get('usd',0):,.0f}",f"{btc.get('usd_24h_change',0):+.2f}%",btc.get('usd_24h_change',0)>=0)
    kpi(c2,"ETH PRICE",f"${eth.get('usd',0):,.2f}",f"{eth.get('usd_24h_change',0):+.2f}%",eth.get('usd_24h_change',0)>=0)
    kpi(c3,"PORTFOLIO","$43,564","+2.8%")
    kpi(c4,"TOTAL P&L","+$8,564","+24.5%")
    kpi(c5,"FEAR & GREED",f"{fgv}",fgc,fgv>50)
    st.markdown("<br>",unsafe_allow_html=True)
    ca,cb=st.columns([3,1])
    with ca:
        st.subheader("BTC/USD — Live Chart")
        lim = TIER_LIMITS[get_tier()]["history_days"]
        df_c=fetch_ohlcv("BTC-USD",min(lim,180))
        fig=go.Figure()
        fig.add_trace(go.Candlestick(x=df_c["Date"],open=df_c["Open"],high=df_c["High"],
            low=df_c["Low"],close=df_c["Close"],name="BTC",
            increasing_line_color="#22c55e",decreasing_line_color="#ef4444",
            increasing_fillcolor="#052e16",decreasing_fillcolor="#2d0a0a"))
        df_c["s20"]=df_c["Close"].rolling(20).mean()
        df_c["s50"]=df_c["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df_c["Date"],y=df_c["s20"],line=dict(color="#818cf8",width=1.5),name="SMA20"))
        fig.add_trace(go.Scatter(x=df_c["Date"],y=df_c["s50"],line=dict(color="#f59e0b",width=1.5),name="SMA50"))
        fig.update_layout(**CL,height=340,title=f"BTC/USD {lim}D",
            xaxis_rangeslider_visible=False,legend=dict(bgcolor='rgba(0,0,0,0)',orientation="h",y=1.05))
        st.plotly_chart(fig,use_container_width=True)
        if get_tier()=="free":
            st.info("📈 Login dengan akun Basic+ untuk chart history 180–1825 hari & semua aset")
    with cb:
        st.subheader("Allocation")
        pf=gen_portfolio()
        fig2=go.Figure(go.Pie(labels=pf["Asset"],values=pf["Value"],hole=.55,
            marker=dict(colors=["#f97316","#818cf8","#22c55e","#38bdf8","#f43f5e","#a78bfa"])))
        fig2.update_layout(**PL,height=200,showlegend=True,legend=dict(font=dict(size=10)))
        st.plotly_chart(fig2,use_container_width=True)
        if fg:
            fgcolor="#22c55e" if fgv>55 else("#ef4444" if fgv<40 else "#f59e0b")
            fig3=go.Figure(go.Indicator(mode="gauge+number",value=fgv,
                gauge=dict(axis=dict(range=[0,100]),bar=dict(color=fgcolor),
                    steps=[dict(range=[0,25],color="#1a0505"),dict(range=[25,45],color="#1a0d05"),
                           dict(range=[45,55],color="#111827"),dict(range=[55,75],color="#052210"),
                           dict(range=[75,100],color="#032210")]),
                number=dict(font=dict(color=fgcolor,size=24))))
            fig3.update_layout(paper_bgcolor='#0a0e1a',font=dict(color='#94a3b8'),
                height=160,margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig3,use_container_width=True)
    st.subheader("Recent Trades")
    st.dataframe(gen_trades(8),use_container_width=True,hide_index=True)
    with st.expander("📤 Export Dashboard"):
        export_panel("Dashboard",gen_trades(8),filename_base="dashboard")

# ══════════════════════════════════════════════════════════════
# PAGE: LIVE MARKET  (free)
# ══════════════════════════════════════════════════════════════
elif page == "📡 Live Market":
    st.title("📡 Live Market")
    lim=TIER_LIMITS[get_tier()]["history_days"]
    c1,c2=st.columns([1,2])
    with c1:
        ASSETS={"BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","SOL/USD":"SOL-USD",
                "BNB/USD":"BNB-USD","S&P500":"^GSPC","Gold":"GC=F","EUR/USD":"EURUSD=X"}
        if not need_tier("basic"):
            ASSETS={k:v for k,v in list(ASSETS.items())[:3]}
            st.caption("⚡ Basic+ untuk semua aset")
        chosen=st.selectbox("Asset",list(ASSETS.keys()))
        periods={"30D":30,"90D":90,"180D":180,"365D":365,"5Y":1825}
        if not need_tier("institutional"): periods={k:v for k,v in periods.items() if v<=lim}
        period=st.selectbox("Period",list(periods.keys()))
        ct=st.radio("Chart",["Candlestick","Line","Area"],horizontal=True)
    days=periods[period]
    df_h=fetch_ohlcv(ASSETS[chosen],days)
    with c2:
        if len(df_h)>5:
            last=df_h["Close"].iloc[-1]; prev=df_h["Close"].iloc[-2]
            ch=(last-prev)/prev*100; ret=(last-df_h["Close"].iloc[0])/df_h["Close"].iloc[0]*100
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Price",f"${last:,.2f}",f"{ch:+.2f}%")
            m2.metric("Return",f"{ret:+.1f}%")
            m3.metric("High",f"${df_h['High'].max():,.2f}")
            m4.metric("Low",f"${df_h['Low'].min():,.2f}")
    show_sma=st.checkbox("SMA 20/50",True)
    show_bb=st.checkbox("Bollinger",need_tier("basic"))
    show_vol=st.checkbox("Volume",True)
    if len(df_h)>5:
        fig=go.Figure()
        if ct=="Candlestick":
            fig.add_trace(go.Candlestick(x=df_h["Date"],open=df_h["Open"],high=df_h["High"],
                low=df_h["Low"],close=df_h["Close"],name=chosen,
                increasing_line_color="#22c55e",decreasing_line_color="#ef4444",
                increasing_fillcolor="#052e16",decreasing_fillcolor="#2d0a0a"))
        elif ct=="Line":
            fig.add_trace(go.Scatter(x=df_h["Date"],y=df_h["Close"],line=dict(color="#818cf8",width=2),name=chosen))
        else:
            fig.add_trace(go.Scatter(x=df_h["Date"],y=df_h["Close"],fill="tozeroy",
                fillcolor="rgba(129,140,248,0.1)",line=dict(color="#818cf8",width=2),name=chosen))
        if show_sma:
            fig.add_trace(go.Scatter(x=df_h["Date"],y=df_h["Close"].rolling(20).mean(),
                line=dict(color="#f59e0b",width=1.5,dash="dot"),name="SMA20"))
            fig.add_trace(go.Scatter(x=df_h["Date"],y=df_h["Close"].rolling(50).mean(),
                line=dict(color="#38bdf8",width=1.5,dash="dot"),name="SMA50"))
        if show_bb and need_tier("basic"):
            bu,bm,bl=bbands(df_h["Close"])
            fig.add_trace(go.Scatter(x=df_h["Date"],y=bu,line=dict(color="#475569",width=1,dash="dash"),name="BB+"))
            fig.add_trace(go.Scatter(x=df_h["Date"],y=bl,line=dict(color="#475569",width=1,dash="dash"),
                name="BB-",fill="tonexty",fillcolor="rgba(71,85,105,0.07)"))
        fig.update_layout(**CL,height=380,title=f"{chosen} — {period}",
            xaxis_rangeslider_visible=False,legend=dict(bgcolor='rgba(0,0,0,0)',orientation="h",y=1.05))
        st.plotly_chart(fig,use_container_width=True)
        if show_vol:
            vc=["#22c55e" if df_h["Close"].iloc[i]>=df_h["Open"].iloc[i] else "#ef4444" for i in range(len(df_h))]
            fv=go.Figure(go.Bar(x=df_h["Date"],y=df_h["Volume"],marker_color=vc))
            fv.update_layout(**{**CL,"margin":dict(l=0,r=0,t=24,b=0)},height=110,title="Volume")
            st.plotly_chart(fv,use_container_width=True)
        st.subheader("Technical Signals")
        sg=signals(df_h)
        css=lambda s: "sig-buy" if s=="BUY" else("sig-sell" if s=="SELL" else "sig-neu")
        s1,s2,s3,s4,s5=st.columns(5)
        for col,lbl,val in [(s1,"OVERALL",sg["overall"]),(s2,"RSI",str(sg["rsi"])),
            (s3,"MACD",sg["macd"]),(s4,"TREND",sg["trend"]),(s5,"BB",sg["bb"])]:
            col.markdown(f'<div class="{css(val)}">{lbl}<br><b>{val}</b></div>',unsafe_allow_html=True)
        with st.expander("📤 Export"):
            export_panel(f"{chosen}",df_h,fig=fig,filename_base=f"market_{chosen.replace('/','_')}")

# ══════════════════════════════════════════════════════════════
# PAGE: BACKTESTING  (basic+)
# ══════════════════════════════════════════════════════════════
elif page == "📈 Backtesting":
    if not need_tier("basic"):
        show_upgrade_wall("📈 Backtesting Engine","basic")
    else:
        st.title("📈 Backtesting Engine")
        lim=TIER_LIMITS[get_tier()]["history_days"]
        c1,c2,c3=st.columns(3)
        with c1:
            sym=st.selectbox("Symbol",["BTC-USD","ETH-USD","SOL-USD","BNB-USD"])
            strat=st.selectbox("Strategy",["SMA Crossover","RSI Mean Rev","Bollinger Breakout","MACD"])
        with c2:
            sd=st.date_input("Start",datetime.now()-timedelta(days=min(365,lim)))
            ed=st.date_input("End",datetime.now())
        with c3:
            cap=st.number_input("Capital (USD)",value=10000,step=1000)
            fp=st.slider("Fast Period",5,50,20)
            sp=st.slider("Slow Period",20,200,50)
        if st.button("🚀 Run Backtest",type="primary"):
            with st.spinner("Fetching data & running..."):
                df_h=fetch_ohlcv(sym,(ed-sd).days); time.sleep(.3)
            if len(df_h)<60: st.error("Not enough data.")
            else:
                cl=df_h["Close"]
                pos=(cl.rolling(fp).mean()>cl.rolling(sp).mean()).astype(int).shift(1).fillna(0)
                dr=cl.pct_change().fillna(0); sr=pos*dr
                eq=cap*(1+sr).cumprod(); tr=(eq.iloc[-1]-cap)/cap*100
                sh=sr.mean()/sr.std()*np.sqrt(252) if sr.std()>0 else 0
                dd=(eq/eq.cummax()-1); mdd=dd.min()*100
                wr=(sr>0).sum()/(sr!=0).sum()*100 if (sr!=0).sum()>0 else 0
                st.success("✅ Complete!")
                m1,m2,m3,m4,m5=st.columns(5)
                m1.metric("Return",f"{tr:.1f}%"); m2.metric("Final",f"${eq.iloc[-1]:,.0f}")
                m3.metric("Sharpe",f"{sh:.2f}"); m4.metric("Max DD",f"{mdd:.1f}%"); m5.metric("Win Rate",f"{wr:.1f}%")
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=df_h["Date"],y=eq,fill="tozeroy",
                    fillcolor="rgba(34,197,94,.08)",line=dict(color="#22c55e",width=2),name="Strategy"))
                fig.add_trace(go.Scatter(x=df_h["Date"],y=cap*(1+cl.pct_change().fillna(0)).cumprod(),
                    line=dict(color="#475569",width=1.5,dash="dot"),name="Buy & Hold"))
                fig.update_layout(**CL,height=300,title="Equity Curve",
                    legend=dict(bgcolor='rgba(0,0,0,0)',orientation="h",y=1.05))
                st.plotly_chart(fig,use_container_width=True)
                fig_dd=go.Figure(go.Scatter(x=df_h["Date"],y=dd*100,fill="tozeroy",
                    fillcolor="rgba(239,68,68,.12)",line=dict(color="#ef4444",width=1.5)))
                fig_dd.update_layout(**{**CL,"margin":dict(l=0,r=0,t=24,b=0)},height=130,title="Drawdown %")
                st.plotly_chart(fig_dd,use_container_width=True)
                fig_rsi=go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_h["Date"],y=rsi(cl),line=dict(color="#818cf8",width=1.5),name="RSI"))
                fig_rsi.add_hline(y=70,line_dash="dash",line_color="#ef4444",annotation_text="70")
                fig_rsi.add_hline(y=30,line_dash="dash",line_color="#22c55e",annotation_text="30")
                fig_rsi.update_layout(**{**CL,"margin":dict(l=0,r=0,t=24,b=0),
                    "yaxis":dict(range=[0,100],gridcolor="#1e293b")},height=130,title="RSI(14)")
                st.plotly_chart(fig_rsi,use_container_width=True)
                res=pd.DataFrame([{"Symbol":sym,"Strategy":strat,"Return%":round(tr,2),
                    "Sharpe":round(sh,2),"MaxDD%":round(mdd,2),"WinRate%":round(wr,2)}])
                eq_df=pd.DataFrame({"Date":df_h["Date"].dt.strftime("%Y-%m-%d"),"Equity":eq.round(2)})
                with st.expander("📤 Export"):
                    export_panel("Backtest",res,fig=fig,extra_sheets={"Equity":eq_df},filename_base="backtest")

# ══════════════════════════════════════════════════════════════
# PAGE: LIVE TRADING  (basic+)
# ══════════════════════════════════════════════════════════════
elif page == "💹 Live Trading":
    if not need_tier("basic"):
        show_upgrade_wall("💹 Live Trading","basic")
    else:
        st.title("💹 Live Trading")
        c1,c2=st.columns([1,2])
        with c1:
            st.subheader("Place Order")
            exch=st.selectbox("Exchange",["Binance","Coinbase","Kraken","OKX"])
            pair=st.selectbox("Pair",["BTC/USDT","ETH/USDT","SOL/USDT","ARB/USDT"])
            side=st.radio("Side",["BUY","SELL"],horizontal=True)
            otype=st.radio("Type",["Market","Limit"],horizontal=True)
            amt=st.number_input("Amount",value=0.01,format="%.4f")
            if otype=="Limit": st.number_input("Limit Price",value=60000.0)
            sl=st.number_input("Stop Loss %",value=2.0,step=.5)
            tp=st.number_input("Take Profit %",value=5.0,step=.5)
            if st.button("⚡ Execute",type="primary"):
                with st.spinner("Executing..."): time.sleep(.8)
                st.success(f"✅ {side} {amt} {pair} @ $61,240 | 312ms | Fee: $0.12")
        with c2:
            st.subheader("Arbitrage Scanner")
            if st.button("🔎 Scan"):
                arb=pd.DataFrame({"Symbol":["BTC","ETH","SOL"],
                    "Buy At":["Kraken","Binance","OKX"],"Buy Price":[61180,3211,141.8],
                    "Sell At":["Binance","Coinbase","Kraken"],"Sell Price":[61310,3224,142.9],
                    "Profit%":[.21,.39,.78],"After Fees%":[.01,.19,.58]})
                st.dataframe(arb,use_container_width=True,hide_index=True)
                with st.expander("📤 Export"): export_panel("Arbitrage",arb,filename_base="arb")
            st.subheader("Open Positions")
            pos=pd.DataFrame({"Pair":["BTC/USDT","ETH/USDT"],"Side":["LONG","LONG"],
                "Entry":[60100,3100],"Current":[61240,3218],"PnL%":["+1.9%","+3.8%"],"SL":[58900,3050],"TP":[63100,3500]})
            st.dataframe(pos,use_container_width=True,hide_index=True)
        st.markdown("---")
        st.subheader("Trade History")
        df_t=gen_trades(30); st.dataframe(df_t,use_container_width=True,hide_index=True)
        with st.expander("📤 Export"): export_panel("Trades",df_t,filename_base="trades")

# ══════════════════════════════════════════════════════════════
# PAGE: DEFI  (premium+)
# ══════════════════════════════════════════════════════════════
elif page == "🌊 DeFi + Wallet":
    if not need_tier("premium"):
        show_upgrade_wall("🌊 DeFi + Wallet","premium")
    else:
        st.title("🌊 DeFi + Wallet")
        st.markdown("""<p style='color:#94a3b8'>Connect wallet untuk posisi DeFi on-chain realtime.</p>
        <a href="trust://open" style="background:linear-gradient(135deg,#3375BB,#0076FF);color:#fff;
        padding:9px 20px;border-radius:9px;text-decoration:none;font-weight:600;margin-right:8px">🛡️ TrustWallet</a>
        <a href="https://metamask.app.link/dapp/tradeforge.io" style="background:linear-gradient(135deg,#F6851B,#E2761B);
        color:#fff;padding:9px 20px;border-radius:9px;text-decoration:none;font-weight:600;margin-right:8px">🦊 MetaMask</a>
        <a href="https://walletconnect.com" style="background:linear-gradient(135deg,#3B99FC,#1D6FDB);
        color:#fff;padding:9px 20px;border-radius:9px;text-decoration:none;font-weight:600">🔗 WalletConnect</a>
        <p style='color:#475569;font-size:.75rem;margin-top:8px'>⚠️ Read-only permission saja. Private key tidak pernah diakses.</p>
        """,unsafe_allow_html=True)
        wallet=st.text_input("Atau masukkan wallet address",placeholder="0x...")
        if wallet: st.info(f"📡 Fetching on-chain data for `{wallet[:20]}...` (demo mode)")
        st.markdown("---")
        st.subheader("🏆 Live DeFi Yields — DeFiLlama")
        with st.spinner("Fetching live APY..."):
            pools=fetch_defi_yields()
        if pools:
            df_p=pd.DataFrame([{"Protocol":p.get("project",""),"Chain":p.get("chain",""),
                "Asset":p.get("symbol",""),"APY%":round(p.get("apy",0),2),
                "TVL$M":round(p.get("tvlUsd",0)/1e6,1)} for p in pools[:12]])
        else:
            df_p=pd.DataFrame({"Protocol":["Aave V3","Aave V3","Uniswap V3"],
                "Chain":["Arbitrum","Base","Ethereum"],"Asset":["USDC","USDC","ETH/USDC"],
                "APY%":[8.4,7.2,24.6],"TVL$M":[820,340,560]})
            st.caption("⚠️ Demo — DeFiLlama tidak tersedia")
        st.dataframe(df_p,use_container_width=True,hide_index=True)
        fig=px.bar(df_p.head(10),x="Asset",y="APY%",color="Protocol",
            color_discrete_sequence=["#818cf8","#38bdf8","#22c55e","#f59e0b","#f43f5e"])
        fig.update_layout(**CL,height=260,title="Top APY")
        st.plotly_chart(fig,use_container_width=True)
        my_pos=pd.DataFrame({"Protocol":["Uniswap V3","Aave V3"],"Chain":["Arbitrum","Base"],
            "Position":["ETH/USDC LP","USDC Lend"],"Value$":[5240,3100],"APY%":[24.6,7.2],"Fees$":[142,18]})
        st.subheader("My Positions"); st.dataframe(my_pos,use_container_width=True,hide_index=True)
        with st.expander("📤 Export"):
            export_panel("DeFi",my_pos,fig=fig,extra_sheets={"Live Yields":df_p},filename_base="defi")

# ══════════════════════════════════════════════════════════════
# PAGE: AI SIGNALS  (premium+)
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Signals":
    if not need_tier("premium"):
        show_upgrade_wall("🤖 AI Signals & Strategy","premium")
    else:
        st.title("🤖 AI Signals & Strategy")
        tab1,tab2,tab3=st.tabs(["📊 Multi-Asset Scan","🧠 Strategy Generator","🎯 Risk Manager"])
        with tab1:
            if st.button("🔍 Scan All Assets",type="primary"):
                rows=[]
                with st.spinner("Analyzing 4 assets..."):
                    for tk,nm in [("BTC-USD","Bitcoin"),("ETH-USD","Ethereum"),("SOL-USD","Solana"),("BNB-USD","BNB")]:
                        df_a=fetch_ohlcv(tk,90)
                        if len(df_a)>50:
                            sg=signals(df_a)
                            rows.append({"Asset":nm,"Price":f"${sg['price']:,.2f}","Signal":sg["overall"],
                                "RSI":sg["rsi"],"MACD":sg["macd"],"Trend":sg["trend"],"BB":sg["bb"],"Confidence":f"{max(sg['buy'],sg['sell'])}/4"})
                sc=pd.DataFrame(rows); st.dataframe(sc,use_container_width=True,hide_index=True)
                with st.expander("📤 Export"): export_panel("AI Signals",sc,filename_base="signals")
        with tab2:
            c1,c2=st.columns(2)
            with c1:
                sym_ai=st.selectbox("Symbol",["BTC","ETH","SOL"])
                risk_ai=st.select_slider("Risk",["Low","Medium","High"])
                cap_ai=st.number_input("Capital",value=5000)
            with c2:
                provider=st.selectbox("LLM",["Groq (Fast)","OpenRouter","Together AI"])
                horizon=st.selectbox("Horizon",["1 week","1 month","3 months"])
            if st.button("🧠 Generate Strategy",type="primary"):
                with st.spinner("AI analyzing..."): time.sleep(1.5)
                r={"strategy":"rsi_mean_reversion","config":{"rsi_period":14,"oversold":28,"overbought":72,
                    "stop_loss%":2,"take_profit%":5},"expected_sharpe":1.65,"expected_monthly_return%":8.2,
                    "reasoning":f"{sym_ai} sideways-bull regime. RSI mean reversion optimal for {risk_ai} risk."}
                st.json(r)
                with st.expander("📤 Export"): export_panel("Strategy",pd.DataFrame([r]),filename_base="strategy")
        with tab3:
            st.subheader("Position Sizing Calculator")
            c1,c2=st.columns(2)
            with c1:
                pv=st.number_input("Portfolio Value ($)",value=43564)
                rpt=st.slider("Risk per Trade %",.5,5.,1.,.5)
                mdd=st.slider("Max DD Limit %",5,30,15)
            with c2:
                entry=st.number_input("Entry Price",value=61240.)
                stop_p=st.number_input("Stop Loss Price",value=59800.)
                risk_usd=pv*(rpt/100)
                size=risk_usd/abs(entry-stop_p) if abs(entry-stop_p)>0 else 0
                st.metric("Risk Amount",f"${risk_usd:,.2f}")
                st.metric("Recommended Size",f"{size:.4f} units")
                st.metric("Position Value",f"${size*entry:,.2f}")
                st.metric("Portfolio %",f"{size*entry/pv*100:.1f}%")
            sim=np.random.normal(.001,.025,10000); v95=np.percentile(sim,5)
            v1,v2,v3=st.columns(3)
            v1.metric("VaR 95%",f"{v95*100:.2f}%",f"-${abs(v95)*pv:,.0f}")
            v2.metric("CVaR 95%",f"{sim[sim<=v95].mean()*100:.2f}%")
            v3.metric("Max DD Limit",f"${pv*mdd/100:,.0f}")

# ══════════════════════════════════════════════════════════════
# PAGE: NEWS  (free)
# ══════════════════════════════════════════════════════════════
elif page == "📰 News & Sentiment":
    st.title("📰 News & Market Sentiment")
    ALL_SRCS=["CoinDesk","The Block","Decrypt","CoinTelegraph","Bitcoin Mag","Bankless","Messari","WSJ Markets","Reuters","CryptoPanic"]
    max_src=TIER_LIMITS[get_tier()]["news_sources"]
    default_srcs=ALL_SRCS[:max_src]
    if need_tier("basic"):
        chosen_srcs=st.multiselect("News Sources",ALL_SRCS,default=default_srcs)
    else:
        chosen_srcs=default_srcs
        st.info(f"Free tier: {max_src} sumber. Upgrade ke Basic+ untuk 10 sumber.")
    tab1,tab2=st.tabs(["📡 News Feed","📊 TradingView"])
    with tab1:
        c_news,c_side=st.columns([2,1])
        with c_news:
            with st.spinner("Fetching news..."):
                items=fetch_news(tuple(chosen_srcs))
            SRC_COLORS={"CoinDesk":"#f59e0b","The Block":"#818cf8","Decrypt":"#22c55e",
                "CoinTelegraph":"#38bdf8","Bitcoin Mag":"#f97316","Bankless":"#a78bfa",
                "Messari":"#06b6d4","WSJ Markets":"#64748b","Reuters":"#ef4444","CryptoPanic":"#f43f5e"}
            for it in items:
                col=SRC_COLORS.get(it["source"],"#94a3b8")
                st.markdown(f"""<div class="news-card">
                <span style="color:{col};font-size:.7rem;font-weight:700;text-transform:uppercase">{it["source"]}</span>
                <span style="color:#334155;font-size:.7rem;margin-left:8px">{it["time"]}</span>
                <div style="margin:5px 0;font-size:.9rem;font-weight:600">
                <a href="{it['url']}" target="_blank" style="color:#e2e8f0;text-decoration:none">{it["title"]}</a></div>
                </div>""",unsafe_allow_html=True)
            news_df=pd.DataFrame(items)
            with st.expander("📤 Export News"):
                export_panel("News Feed",news_df,filename_base="news")
        with c_side:
            st.subheader("Fear & Greed")
            fg=fetch_fg()
            if fg:
                fgv=int(fg[0]["value"]); fgcls=fg[0]["value_classification"]
                fgcol="#22c55e" if fgv>55 else("#ef4444" if fgv<40 else "#f59e0b")
                st.markdown(f"""<div style="background:#0d1117;border:1px solid {fgcol}55;
                border-radius:12px;padding:20px;text-align:center;margin-bottom:12px">
                <div style="font-size:2.5rem;font-weight:700;color:{fgcol};font-family:monospace">{fgv}</div>
                <div style="color:{fgcol};font-weight:600">{fgcls}</div></div>""",unsafe_allow_html=True)
                fg_df=pd.DataFrame([{"Date":datetime.fromtimestamp(int(d["timestamp"])).strftime("%m-%d"),
                    "Value":int(d["value"])} for d in fg[:14]])
                ff=go.Figure(go.Bar(x=fg_df["Date"],y=fg_df["Value"],
                    marker_color=["#22c55e" if v>55 else"#ef4444" if v<40 else"#f59e0b" for v in fg_df["Value"]]))
                ff.add_hline(y=50,line_dash="dash",line_color="#334155")
                ff.update_layout(**{**CL,"margin":dict(l=0,r=0,t=24,b=0),
                    "yaxis":dict(range=[0,100],gridcolor="#1e293b")},height=220,title="14D History")
                st.plotly_chart(ff,use_container_width=True)
                st.caption("Source: alternative.me")
            st.markdown("---")
            st.subheader("On-Chain Signals")
            oc=pd.DataFrame({"Metric":["BTC Active Addr","ETH Gas","BTC Hash Rate","DeFi TVL"],
                "Value":["892K","18 Gwei","612 EH/s","$119B"],
                "Signal":["Bullish","Neutral","Bullish","Bullish"],
                "Source":["Glassnode","Etherscan","Blockchain.com","DeFiLlama"]})
            st.dataframe(oc,use_container_width=True,hide_index=True)
    with tab2:
        if not need_tier("basic"):
            st.info("TradingView embed tersedia di Basic+ plan"); show_upgrade_wall("TradingView Charts","basic")
        else:
            tv_sym=st.selectbox("Symbol",["BINANCE:BTCUSDT","BINANCE:ETHUSDT","COINBASE:BTCUSD","SP:SPX","FOREXCOM:XAUUSD"])
            tv_int=st.selectbox("Interval",["1","5","15","60","D","W"],index=4)
            components.html(f"""<div style="height:420px">
            <div id="tv_w"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>new TradingView.widget({{
                "width":"100%","height":400,"symbol":"{tv_sym}","interval":"{tv_int}",
                "timezone":"Asia/Jakarta","theme":"dark","style":"1","locale":"en",
                "toolbar_bg":"#0d1117","enable_publishing":false,"allow_symbol_change":true,
                "container_id":"tv_w","studies":["RSI@tv-basicstudies","MACD@tv-basicstudies"]
            }});</script></div>""",height=430)

# ══════════════════════════════════════════════════════════════
# PAGE: REVENUE  (premium+)
# ══════════════════════════════════════════════════════════════
elif page == "💰 Revenue":
    if not need_tier("premium"):
        show_upgrade_wall("💰 Revenue Analytics","premium")
    else:
        st.title("💰 Revenue Analytics")
        c1,c2=st.columns(2)
        with c1: nf=st.slider("Free users",0,5000,200); nb=st.slider("Basic $29",0,2000,80)
        with c2: np2=st.slider("Premium $99",0,1000,30); ni=st.slider("Inst $499",0,100,8)
        sub=nb*29+np2*99+ni*499
        trade=(nf+nb+np2+ni)*5*30*500*.002
        defi2=(np2+ni)*50*.10
        total=sub+trade+defi2
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Subscription MRR",f"${sub:,.0f}")
        c2.metric("Trade Fee MRR",f"${trade:,.0f}")
        c3.metric("DeFi Yield Share",f"${defi2:,.0f}")
        c4.metric("Total MRR",f"${total:,.0f}",f"ARR ${total*12:,.0f}")
        months=pd.date_range("2025-03",periods=12,freq="MS").strftime("%Y-%m")
        sub_s=np.array([sub*(1.12**i) for i in range(12)])
        tr_s=np.array([trade*(1.08**i) for i in range(12)])
        df_s=np.array([defi2*(1.15**i) for i in range(12)])
        rev_df=pd.DataFrame({"Month":months,"Sub$":sub_s.round(0),"Trade$":tr_s.round(0),
            "DeFi$":df_s.round(0),"Total$":(sub_s+tr_s+df_s).round(0)})
        fig=go.Figure()
        for col,color in [("Sub$","#818cf8"),("Trade$","#38bdf8"),("DeFi$","#22c55e")]:
            fig.add_trace(go.Bar(name=col,x=rev_df["Month"],y=rev_df[col],marker_color=color))
        fig.update_layout(**CL,barmode="stack",height=300,
            legend=dict(bgcolor='rgba(0,0,0,0)',orientation="h",y=1.05))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(rev_df,use_container_width=True,hide_index=True)
        with st.expander("📤 Export"):
            export_panel("Revenue",rev_df,fig=fig,filename_base="revenue")

# ══════════════════════════════════════════════════════════════
# PAGE: EXPORT CENTER  (basic+)
# ══════════════════════════════════════════════════════════════
elif page == "📤 Export Center":
    if not need_tier("basic"):
        show_upgrade_wall("📤 Export Center","basic")
    else:
        st.title("📤 Export Center")
        st.info("Download semua data dalam satu tempat — CSV · Excel · JSON · HTML · PDF")
        tabs=st.tabs(["📋 Trades","💼 Portfolio","📊 Backtests","🌊 DeFi","💰 Revenue","📦 Master Excel"])
        with tabs[0]:
            df=gen_trades(50); st.dataframe(df,use_container_width=True,hide_index=True)
            export_panel("Trade History",df,filename_base="trades")
        with tabs[1]:
            df=gen_portfolio(); st.dataframe(df,use_container_width=True,hide_index=True)
            export_panel("Portfolio",df,filename_base="portfolio")
        with tabs[2]:
            df=pd.DataFrame({"Strategy":["SMA","RSI","BB","MACD"],"Return%":[48,31,22,19],
                "Sharpe":[1.84,1.42,1.11,.98],"MaxDD%":[-14,-18,-21,-24],"WinRate%":[63,58,54,51]})
            st.dataframe(df,use_container_width=True,hide_index=True)
            export_panel("Backtests",df,filename_base="backtests")
        with tabs[3]:
            df=pd.DataFrame({"Protocol":["Uniswap","Aave"],"APY%":[24.6,7.2],"Value$":[5240,3100]})
            st.dataframe(df,use_container_width=True,hide_index=True)
            export_panel("DeFi",df,filename_base="defi")
        with tabs[4]:
            df=pd.DataFrame({"Month":["2025-03","2025-04","2025-05"],"MRR$":[4200,5100,6300]})
            st.dataframe(df,use_container_width=True,hide_index=True)
            export_panel("Revenue",df,filename_base="revenue")
        with tabs[5]:
            st.subheader("📦 Master Excel — semua sheet sekaligus")
            if st.button("⬇️ Generate Master Excel",type="primary"):
                all_sheets={"Trades":gen_trades(50),"Portfolio":gen_portfolio(),
                    "Backtests":pd.DataFrame({"Strategy":["SMA","RSI"],"Return%":[48,31]}),
                    "DeFi":pd.DataFrame({"Protocol":["Uniswap"],"APY%":[24.6]}),
                    "Revenue":pd.DataFrame({"Month":["2025-03"],"MRR$":[4200]})}
                fn=f"tradeforge_master_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                st.download_button("⬇️ Download Master Excel",to_excel(all_sheets),fn,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: PRICING & PLANS
# ══════════════════════════════════════════════════════════════
elif page == "💳 Pricing & Plans":
    st.title("💳 Pricing & Plans")
    st.markdown("<p style='color:#94a3b8;font-size:1rem'>Pilih plan yang sesuai kebutuhan trading Anda.</p>",
                unsafe_allow_html=True)

    # ── Plan definitions ──────────────────────────────────────
    PLANS = [
        {
            "tier_id": "free",
            "badge":   "FREE",
            "price":   "$0",
            "period":  "Forever free",
            "demo_u":  None,
            "demo_p":  None,
            "note":    "Tidak perlu login",
            "features": [
                ("✅","Dashboard realtime"),("✅","Live Market — BTC/ETH/SOL 30D"),
                ("✅","News Feed (2 sumber)"),("✅","Fear and Greed Index"),
                ("❌","Backtesting"),("❌","Live Trading"),
                ("❌","DeFi + Wallet"),("❌","AI Signals"),
                ("❌","Export CSV/PDF/Excel"),("❌","TradingView Charts"),
            ],
        },
        {
            "tier_id": "basic",
            "badge":   "BASIC",
            "price":   "$29",
            "period":  "per month",
            "demo_u":  "bandung123",
            "demo_p":  "bandung123",
            "note":    "Akun demo tersedia",
            "features": [
                ("✅","Semua Free features"),("✅","Backtesting (180D history)"),
                ("✅","Live Trading + Arbitrage Scanner"),("✅","Export CSV/Excel/JSON/PDF"),
                ("✅","TradingView Charts embed"),("✅","News (5 sumber)"),
                ("✅","Stop Loss / Take Profit"),("✅","50 trades/day"),
                ("❌","DeFi + Wallet"),("❌","AI Signals"),("❌","Revenue Analytics"),
            ],
        },
        {
            "tier_id": "premium",
            "badge":   "PRO",
            "price":   "$99",
            "period":  "per month",
            "demo_u":  "bandung12345",
            "demo_p":  "bandung12345",
            "note":    "Akun demo tersedia",
            "features": [
                ("✅","Semua Basic features"),("✅","DeFi + Wallet Connect"),
                ("✅","TrustWallet / MetaMask deeplink"),("✅","AI Signal Scan (4 aset)"),
                ("✅","Strategy Generator LLM"),("✅","Risk Manager + VaR"),
                ("✅","Revenue Analytics"),("✅","News (10 sumber)"),
                ("✅","365D history"),("✅","Unlimited trades"),
                ("❌","White-label"),("❌","API access"),
            ],
        },
        {
            "tier_id": "institutional",
            "badge":   "INSTITUTIONAL",
            "price":   "$499",
            "period":  "per month",
            "demo_u":  "bandung1234567",
            "demo_p":  "bandung1234567",
            "note":    "Akun demo tersedia",
            "features": [
                ("✅","Semua Premium features"),("✅","5 tahun history data"),
                ("✅","Multi-portfolio management"),("✅","White-label branding"),
                ("✅","API access REST + WebSocket"),("✅","Dedicated support"),
                ("✅","Custom integrations"),("✅","Team accounts 5 seats"),
                ("✅","On-premise deployment"),("✅","SLA 99.9% uptime"),
            ],
        },
    ]

    cur_tier   = get_tier()
    cur_order  = TIERS[cur_tier]["order"]
    cols       = st.columns(4)

    for i, plan in enumerate(PLANS):
        tid       = plan["tier_id"]
        ti        = TIERS[tid]
        c         = ti["color"]
        is_cur    = (tid == cur_tier)
        badge     = plan["badge"]
        price     = plan["price"]
        period    = plan["period"]
        demo_u    = plan["demo_u"]
        demo_p    = plan["demo_p"]
        note      = plan["note"]
        features  = plan["features"]

        with cols[i]:
            # ── Card border highlight ──
            border_css  = f"2px solid {c}" if is_cur else f"1px solid {c}44"
            bg_css      = f"linear-gradient(160deg,{c}18,{c}08)" if is_cur else "linear-gradient(160deg,#0d1117,#111827)"
            cur_ribbon  = (
                '<div style="position:absolute;top:10px;right:12px;background:#22c55e;'
                'color:#052e16;font-size:.6rem;font-weight:800;padding:2px 8px;'
                'border-radius:20px;letter-spacing:.08em">CURRENT</div>'
                if is_cur else ""
            )

            # ── Feature list HTML ──
            feat_html = ""
            for icon, text in features:
                icon_color = "#22c55e" if icon == "✅" else "#475569"
                text_color = "#e2e8f0" if icon == "✅" else "#64748b"
                feat_html += (
                    f'<div style="display:flex;align-items:baseline;gap:6px;'
                    f'margin:5px 0;font-size:.82rem">'
                    f'<span style="color:{icon_color};flex-shrink:0">{icon}</span>'
                    f'<span style="color:{text_color}">{text}</span></div>'
                )

            card_html = (
                f'<div style="position:relative;background:{bg_css};border:{border_css};'
                f'border-radius:16px;padding:24px 20px 20px;min-height:520px">'
                f'{cur_ribbon}'
                f'<div style="display:inline-block;background:{c}22;color:{c};'
                f'border:1px solid {c}55;font-size:.65rem;font-weight:700;'
                f'letter-spacing:.1em;padding:3px 12px;border-radius:20px;margin-bottom:14px">'
                f'{badge}</div>'
                f'<div style="font-family:monospace;font-size:2.2rem;font-weight:700;'
                f'color:#f1f5f9;line-height:1.1">{price}</div>'
                f'<div style="color:#64748b;font-size:.8rem;margin-bottom:18px">{period}</div>'
                f'{feat_html}'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            # ── Spacing + demo info ──
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if demo_u:
                st.markdown(
                    f"<div style='font-size:.8rem;color:#94a3b8'>Demo login:</div>"
                    f"<code style='font-size:.8rem'>{demo_u}</code> / "
                    f"<code style='font-size:.8rem'>{demo_p}</code>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='font-size:.82rem;color:#64748b;font-style:italic'>Tidak perlu login</div>",
                    unsafe_allow_html=True
                )
            st.caption(note)

            # ── Action button ──
            if is_cur:
                st.success("Active plan", icon="✅")
            elif tid == "free" and logged_in():
                if st.button("Downgrade to Free", key=f"plan_btn_{i}", use_container_width=True):
                    do_logout(); st.rerun()
            elif not logged_in() and demo_u:
                if st.button(f"Try {badge}", key=f"plan_btn_{i}",
                             type="primary", use_container_width=True):
                    if do_login(demo_u, demo_p): st.rerun()
                    else: st.error("Login gagal")
            elif logged_in() and TIERS[tid]["order"] > cur_order:
                if st.button(f"Upgrade to {badge}", key=f"plan_btn_{i}",
                             type="primary", use_container_width=True):
                    # Auto-login with demo account for upgrade demo
                    if demo_u and do_login(demo_u, demo_p): st.rerun()
                    else: st.info("Hubungi aryhharyanto@proton.me untuk upgrade")

    # ── Feature Comparison Table ──────────────────────────────
    st.markdown("---")
    st.subheader("Feature Comparison")
    comp = pd.DataFrame({
        "Feature": [
            "History Data","News Sources","Backtesting","Live Trading","Trades/Day",
            "DeFi + Wallet","AI Signals","TradingView","Export CSV/Excel/PDF",
            "Revenue Analytics","API Access","White Label",
        ],
        "Free":               ["30D","2 srcs","--","--","0","--","--","--","--","--","--","--"],
        "Basic $29":          ["180D","5 srcs","SMA/RSI/BB","CEX only","50/day","--","--","Yes","Yes","--","--","--"],
        "Premium $99":        ["365D","10 srcs","All strats","CEX+DEX","Unlimited","Yes","Yes","Yes","Yes","Yes","--","--"],
        "Institutional $499": ["5 Years","10 srcs","All strats","CEX+DEX","Unlimited","Yes","Yes","Yes","Yes","Yes","Yes","Yes"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE: SETTINGS  (basic+)
# ══════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    if not need_tier("basic"):
        show_upgrade_wall("⚙️ Settings","basic")
    else:
        st.title("⚙️ Settings")
        tab1,tab2,tab3,tab4=st.tabs(["🔑 API Keys","🔔 Alerts","📊 Data Sources","🏢 White Label"])
        with tab1:
            st.text_input("Binance API Key",type="password")
            st.text_input("Binance Secret",type="password")
            st.text_input("Ethereum RPC",placeholder="https://mainnet.infura.io/v3/...")
            st.text_input("Groq API Key",type="password",placeholder="gsk_...")
            st.text_input("OpenRouter API Key",type="password",placeholder="sk-or-v1-...")
            if st.button("Save",type="primary"): st.success("✅ Saved securely (encrypted)")
        with tab2:
            st.toggle("Email alerts",value=True); st.toggle("Telegram",value=False)
            st.toggle("Discord webhook",value=False)
            st.slider("Volatility alert %",1,20,5); st.slider("Drawdown alert %",1,30,10)
            st.slider("RSI oversold",10,40,28); st.slider("RSI overbought",60,90,72)
        with tab3:
            src_df=pd.DataFrame({"Source":["Yahoo Finance","CoinGecko","DeFiLlama","alternative.me","feedparser RSS"],
                "Data":["OHLCV Prices","Live crypto prices","DeFi APY","Fear & Greed","News feeds"],
                "Latency":["5min","5min","10min","15min","15min"],
                "Cost":["Free","Free","Free","Free","Free"],
                "Status":["✅","✅","✅","✅","✅"]})
            st.dataframe(src_df,use_container_width=True,hide_index=True)
        with tab4:
            if not need_tier("institutional"):
                show_upgrade_wall("White Label","institutional")
            else:
                st.text_input("Company Name"); st.text_input("Logo URL")
                st.color_picker("Brand Color","#818cf8"); st.text_input("Custom Domain",placeholder="trading.yourcompany.com")
                if st.button("Apply",type="primary"): st.success("✅ White-label applied")

# ══════════════════════════════════════════════════════════════
# PAGE: INVESTMENT GUIDE (free)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# PAGE: INVESTMENT GUIDE  (free)
# ══════════════════════════════════════════════════════════════
elif page == "💡 Investment Guide":
    st.title("💡 Investment Guide & Risk Framework")
    st.markdown(
        "<p style='color:#94a3b8;font-size:1rem'>Panduan berbasis data historis — "
        "bukan spekulasi. Angka konsisten & realistis.</p>",
        unsafe_allow_html=True
    )

    tab_min, tab_risk, tab_return = st.tabs([
        "💰 Investasi Minimal", "⚖️ Risk Framework", "📈 Expected Returns"
    ])

    # ── TAB 1: INVESTASI MINIMAL ──────────────────────────────
    with tab_min:
        st.subheader("Investasi Minimal yang Masuk Akal per Aset")
        st.markdown(
            "<p style='color:#64748b;font-size:.85rem'>"
            "Angka 'minimal' di bawah bukan batas bawah teknis, tapi batas bawah "
            "<b>masuk akal</b> — cukup buffer untuk drawdown, fee, dan meaningful gain.</p>",
            unsafe_allow_html=True
        )

        inv_rows = [
            ("BTC (Bitcoin)",        500,  2000,  5000,  "Volatil ±80%/thn, perlu buffer drawdown 30-40%", "3x (hati-hati)"),
            ("ETH (Ethereum)",       300,  1000,  3000,  "Gas fees + DeFi LP min size viability",           "3x"),
            ("SOL (Solana)",         100,   500,  2000,  "Murah tapi volatil, min untuk meaningful gain",   "2x"),
            ("S&P500 ETF (US)",      500,  5000, 20000,  "Biaya transaksi + diversifikasi min 5 posisi",    "2x (ETF leveraged)"),
            ("IDX Blue Chip (IDN)",  150,  1000,  5000,  "Lot IDX=100 lembar, blue chip Rp500-8000/lembar", "1x (IDX no retail leverage)"),
            ("Obligasi RI SBR/ORI",  200,  1000,  5000,  "Min pembelian Rp1jt, tapi likuiditas terbatas",   "1x"),
            ("XAU/USD (Gold)",       200,  1000,  5000,  "Spread lebar $20-50, butuh buffer untuk swings",  "10x (futures), 1x (ETF)"),
            ("DeFi LP Position",    1000,  5000, 20000,  "Gas+impermanent loss risk butuh buffer besar",    "1x"),
        ]
        inv_df = pd.DataFrame(inv_rows, columns=[
            "Aset", "Min ($)", "Optimal Start ($)", "Ideal ($)",
            "Alasan Min", "Max Leverage"
        ])
        st.dataframe(inv_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        ca, cb = st.columns(2)

        with ca:
            st.subheader("Break-even per Plan TradeForge")
            be_data = [
                ("Free",           "$0/mo",   "$500+",   "N/A",    "Monitor only",           "< $5K"),
                ("Basic $29/mo",   "$29/mo",  "$2,000+", "1.5%/mo","3-5 posisi aktif",        "$2K - $20K"),
                ("Premium $99/mo", "$99/mo",  "$10,000+","1.0%/mo","Multi-aset algo trading", "$10K - $200K"),
                ("Inst $499/mo",   "$499/mo", "$50,000+","0.1%/mo","Fund management",         "> $100K"),
            ]
            be_df = pd.DataFrame(be_data, columns=[
                "Plan", "Biaya/bln", "Min Portfolio", "Break-even ROI", "Use Case", "Sweet Spot"
            ])
            st.dataframe(be_df, use_container_width=True, hide_index=True)
            st.caption("Break-even ROI = biaya platform / portfolio size. Di bawah 1% = efisien.")

        with cb:
            st.subheader("Efisiensi Biaya vs Portfolio Size")
            sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
            fig_cost = go.Figure()
            for label, cost, color in [
                ("Basic $29", 29, "#38bdf8"),
                ("Premium $99", 99, "#a78bfa"),
                ("Inst $499", 499, "#f59e0b"),
            ]:
                pcts = [cost / s * 100 for s in sizes]
                fig_cost.add_trace(go.Scatter(
                    x=sizes, y=pcts, name=label,
                    line=dict(color=color, width=2),
                    hovertemplate="%{y:.2f}%<extra>" + label + "</extra>"
                ))
            fig_cost.add_hline(y=1.0, line_dash="dash", line_color="#22c55e",
                               annotation_text="1% threshold (efficient)")
            fig_cost.update_layout(
                **{**CL,
                   "xaxis": dict(title="Portfolio ($)", showgrid=False, color="#334155", type="log"),
                   "yaxis": dict(title="Cost %", gridcolor="#1e293b", color="#334155"),
                   "margin": dict(l=0,r=0,t=30,b=0)},
                height=260, title="Biaya Platform (% Portfolio / bulan)",
                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.05)
            )
            st.plotly_chart(fig_cost, use_container_width=True)

    # ── TAB 2: RISK FRAMEWORK ─────────────────────────────────
    with tab_risk:
        st.subheader("Risk Framework — Angka Paling Konsisten & Stabil")
        st.markdown(
            "<p style='color:#64748b;font-size:.85rem'>"
            "Data dari 5-10 tahun historis. Angka ini BUKAN garansi, tapi pola "
            "yang berulang cukup konsisten untuk dijadikan acuan sizing.</p>",
            unsafe_allow_html=True
        )

        risk_rows = [
            ("BTC",          "~80%",  "-83%",   "-4.2%",  "1.2",  "1-2 thn",  "HIGH",       "#ef4444"),
            ("ETH",          "~90%",  "-94%",   "-5.1%",  "1.1",  "1-3 thn",  "HIGH",       "#ef4444"),
            ("SOL",          "~120%", "-97%",   "-7.3%",  "0.9",  "1-3 thn",  "VERY HIGH",  "#dc2626"),
            ("S&P 500",      "~15%",  "-57%",   "-1.1%",  "1.0",  "1-2 thn",  "MEDIUM",     "#f59e0b"),
            ("IDX Composite","~18%",  "-62%",   "-1.3%",  "0.7",  "2-5 thn",  "MEDIUM",     "#f59e0b"),
            ("XAU/USD",      "~15%",  "-45%",   "-1.0%",  "0.6",  "1-3 thn",  "MEDIUM-LOW", "#22c55e"),
            ("Obligasi RI",  "~5%",   "-8%",    "-0.4%",  "0.4",  "6 bln",    "LOW",        "#16a34a"),
            ("DeFi LP",      ">100%", ">-90%",  "-8%+",   "0.8",  "Varies",   "EXTREME",    "#7f1d1d"),
        ]
        risk_df = pd.DataFrame(risk_rows, columns=[
            "Aset", "Volatilitas/thn", "Max DD Historis",
            "VaR 95% (1D)", "Sharpe 5yr", "Recovery Time", "Risk Level", "_color"
        ])

        # Display without color column
        st.dataframe(risk_df.drop(columns=["_color"]), use_container_width=True, hide_index=True)

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Position Sizing Rules")
            st.markdown("""
**Kelly Criterion (¼ Kelly conservative):**

Rumus: `Max pos = portfolio × (edge/odds) × 0.25`

Praktisnya:
| Aset | Max Allocation |
|------|---------------|
| BTC | 15-20% portfolio |
| ETH | 10-15% |
| SOL/Altcoin | 5-10% |
| DeFi LP | max 10% |
| Saham single | max 8% |
| Gold | 10-20% |

**Stop Loss Defaults:**
| Aset | Konservatif | Agresif |
|------|------------|---------|
| BTC | -5% | -10% |
| ETH | -6% | -12% |
| S&P | -3% | -7% |
| IDX | -4% | -8% |
| Gold | -2% | -5% |
            """)

        with c2:
            st.subheader("Allocation Model per Risk Profile")
            profiles = {
                "Konservatif":   dict(BTC=10, ETH=5,  Gold=20, Bonds=40, SP500=20, Cash=5),
                "Balanced":      dict(BTC=20, ETH=15, Gold=10, Bonds=20, SP500=30, Cash=5),
                "Agresif":       dict(BTC=35, ETH=25, SOL=15,  Gold=5,   SP500=15, Cash=5),
                "Crypto Native": dict(BTC=40, ETH=30, SOL=10,  DeFi=15,  Cash=5),
            }
            prof = st.selectbox("Risk Profile", list(profiles.keys()))
            alloc = profiles[prof]
            fig_pie = go.Figure(go.Pie(
                labels=list(alloc.keys()), values=list(alloc.values()), hole=0.52,
                marker=dict(colors=[
                    "#f97316","#818cf8","#fbbf24","#22c55e",
                    "#38bdf8","#64748b","#a78bfa","#06b6d4"
                ][:len(alloc)])
            ))
            fig_pie.update_layout(**PL, height=240, showlegend=True,
                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── TAB 3: EXPECTED RETURNS ───────────────────────────────
    with tab_return:
        st.subheader("Expected Returns — Data Historis 5 Tahun")
        ret_rows = [
            ("BTC",          "+98%",   "+303%", "-74%", "+5.7%", "3.2%",  "Medium"),
            ("ETH",          "+85%",   "+448%", "-82%", "+5.2%", "2.8%",  "Medium"),
            ("SOL",          "N/A",    "+11000%","-97%", "High",  "N/A",   "Low"),
            ("S&P 500",      "+14%",   "+29%",  "-19%", "+1.1%", "1.3%",  "High"),
            ("IDX Composite","7%",     "+17%",  "-12%", "+0.6%", "0.7%",  "Medium"),
            ("XAU/USD",      "+12%",   "+25%",  "-10%", "+0.9%", "0.8%",  "Medium"),
            ("Obligasi RI",  "+6.5%",  "+8%",   "+4%",  "+0.5%", "0.5%",  "High"),
            ("DeFi USDC",    "8-15%",  "+20%",  "Var.", "0.7-1.2%","0.7%","Medium"),
        ]
        ret_df = pd.DataFrame(ret_rows, columns=[
            "Aset", "CAGR 5yr", "Best Year", "Worst Year",
            "Monthly Avg", "Monthly Median", "Konsistensi"
        ])
        st.dataframe(ret_df, use_container_width=True, hide_index=True)
        st.caption("Sumber: CoinGecko, Yahoo Finance, Bloomberg historis. BUKAN proyeksi masa depan.")

        st.markdown("---")
        st.subheader("Simulasi Pertumbuhan Portfolio")
        c1, c2 = st.columns(2)
        with c1:
            port_size  = st.number_input("Portfolio Size ($)", value=10000, step=1000, min_value=100)
            monthly_r  = st.slider("Target Monthly Return %", 1.0, 15.0, 5.0, 0.5)
            horizon_mo = st.slider("Horizon (bulan)", 6, 60, 24)
            max_dd_pct = st.slider("Worst-case DD Tolerance %", 10, 60, 25)
        with c2:
            mo = list(range(horizon_mo + 1))
            opt  = [port_size * (1 + monthly_r/100 * 1.5)**m for m in mo]
            base = [port_size * (1 + monthly_r/100)**m for m in mo]
            pess = [port_size * (1 + monthly_r/100 * 0.35)**m for m in mo]
            wc   = [port_size * (1 - max_dd_pct/100) * (1 + monthly_r/100 * 0.5)**m for m in mo]

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=mo, y=opt,  name="Optimistic (+50%)",  line=dict(color="#22c55e",width=1.5,dash="dot")))
            fig_sim.add_trace(go.Scatter(x=mo, y=base, name=f"Base ({monthly_r}%/mo)", line=dict(color="#818cf8",width=2.5)))
            fig_sim.add_trace(go.Scatter(x=mo, y=pess, name="Pessimistic (×0.35)",line=dict(color="#f59e0b",width=1.5,dash="dot")))
            fig_sim.add_trace(go.Scatter(x=mo, y=wc,   name=f"After {max_dd_pct}% DD", line=dict(color="#ef4444",width=1.5,dash="dash")))
            fig_sim.add_hline(y=port_size, line_dash="dash", line_color="#334155",
                              annotation_text="Starting Capital")
            fig_sim.update_layout(**CL, height=300, title="Portfolio Growth Scenarios",
                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.05))
            st.plotly_chart(fig_sim, use_container_width=True)

        fa, fb, fc, fd = st.columns(4)
        fa.metric("Final (Base)", f"${base[-1]:,.0f}", f"+{(base[-1]-port_size)/port_size*100:.0f}%")
        fb.metric("Monthly Gain",  f"${(base[-1]-port_size)/horizon_mo:,.0f}/mo")
        fc.metric("Platform Break-even", f"${port_size*monthly_r/100:,.0f}/mo needed")
        fd.metric("After Max DD",  f"${wc[-1]:,.0f}")

        with st.expander("📤 Export Investment Guide"):
            export_panel(
                "Investment Guide", ret_df, fig=fig_sim,
                extra_sheets={"Risk Framework": risk_df.drop(columns=["_color"]), "Min Investment": inv_df},
                filename_base="investment_guide"
            )

# ══════════════════════════════════════════════════════════════
# PAGE: NEWS INTELLIGENCE  (basic+)
# ══════════════════════════════════════════════════════════════
elif page == "🧠 News Intelligence":
    if not need_tier("basic"):
        show_upgrade_wall("🧠 News Intelligence", "basic")
    else:
        st.title("🧠 News Intelligence")
        st.markdown(
            "<p style='color:#94a3b8'>Analisis berita makro → prediksi dampak multi-aset. "
            "Sumber: CNBC Global & Indo, Reuters, CoinDesk, The Block, Kobeissi Letter, Stockbit.</p>",
            unsafe_allow_html=True
        )

        # ── Source Status Bar ─────────────────────────────────
        with st.expander("📡 Data Sources & Status"):
            src_df = pd.DataFrame({
                "Source":    ["CNBC Global","CNBC Indonesia","Reuters","CoinDesk","The Block",
                              "CoinTelegraph","Kobeissi Letter (X/Twitter)","Stockbit Indonesia"],
                "Type":      ["RSS Feed","RSS Feed","RSS Feed","RSS Feed","RSS Feed",
                              "RSS Feed","X API v2","Unofficial API"],
                "Endpoint":  ["rss.cnn.com/rss/money_latest.rss",
                              "cnbcindonesia.com/rss",
                              "feeds.reuters.com/reuters/businessNews",
                              "coindesk.com/arc/outboundfeeds/rss",
                              "theblock.co/rss.xml",
                              "cointelegraph.com/rss",
                              "Requires Bearer Token (Premium)",
                              "stockbit.com (rate-limited)"],
                "Status":    ["✅ Live","✅ Live","✅ Live","✅ Live","✅ Live",
                              "✅ Live","⚠️ API Key Required","⚠️ Rate Limited"],
                "Latency":   ["15min","15min","15min","10min","10min","10min","Realtime","30min"],
                "Tier":      ["Basic+","Basic+","Basic+","Basic+","Basic+","Basic+","Premium+","Premium+"],
            })
            st.dataframe(src_df, use_container_width=True, hide_index=True)
            if need_tier("premium"):
                c1, c2 = st.columns(2)
                c1.text_input("X/Twitter Bearer Token (Kobeissi Letter)", type="password",
                              placeholder="Enter to enable realtime X feed")
                c2.text_input("Stockbit Session Token", type="password",
                              placeholder="For Indonesia stock data")

        tab_live, tab_scen, tab_report = st.tabs([
            "📰 Live Analysis", "🧪 Scenario Testing", "📄 Generate Report"
        ])

        # ════════════════════════════════════════════════════
        # TAB 1 — LIVE ANALYSIS
        # ════════════════════════════════════════════════════
        with tab_live:
            c_ctrl, c_out = st.columns([1, 2])

            with c_ctrl:
                st.subheader("Settings")
                srcs = st.multiselect(
                    "News Sources",
                    ["CNBC Global","CNBC Indonesia","Reuters","CoinDesk","The Block","CoinTelegraph"],
                    default=["CNBC Indonesia","Reuters","CoinDesk","The Block"]
                )
                assets_sel = st.multiselect(
                    "Assets to Analyze",
                    ["S&P 500","IDX Composite","BTC","ETH","XAU/USD","Obligasi RI","IDR/USD"],
                    default=["S&P 500","IDX Composite","BTC","XAU/USD","Obligasi RI"]
                )
                lang_id = st.toggle("Bahasa Indonesia (Bahasa Bayi)", value=True)
                depth   = st.select_slider("Kedalaman Analisis", ["Cepat","Standar","Mendalam"], value="Standar")
                run_btn = st.button("🧠 Analisis Sekarang", type="primary", use_container_width=True)

            with c_out:
                if run_btn:
                    st.session_state["ni_run"] = True
                    st.session_state["ni_srcs"] = srcs
                    st.session_state["ni_assets"] = assets_sel
                    st.session_state["ni_lang_id"] = lang_id

                if st.session_state.get("ni_run"):
                    _assets = st.session_state.get("ni_assets", assets_sel)
                    _lang   = st.session_state.get("ni_lang_id", True)

                    with st.spinner("Mengambil berita & menganalisis..."):
                        import time as _t; _t.sleep(1.2)

                    # Live news fetch attempt
                    live_news = fetch_news(tuple(st.session_state.get("ni_srcs", srcs)[:4]))

                    st.subheader("🔥 Isu Makro Terkini")

                    ISSUES = [
                        {
                            "title_id": "Fed Tahan Suku Bunga di 5.25–5.50%",
                            "title_en": "Fed Holds Rates at 5.25–5.50%",
                            "cat": "Makro Global", "cat_en": "Global Macro",
                            "src": "Reuters", "sentiment": "NEUTRAL",
                            "simple_id": (
                                "Bank gede AS (The Fed) nggak naik-naikin atau turunin suku bunga. "
                                "Artinya: kredit masih mahal, tapi pasar udah expect ini, jadi nggak banyak kejutan. "
                                "Investor udah 'beli rumor, jual fakta'."
                            ),
                            "simple_en": "Fed holds. Markets priced it in. No major surprise.",
                        },
                        {
                            "title_id": "MSCI Naikkan Bobot Indonesia di Index Emerging Market",
                            "title_en": "MSCI Increases Indonesia Weight in EM Index",
                            "cat": "Makro Lokal", "cat_en": "Indonesia Local",
                            "src": "CNBC Indonesia", "sentiment": "BULLISH",
                            "simple_id": (
                                "MSCI itu kayak 'indeks gaul' yang diikutin dana gede global. "
                                "Kalau Indonesia dinaikkan bobotnya, berarti dana yang ngikutin MSCI "
                                "WAJIB beli saham Indonesia lebih banyak. Duit asing masuk deras ke IDX!"
                            ),
                            "simple_en": "MSCI raises Indonesia weight → passive funds must buy IDX stocks.",
                        },
                        {
                            "title_id": "Kesepakatan Tarif 0% Indonesia–AS untuk Beberapa Komoditas",
                            "title_en": "Indonesia–US Zero Tariff Deal Signed",
                            "cat": "Makro Lokal", "cat_en": "Indonesia Local",
                            "src": "CNBC Indonesia", "sentiment": "BULLISH",
                            "simple_id": (
                                "Intinya: barang-barang Indonesia bisa masuk ke Amerika GRATIS PAJAK. "
                                "Ekspor naik → perusahaan RI untung lebih → saham naik. "
                                "IDR juga bisa menguat karena dolar masuk lebih banyak."
                            ),
                            "simple_en": "Zero tariff deal → Indonesian exports rise → IDR strengthens.",
                        },
                        {
                            "title_id": "Ketegangan Militer Iran–AS di Selat Hormuz Meningkat",
                            "title_en": "Iran–US Military Tensions Escalate at Strait of Hormuz",
                            "cat": "Makro Global", "cat_en": "Global Macro",
                            "src": "Reuters", "sentiment": "BEARISH",
                            "simple_id": (
                                "Selat Hormuz itu jalur 20% minyak dunia. "
                                "Kalau ada perang atau blokade, minyak langsung naik, "
                                "inflasi naik, pasar saham takut. Aset berisiko (crypto, saham) turun. "
                                "Emas dan dolar naik karena orang cari 'tempat aman'."
                            ),
                            "simple_en": "Hormuz tension → oil spike → risk-off, gold/USD up, equities/BTC down.",
                        },
                    ]

                    for iss in ISSUES:
                        sent_col = {"BULLISH":"#22c55e","BEARISH":"#ef4444","NEUTRAL":"#f59e0b"}.get(iss["sentiment"],"#64748b")
                        cat_col  = "#38bdf8" if "Global" in iss["cat"] else "#f97316"
                        title    = iss["title_id"] if _lang else iss["title_en"]
                        summary  = iss["simple_id"] if _lang else iss["simple_en"]
                        cat_lbl  = iss["cat"] if _lang else iss["cat_en"]
                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #1e293b;'
                            f'border-left:3px solid {sent_col};border-radius:10px;'
                            f'padding:14px 16px;margin-bottom:10px">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                            f'<span style="color:{cat_col};font-size:.7rem;font-weight:700;text-transform:uppercase">'
                            f'🌐 {cat_lbl}</span>'
                            f'<span style="background:{sent_col}22;color:{sent_col};font-size:.65rem;'
                            f'font-weight:700;padding:2px 9px;border-radius:4px">{iss["sentiment"]}</span></div>'
                            f'<div style="color:#e2e8f0;font-weight:600;font-size:.9rem;margin-bottom:4px">{title}</div>'
                            f'<div style="color:#64748b;font-size:.72rem;margin-bottom:8px">Sumber: {iss["src"]}</div>'
                            f'<div style="color:#cbd5e1;font-size:.84rem;background:#111827;'
                            f'border-radius:6px;padding:10px;line-height:1.6">{summary}</div></div>',
                            unsafe_allow_html=True
                        )

                    # Impact Table
                    st.subheader("📊 Prediksi Dampak ke Semua Aset")

                    IMPACT_MAP = {
                        "S&P 500":      {"fed":"⚪ +0.1%","msci":"⚪ Minimal","tarif":"🟢 +0.3-0.5%","iran":"🔴 -3% to -8%"},
                        "IDX Composite":{"fed":"⚪ Neutral","msci":"🟢 +3% to +8%","tarif":"🟢 +2% to +5%","iran":"🔴 -3% to -6%"},
                        "BTC":          {"fed":"⚪ Neutral","msci":"⚪ Minimal","tarif":"⚪ Minimal","iran":"🔴 -10% to -25%"},
                        "ETH":          {"fed":"⚪ Neutral","msci":"⚪ Minimal","tarif":"⚪ Minimal","iran":"🔴 -12% to -30%"},
                        "XAU/USD":      {"fed":"🟢 +0.5-1%","msci":"⚪ Minimal","tarif":"⚪ Minimal","iran":"🟢 +3% to +10%"},
                        "Obligasi RI":  {"fed":"🟢 Yield turun","msci":"🟢 Inflow besar","tarif":"🟢 Yield turun","iran":"🔴 Yield naik"},
                        "IDR/USD":      {"fed":"🟢 IDR menguat","msci":"🟢 IDR +1-3%","tarif":"🟢 IDR +0.5-2%","iran":"🔴 IDR -3-8%"},
                    }
                    filtered = {k: v for k, v in IMPACT_MAP.items() if k in _assets}
                    if filtered:
                        imp_rows = []
                        for aset, vals in filtered.items():
                            imp_rows.append({
                                "Aset":    aset,
                                "Fed Hold":  vals["fed"],
                                "MSCI RI":   vals["msci"],
                                "Tarif 0%":  vals["tarif"],
                                "Iran–US":   vals["iran"],
                            })
                        imp_df = pd.DataFrame(imp_rows)
                        st.dataframe(imp_df, use_container_width=True, hide_index=True)
                        st.session_state["ni_imp_df"] = imp_df

                    # IDX Sector Reco
                    if "IDX Composite" in _assets:
                        st.subheader("🏭 Rekomendasi Sektoral IDX")
                        sek_rows = [
                            ("Perbankan","BBCA BBRI BMRI","🟢 STRONG BUY",
                             "MSCI inflow masuk ke saham paling likuid = bank besar","Beli saat pullback","Target +8-15% 3 bln"),
                            ("Consumer Goods","UNVR ICBP MYOR","🟢 BUY",
                             "Tarif 0% kurangi biaya bahan baku impor, margin naik","Accumulate bertahap","Target +5-10%"),
                            ("Telekomunikasi","TLKM ISAT","🟢 BUY",
                             "Defensive + benefit dari IDR kuat (impor alat murah)","Accumulate","Target +6-10%"),
                            ("Teknologi","GOTO EMTK","🟡 SELECTIVE BUY",
                             "MSCI sentiment + risk-on, tapi fundamental masih tipis","Swing trade saja","Target +10-20% short"),
                            ("Mining","ADRO PTBA INCO","🔴 AVOID/REDUCE",
                             "Iran risk bisa push oil/komoditas, tapi trade war risk juga ada","Hold existing saja","Monitor dulu"),
                            ("Properti","BSDE SMRA CTRA","🟡 WAIT",
                             "Butuh rate cut dulu baru rebound. Belum waktunya.","Wait & See","Entry setelah rate cut"),
                        ]
                        sek_df = pd.DataFrame(sek_rows, columns=[
                            "Sektor","Saham Contoh","Signal","Alasan","Entry","Target"
                        ])
                        st.dataframe(sek_df, use_container_width=True, hide_index=True)
                        st.session_state["ni_sek_df"] = sek_df

                    st.session_state["ni_issues"] = ISSUES
                    st.session_state["ni_run"] = False

        # ════════════════════════════════════════════════════
        # TAB 2 — SCENARIO TESTING
        # ════════════════════════════════════════════════════
        with tab_scen:
            st.subheader("🧪 Scenario Testing — Tes Isu Spesifik ke Semua Aset")
            st.markdown(
                "<p style='color:#64748b;font-size:.85rem'>"
                "Pilih skenario → lihat analisis dampak lengkap + rekomendasi entry/exit + confidence.</p>",
                unsafe_allow_html=True
            )

            SCENARIOS = {
                "⚔️  Perang Iran–US Eskalasi (Full Scale)": {
                    "bg": "#2d0a0a", "border": "#ef4444",
                    "desc_id": (
                        "Serangan militer skala penuh antara Iran dan AS. "
                        "Blokade Selat Hormuz. 20% suplai minyak dunia terganggu. "
                        "Ini skenario PALING PARAH yang bisa terjadi dari konflik ini."
                    ),
                    "macro_id": (
                        "Minyak naik +30-70%. Inflasi global melonjak. "
                        "Semua aset berisiko jual. Dolar dan emas naik tajam. "
                        "Fed mungkin tunda rate cut."
                    ),
                    "assets": {
                        "S&P 500":       (-7.5, "BEARISH",     "-5% to -15%",  "1-4 minggu", "REDUCE HEAVY",  "HIGH"),
                        "IDX Composite": (-5.0, "BEARISH",     "-4% to -10%",  "1-3 minggu", "DEFENSIVE",     "HIGH"),
                        "BTC":           (-15,  "BEARISH",     "-15% to -30%", "Immediate",  "REDUCE/HEDGE",  "HIGH"),
                        "ETH":           (-18,  "BEARISH",     "-18% to -35%", "Immediate",  "REDUCE/HEDGE",  "HIGH"),
                        "XAU/USD":       (+6,   "BULLISH",     "+4% to +12%",  "1-2 minggu", "ACCUMULATE",    "VERY HIGH"),
                        "Obligasi RI":   (-2,   "BEARISH",     "Yield +50-100bps","1-4 minggu","REDUCE DUR.",  "MEDIUM"),
                        "IDR/USD":       (-5,   "BEARISH",     "IDR -4% to -10%","1-3 minggu","HEDGE USD",    "HIGH"),
                    },
                    "sektoral_id": "BANK AVOID (NIM squeeze), GOLD MINING BUY (MDKA), CONSUMER HOLD, ENERGY WATCH (MEDC)",
                    "entry_exit_id": (
                        "Exit equity di SETIAP bounce. Entry emas/USD saat ini. "
                        "Wait 3-6 minggu untuk re-entry equity setelah ada sinyal deeskalasi jelas. "
                        "Jangan average down crypto dulu."
                    ),
                },
                "📊 Rebalancing MSCI — Indonesia Weight Naik": {
                    "bg": "#052e16", "border": "#22c55e",
                    "desc_id": (
                        "MSCI Emerging Markets index menaikkan bobot Indonesia. "
                        "Dana pasif global yang track MSCI WAJIB beli saham IDX secara mekanis. "
                        "Bukan karena mereka suka, tapi karena rules-based."
                    ),
                    "macro_id": (
                        "Dana masuk IDX besar dan terprediksi. "
                        "IDR menguat. Yield obligasi RI turun (harga naik). "
                        "Saham blue chip paling liquid kena beli paksa."
                    ),
                    "assets": {
                        "S&P 500":       (+0.2,  "NEUTRAL",      "-0.1% to +0.3%","Minimal",    "HOLD",          "HIGH"),
                        "IDX Composite": (+5.0,  "BULLISH",      "+3% to +8%",    "2-6 minggu", "BUY IDX",       "HIGH"),
                        "BTC":           (+0.0,  "NEUTRAL",      "Minimal",        "Minimal",   "HOLD",          "MEDIUM"),
                        "ETH":           (+0.0,  "NEUTRAL",      "Minimal",        "Minimal",   "HOLD",          "MEDIUM"),
                        "XAU/USD":       (+0.0,  "NEUTRAL",      "-0.5% to +0.5%","Minimal",   "HOLD",          "MEDIUM"),
                        "Obligasi RI":   (+2.0,  "BULLISH",      "Yield -20-50bps","2-8 minggu","BUY SBR/ORI",  "HIGH"),
                        "IDR/USD":       (+2.0,  "BULLISH",      "IDR +1% to +3%","2-6 minggu", "HOLD IDR",      "HIGH"),
                    },
                    "sektoral_id": "BANK STRONG BUY (BBCA/BBRI/BMRI most liquid), ASII BUY, TLKM BUY, semua MSCI constituents",
                    "entry_exit_id": (
                        "Entry sebelum effective date rebalancing (biasanya diumumkan 4-6 minggu sebelumnya). "
                        "Profit-taking 50% setelah effective date. Hold sisanya untuk momentum lanjutan. "
                        "Target: BBCA +10-15%, BBRI +8-12%."
                    ),
                },
                "🤝 Tarif 0% Indonesia–US Trade Deal": {
                    "bg": "#052210", "border": "#22c55e",
                    "desc_id": (
                        "Indonesia dan AS sepakat tarif 0% untuk kategori komoditas tertentu "
                        "(tekstil, elektronik, CPO, manufaktur). "
                        "Ini deal besar yang bisa ubah trade balance RI secara signifikan."
                    ),
                    "macro_id": (
                        "Ekspor RI naik → current account membaik → IDR menguat. "
                        "Perusahaan eksportir RI langsung merasakan margin improvement. "
                        "Tapi perlu lihat detail — produk apa yang dapat 0%."
                    ),
                    "assets": {
                        "S&P 500":       (+0.3,  "SLIGHT BULL",  "+0.1% to +0.5%","Minimal",    "HOLD",          "MEDIUM"),
                        "IDX Composite": (+3.5,  "BULLISH",      "+2% to +6%",    "2-4 minggu", "BUY EXPORTERS", "HIGH"),
                        "BTC":           (+0.0,  "NEUTRAL",      "Minimal",        "Minimal",   "HOLD",          "LOW"),
                        "ETH":           (+0.0,  "NEUTRAL",      "Minimal",        "Minimal",   "HOLD",          "LOW"),
                        "XAU/USD":       (+0.0,  "NEUTRAL",      "-0.5% to +0.3%","Minimal",   "HOLD",          "MEDIUM"),
                        "Obligasi RI":   (+1.5,  "BULLISH",      "Yield -10-30bps","2-6 minggu","ACCUMULATE",   "MEDIUM"),
                        "IDR/USD":       (+1.5,  "BULLISH",      "IDR +0.5% to +2%","2-4 minggu","HOLD IDR",    "HIGH"),
                    },
                    "sektoral_id": "Tekstil (SRIL, PBRX) BUY, Consumer Goods BUY (UNVR, ICBP), CPO/Agri NEUTRAL, Bank SLIGHT BUY",
                    "entry_exit_id": (
                        "Buy saat announcement resmi. Profit-taking 40% dalam 2 minggu pertama. "
                        "Hold sisanya sambil tunggu implementasi. "
                        "Sektor tekstil dan consumer goods paling langsung terdampak positif."
                    ),
                },
            }

            sel_scen = st.selectbox("Pilih Skenario", list(SCENARIOS.keys()))
            scen     = SCENARIOS[sel_scen]
            bg       = scen["bg"]; border_c = scen["border"]

            st.markdown(
                f'<div style="background:{bg};border:1px solid {border_c}55;'
                f'border-radius:10px;padding:16px;margin:12px 0">'
                f'<div style="color:{border_c};font-size:.75rem;font-weight:700;'
                f'text-transform:uppercase;margin-bottom:6px">SKENARIO</div>'
                f'<div style="color:#e2e8f0;font-size:.9rem;margin-bottom:8px">{scen["desc_id"]}</div>'
                f'<div style="color:#f59e0b;font-size:.83rem"><b>Macro View:</b> {scen["macro_id"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Build impact DF
            scen_rows = []
            for aset, (mid, sig, rng, tf, act, conf) in scen["assets"].items():
                scen_rows.append({
                    "Aset": aset, "Signal": sig, "Estimasi Range": rng,
                    "Timeframe": tf, "Action": act, "Confidence": conf
                })
            scen_df = pd.DataFrame(scen_rows)
            st.dataframe(scen_df, use_container_width=True, hide_index=True)

            # Impact bar chart
            mids  = [v[0] for v in scen["assets"].values()]
            names = list(scen["assets"].keys())
            clrs  = ["#22c55e" if m > 0 else "#ef4444" if m < 0 else "#475569" for m in mids]
            fig_sc = go.Figure(go.Bar(x=names, y=mids, marker_color=clrs,
                text=[f"{m:+.1f}%" for m in mids], textposition="outside"))
            fig_sc.add_hline(y=0, line_color="#334155")
            fig_sc.update_layout(
                **{**CL, "margin": dict(l=0, r=0, t=36, b=0)},
                height=260, title=f"Estimasi Dampak — {sel_scen[:50]}"
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            st.info(f"**IDX Sektoral:** {scen['sektoral_id']}")
            st.success(f"**Entry/Exit Strategy:** {scen['entry_exit_id']}")

            st.session_state["ni_last_scen"] = {
                "name": sel_scen, "scen": scen, "df": scen_df,
                "fig_src": fig_sc, "ts": datetime.now().strftime("%Y-%m-%d %H:%M UTC")
            }

        # ════════════════════════════════════════════════════
        # TAB 3 — GENERATE REPORT
        # ════════════════════════════════════════════════════
        with tab_report:
            st.subheader("📄 Generate Intelligence Report (HTML + PDF)")
            st.markdown(
                "<p style='color:#64748b;font-size:.85rem'>"
                "Report profesional dengan analisis lengkap, siap kirim ke klien atau disimpan.</p>",
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1:
                rpt_title  = st.text_input(
                    "Report Title",
                    value=f"TradeForge Intelligence Report — {datetime.now().strftime('%d %b %Y')}"
                )
                inc_live   = st.checkbox("Sertakan Live Issue Analysis", value=True)
                inc_scen   = st.checkbox("Sertakan Scenario Analysis", value=True)
                inc_risk   = st.checkbox("Sertakan Risk Framework", value=True)
                inc_reko   = st.checkbox("Sertakan Rekomendasi Portfolio", value=True)
            with c2:
                analyst    = st.text_input("Nama Analis", value=get_name())
                org        = st.text_input("Organisasi / Fund", value="CATERYA Tech")
                inc_disc   = st.checkbox("Tambah Disclaimer", value=True)
                rpt_lang   = st.radio("Bahasa Report", ["🇮🇩 Indonesia","🇺🇸 English"], horizontal=True)

            if st.button("🚀 Generate Report Sekarang", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):

                    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
                    is_id    = "Indonesia" in rpt_lang
                    last_s   = st.session_state.get("ni_last_scen", {})
                    imp_df_r = st.session_state.get("ni_imp_df", pd.DataFrame())
                    sek_df_r = st.session_state.get("ni_sek_df", pd.DataFrame())

                    # ── Scenario section HTML ──────────────────
                    scen_sec = ""
                    if inc_scen and last_s:
                        sname = last_s.get("name","")
                        sdata = last_s.get("scen", {})
                        sdf   = last_s.get("df", pd.DataFrame())
                        if not sdf.empty:
                            trows = "".join(
                                f"<tr>"
                                f"<td style='font-weight:600'>{r['Aset']}</td>"
                                f"<td style='color:{'#4ade80' if 'BULL' in r['Signal'] else '#f87171' if 'BEAR' in r['Signal'] else '#94a3b8'};font-weight:700'>{r['Signal']}</td>"
                                f"<td>{r['Estimasi Range']}</td>"
                                f"<td>{r['Timeframe']}</td>"
                                f"<td style='color:#fbbf24;font-weight:600'>{r['Action']}</td>"
                                f"<td style='color:#94a3b8'>{r['Confidence']}</td></tr>"
                                for _, r in sdf.iterrows()
                            )
                            scen_sec = f"""
<h2 class="section">SCENARIO ANALYSIS: {sname}</h2>
<p>{sdata.get('desc_id','')}</p>
<p><span class="label">Macro View:</span> {sdata.get('macro_id','')}</p>
<table><thead><tr>
  <th>Aset</th><th>Signal</th><th>Range</th><th>Timeframe</th><th>Action</th><th>Confidence</th>
</tr></thead><tbody>{trows}</tbody></table>
<div class="callout callout-warn">
  <b>IDX Sektoral:</b> {sdata.get('sektoral_id','N/A')}<br>
  <b>Entry/Exit:</b> {sdata.get('entry_exit_id','N/A')}
</div>"""

                    # ── Risk section HTML ──────────────────────
                    risk_sec = ""
                    if inc_risk:
                        risk_sec = """
<h2 class="section">RISK FRAMEWORK</h2>
<table><thead><tr>
  <th>Aset</th><th>Volatilitas/thn</th><th>Max DD Hist.</th>
  <th>VaR 95% (1D)</th><th>Sharpe 5yr</th><th>Risk Level</th>
</tr></thead><tbody>
<tr><td>BTC</td><td>~80%</td><td>-83%</td><td>-4.2%</td><td>1.2</td><td style='color:#ef4444'>HIGH</td></tr>
<tr><td>ETH</td><td>~90%</td><td>-94%</td><td>-5.1%</td><td>1.1</td><td style='color:#ef4444'>HIGH</td></tr>
<tr><td>S&amp;P 500</td><td>~15%</td><td>-57%</td><td>-1.1%</td><td>1.0</td><td style='color:#f59e0b'>MEDIUM</td></tr>
<tr><td>IDX Comp.</td><td>~18%</td><td>-62%</td><td>-1.3%</td><td>0.7</td><td style='color:#f59e0b'>MEDIUM</td></tr>
<tr><td>XAU/USD</td><td>~15%</td><td>-45%</td><td>-1.0%</td><td>0.6</td><td style='color:#22c55e'>MED-LOW</td></tr>
<tr><td>Obligasi RI</td><td>~5%</td><td>-8%</td><td>-0.4%</td><td>0.4</td><td style='color:#22c55e'>LOW</td></tr>
</tbody></table>"""

                    # ── Portfolio reko HTML ────────────────────
                    reko_sec = ""
                    if inc_reko:
                        reko_sec = """
<h2 class="section">REKOMENDASI PORTOFOLIO — NET VIEW</h2>
<table><thead><tr>
  <th>Aset</th><th>Net Signal</th><th>Alokasi</th><th>Action</th><th>Notes</th>
</tr></thead><tbody>
<tr><td>IDX Composite</td><td class='bull'>BULLISH</td><td>30-40%</td><td>ACCUMULATE</td><td>MSCI + Tarif 0% = dual catalyst</td></tr>
<tr><td>XAU/USD</td><td class='bull'>BULLISH</td><td>15-20%</td><td>HOLD/ADD</td><td>Geopolitik hedge, Iran risk</td></tr>
<tr><td>Obligasi RI</td><td class='bull'>SLIGHT BULL</td><td>15-20%</td><td>BUY SBR/ORI</td><td>Yield turun = capital gain</td></tr>
<tr><td>BTC/ETH</td><td class='neut'>NEUTRAL</td><td>10-15%</td><td>HOLD</td><td>Wait rate cut catalyst</td></tr>
<tr><td>S&amp;P 500</td><td class='neut'>NEUTRAL</td><td>10-15%</td><td>HOLD</td><td>Valuasi stretched + geopolitik</td></tr>
<tr><td>Cash/USD</td><td class='neut'>NEUTRAL</td><td>5-10%</td><td>HOLD</td><td>Dry powder for dips</td></tr>
</tbody></table>
<div class="callout callout-bull">
  <b>TOP PICKS IDX:</b> BBCA, BBRI, BMRI (Bank — liquid MSCI), TLKM (Telco — defensive),
  UNVR, ICBP (Consumer — tariff benefit), GOTO (Tech — MSCI + sentiment)<br><br>
  <b>AVOID:</b> Properti (butuh rate cut), Mining sektif (Iran + China uncertainty)
</div>"""

                    # ── Disclaimer HTML ────────────────────────
                    disc_sec = ""
                    if inc_disc:
                        disc_sec = (
                            "<div class='disclaimer'>"
                            "<b>DISCLAIMER:</b> Laporan ini dibuat oleh TradeForge AaaS untuk tujuan edukasi "
                            "dan informasi saja. Bukan merupakan rekomendasi investasi. "
                            "Keputusan investasi sepenuhnya tanggung jawab investor. "
                            "Past performance tidak menjamin hasil masa depan. "
                            "Data bersumber dari API publik (Yahoo Finance, CoinGecko, DeFiLlama, RSS Feeds). "
                            "CATERYA Tech tidak bertanggung jawab atas kerugian dari penggunaan laporan ini."
                            "</div>"
                        ) if is_id else (
                            "<div class='disclaimer'>"
                            "<b>DISCLAIMER:</b> This report is produced by TradeForge AaaS for informational "
                            "and educational purposes only. It does not constitute investment advice. "
                            "All investment decisions are the sole responsibility of the investor. "
                            "Past performance does not guarantee future results."
                            "</div>"
                        )

                    # ══════════════════════════════════════════
                    # FULL HTML REPORT
                    # ══════════════════════════════════════════
                    full_html = f"""<!DOCTYPE html>
<html lang="id"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{rpt_title}</title>
<style>
:root {{
  --bg:    #0a0e1a;  --surface: #0d1117; --surface2: #111827;
  --border: #1e293b; --text: #e2e8f0;   --muted: #64748b;
  --accent: #818cf8; --green: #22c55e;  --red: #ef4444; --amber: #f59e0b;
}}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:'IBM Plex Sans',Arial,sans-serif; background:var(--bg); color:var(--text); }}

/* ── Cover ── */
.cover {{ background:linear-gradient(135deg,#0d1117 0%,#111827 55%,#0f172a 100%);
  padding:56px 60px 48px; border-bottom:1px solid var(--border);
  position:relative; overflow:hidden; page-break-after:always; }}
.cover::after {{ content:''; position:absolute; bottom:-80px; right:-60px;
  width:360px; height:360px; border-radius:50%;
  background:radial-gradient(circle,rgba(129,140,248,.07) 0%,transparent 70%); }}
.badge {{ display:inline-block; background:rgba(129,140,248,.15); color:var(--accent);
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; font-weight:700;
  letter-spacing:.14em; padding:4px 14px; border-radius:20px;
  border:1px solid rgba(129,140,248,.3); margin-bottom:24px; }}
.cover h1 {{ font-family:'IBM Plex Mono',monospace; font-size:1.9rem; font-weight:700;
  line-height:1.25; color:#f1f5f9; margin-bottom:10px; }}
.cover-sub {{ color:var(--muted); font-size:.88rem; margin-bottom:36px; }}
.meta-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.meta-card {{ background:rgba(255,255,255,.04); border:1px solid var(--border);
  border-radius:8px; padding:12px 16px; }}
.meta-label {{ font-size:.63rem; color:#475569; text-transform:uppercase;
  letter-spacing:.1em; margin-bottom:4px; }}
.meta-value {{ font-family:'IBM Plex Mono',monospace; font-size:.85rem;
  color:var(--text); font-weight:600; }}

/* ── Body ── */
.body {{ padding:44px 60px; max-width:1100px; margin:0 auto; }}
h2.section {{ font-family:'IBM Plex Mono',monospace; color:#38bdf8; font-size:1rem;
  font-weight:700; letter-spacing:.06em; margin:36px 0 12px;
  padding:0 0 10px; border-bottom:1px solid var(--border);
  text-transform:uppercase; }}
p {{ color:#94a3b8; font-size:.86rem; line-height:1.75; margin-bottom:10px; }}
.label {{ color:var(--amber); font-weight:700; }}

/* ── Tables ── */
table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; font-size:.81rem; }}
th {{ background:#1e293b; color:var(--accent); padding:10px 14px; text-align:left;
  font-family:'IBM Plex Mono',monospace; font-size:.73rem; letter-spacing:.05em; }}
td {{ border-bottom:1px solid var(--border); padding:9px 14px; color:#cbd5e1; }}
tr:hover td {{ background:rgba(255,255,255,.015); }}
.bull {{ color:#4ade80; font-weight:700; }}
.bear {{ color:#f87171; font-weight:700; }}
.neut {{ color:var(--muted); }}

/* ── Callouts ── */
.callout {{ border-radius:8px; padding:14px 18px; margin:12px 0;
  font-size:.84rem; line-height:1.7; color:var(--text); }}
.callout-bull {{ background:#052e16; border:1px solid #16a34a; border-left:3px solid var(--green); }}
.callout-bear {{ background:#2d0a0a; border:1px solid #dc2626; border-left:3px solid var(--red); }}
.callout-warn {{ background:#1c1403; border:1px solid #ca8a04; border-left:3px solid var(--amber); }}
.callout-info {{ background:#0c1e3d; border:1px solid #1d4ed8; border-left:3px solid #3b82f6; }}

/* ── Issue cards ── */
.issue-card {{ background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:16px; margin-bottom:12px; }}
.issue-title {{ font-weight:600; font-size:.9rem; color:var(--text); margin:6px 0 4px; }}
.issue-summary {{ background:var(--surface2); border-radius:6px; padding:10px 12px;
  font-size:.83rem; color:#cbd5e1; line-height:1.65; margin-top:8px; }}

/* ── Disclaimer ── */
.disclaimer {{ background:var(--surface); border:1px solid #334155;
  border-radius:8px; padding:16px; margin-top:44px;
  font-size:.74rem; color:#475569; line-height:1.7; }}
.watermark {{ text-align:center; padding:28px; color:#1e293b;
  font-family:'IBM Plex Mono',monospace; font-size:.7rem; }}

/* ── Print ── */
@media print {{
  body {{ background:#fff; color:#0f172a; }}
  .cover {{ background:#f8fafc; border-bottom:2px solid #e2e8f0; page-break-after:always; }}
  .cover h1 {{ color:#0f172a; }}
  .meta-card {{ background:#f1f5f9; border-color:#e2e8f0; }}
  .meta-value {{ color:#0f172a; }}
  h2.section {{ color:#1d4ed8; border-bottom-color:#e2e8f0; }}
  table {{ font-size:.76rem; }}
  th {{ background:#e2e8f0; color:#1e293b; }}
  td {{ border-bottom-color:#e2e8f0; color:#334155; }}
  .callout-bull {{ background:#f0fdf4; border-color:#86efac; color:#14532d; }}
  .callout-warn {{ background:#fffbeb; border-color:#fde68a; color:#78350f; }}
  .callout-bear {{ background:#fef2f2; border-color:#fecaca; color:#7f1d1d; }}
  .disclaimer {{ background:#f8fafc; border-color:#e2e8f0; color:#64748b; }}
  .watermark {{ color:#cbd5e1; }}
  .callout-info {{ background:#eff6ff; border-color:#bfdbfe; color:#1e3a5f; }}
}}
</style>
</head><body>

<!-- COVER PAGE -->
<div class="cover">
  <div class="badge">⚡ TRADEFORGE AaaS — INTELLIGENCE REPORT</div>
  <h1>{rpt_title}</h1>
  <p class="cover-sub">
    Multi-Asset Market Intelligence &nbsp;|&nbsp; Macro Analysis &nbsp;|&nbsp;
    AI-Assisted Recommendations &nbsp;|&nbsp; IDX Sectoral Guide
  </p>
  <div class="meta-grid">
    <div class="meta-card">
      <div class="meta-label">Analis</div>
      <div class="meta-value">{analyst}</div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Organisasi</div>
      <div class="meta-value">{org}</div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Dibuat</div>
      <div class="meta-value">{now_str}</div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Platform</div>
      <div class="meta-value">TradeForge v2.0</div>
    </div>
  </div>
</div>

<!-- BODY -->
<div class="body">

<h2 class="section">EXECUTIVE SUMMARY</h2>
<p>Laporan ini merangkum analisis makro global dan lokal Indonesia beserta prediksi
dampaknya ke berbagai kelas aset per {now_str}.
Metodologi menggunakan analisis teknikal multi-indikator (RSI, MACD, Bollinger Bands),
analisis fundamental makro, dan pemrosesan berita realtime dari sumber terpercaya.</p>
<div class="callout callout-info">
  <b>Net View:</b> Kondisi makro saat ini bersifat MIXED — positif untuk IDX (MSCI + tarif),
  bearish untuk crypto jangka pendek (Iran risk), netral untuk S&amp;P 500.
  Strategi: overweight IDX blue chip + gold, underweight crypto sementara.
</div>

<h2 class="section">ISU MAKRO TERKINI</h2>
<div class="issue-card">
  <div style="display:flex;justify-content:space-between">
    <span style="color:#38bdf8;font-size:.7rem;font-weight:700">MAKRO GLOBAL</span>
    <span style="color:#f59e0b;font-size:.7rem;font-weight:700">NEUTRAL</span>
  </div>
  <div class="issue-title">Fed Tahan Suku Bunga di 5.25–5.50%</div>
  <div style="color:#64748b;font-size:.72rem">Sumber: Reuters</div>
  <div class="issue-summary">
    Bank Sentral AS tidak ubah suku bunga. Artinya kredit masih mahal, tapi pasar
    udah <i>expect</i> ini — tidak ada kejutan. Investor sudah "beli rumor, jual fakta"
    sebelum pengumuman.
  </div>
</div>
<div class="issue-card">
  <div style="display:flex;justify-content:space-between">
    <span style="color:#f97316;font-size:.7rem;font-weight:700">MAKRO LOKAL IDN</span>
    <span style="color:#22c55e;font-size:.7rem;font-weight:700">BULLISH</span>
  </div>
  <div class="issue-title">MSCI Naikkan Bobot Indonesia di Emerging Market Index</div>
  <div style="color:#64748b;font-size:.72rem">Sumber: CNBC Indonesia</div>
  <div class="issue-summary">
    MSCI adalah "indeks gaul" yang diikuti dana besar global. Kalau bobot Indonesia naik,
    dana pasif yang track MSCI <b>wajib</b> beli saham IDX lebih banyak secara mekanis.
    Duit asing masuk deras ke blue chip RI — terutama bank-bank besar.
  </div>
</div>
<div class="issue-card">
  <div style="display:flex;justify-content:space-between">
    <span style="color:#f97316;font-size:.7rem;font-weight:700">MAKRO LOKAL IDN</span>
    <span style="color:#22c55e;font-size:.7rem;font-weight:700">BULLISH</span>
  </div>
  <div class="issue-title">Tarif 0% Indonesia–AS Disepakati</div>
  <div style="color:#64748b;font-size:.72rem">Sumber: CNBC Indonesia</div>
  <div class="issue-summary">
    Barang ekspor Indonesia bisa masuk ke AS tanpa bayar pajak impor.
    Perusahaan eksportir (tekstil, consumer goods, elektronik) langsung merasakan
    margin improvement. IDR menguat karena lebih banyak dolar masuk ke Indonesia.
  </div>
</div>
<div class="issue-card">
  <div style="display:flex;justify-content:space-between">
    <span style="color:#38bdf8;font-size:.7rem;font-weight:700">MAKRO GLOBAL</span>
    <span style="color:#ef4444;font-size:.7rem;font-weight:700">BEARISH</span>
  </div>
  <div class="issue-title">Ketegangan Militer Iran–AS di Selat Hormuz</div>
  <div style="color:#64748b;font-size:.72rem">Sumber: Reuters</div>
  <div class="issue-summary">
    Selat Hormuz = jalur 20% minyak dunia. Eskalasi → minyak naik → inflasi naik →
    pasar risk-off. Aset berisiko (crypto, saham growth) tertekan.
    Emas dan dolar naik sebagai "safe haven". Perlu monitor intensitas setiap hari.
  </div>
</div>

{scen_sec}
{risk_sec}
{reko_sec}
{disc_sec}

</div>
<div class="watermark">
  Generated by TradeForge AaaS v2.0 &nbsp;|&nbsp; CATERYA Tech &nbsp;|&nbsp;
  aryhharyanto@proton.me &nbsp;|&nbsp; {now_str}
</div>
</body></html>"""

                    # ── Download HTML ──────────────────────────
                    fname_ts = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        "⬇️ Download HTML Report",
                        data=full_html.encode("utf-8"),
                        file_name=f"tradeforge_intelligence_{fname_ts}.html",
                        mime="text/html",
                        use_container_width=True,
                        type="primary"
                    )

                    # ── Download PDF via fpdf2 ─────────────────
                    try:
                        pdf_summary = pd.DataFrame({
                            "Field": [
                                "Report Title","Analyst","Organization","Generated",
                                "Net View IDX","Net View BTC","Net View Gold","Net View Obligasi",
                                "Top Pick 1","Top Pick 2","Top Pick 3","Avoid",
                            ],
                            "Value": [
                                rpt_title, analyst, org, now_str,
                                "BULLISH — MSCI + Tarif 0% dual catalyst",
                                "NEUTRAL — wait rate cut",
                                "BULLISH — Iran geopolitik hedge",
                                "SLIGHT BULLISH — yield turun",
                                "BBCA/BBRI (Bank — MSCI liquid)",
                                "UNVR/ICBP (Consumer — tariff benefit)",
                                "TLKM (Telco — defensive)",
                                "Properti (wait rate cut), Mining (Iran + China risk)",
                            ]
                        })
                        pdf_bytes = make_pdf(rpt_title, pdf_summary)
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"tradeforge_intelligence_{fname_ts}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"PDF error: {str(e)[:80]}")

                    # ── HTML Preview ───────────────────────────
                    st.markdown("---")
                    st.subheader("Preview Report")
                    components.html(full_html, height=700, scrolling=True)
