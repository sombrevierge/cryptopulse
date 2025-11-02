
import math
import time
from datetime import datetime, timezone
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="CryptoPulse v3", layout="wide")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFI_LLAMA_POOLS = "https://yields.llama.fi/pools"

SYMBOL_TO_ID = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "bnb": "binancecoin",
    "ada": "cardano",
    "xrp": "ripple",
    "dot": "polkadot",
    "matic": "matic-network",
    "avax": "avalanche-2",
    "doge": "dogecoin"
}

# =================== THEME / UX (black-tech) ===================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=Manrope:wght@400;600;800&display=swap');

:root {
  --bg: #0b0d0f;
  --panel: #101214;
  --panel-glass: rgba(255,255,255,0.04);
  --text: #ECECEC;
  --text-dim: #A5A7AD;
  --border: #1b1f24;
  --accent: #ff6b3d;
}

/* base */
html, body, [data-testid="stAppViewContainer"], .block-container { background: var(--bg) !important; color: var(--text) !important; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h1,h2,h3,h4,h5,h6 { font-family: 'Plus Jakarta Sans','Manrope',system-ui,sans-serif; letter-spacing:.2px; }
p, div, span, label { font-family: 'Manrope',system-ui,sans-serif; }
header, [data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 0 !important; box-shadow: none !important; z-index: 999 !important; }

/* sidebar */
section[data-testid="stSidebar"] > div { background: var(--bg) !important; border-right: 1px solid var(--border); color: var(--text) !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* dark inputs / selects */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] div, .stMultiSelect [data-baseweb="select"] div {
  background: #0b0d0f !important; color: #ECECEC !important; border: 1px solid #2a2f36 !important; border-radius: 10px !important;
}

/* dark dropdown menu */
ul[role="listbox"] { background: #0b0d0f !important; color: #ECECEC !important; border:1px solid #2a2f36 !important; }

/* selected tags dark-red */
div[data-baseweb="tag"] { background: #6f1b1b !important; color: #fff !important; border: 1px solid #a12c2c !important; border-radius: 10px !important; }
div[data-baseweb="tag"] svg { fill: #fff !important; }



/* hero */
.hero h1 { margin:0; font-weight:800; }
.hero p { margin:.4rem 0 0 0; color: var(--text-dim); }

/* cards */
.card { background: var(--panel-glass); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }
.metric { font-size: 26px; font-weight: 800; }
.metric-sub { color: var(--text-dim); font-size: 12px; }

/* tables contrast */
[data-testid="stStyledDataFrame"] { color: var(--text) !important; }
thead tr th { color: var(--text) !important; white-space: nowrap; }
tbody tr td { color: var(--text) !important; }

/* glass buttons */
.btn-glass > button { background: var(--panel-glass) !important; border:1px solid var(--border) !important; color: var(--text) !important; border-radius: 12px !important; }
.stDownloadButton > button { background: #ffffff !important; color: #000000 !important; border:1px solid #ffffff !important; border-radius: 12px !important; }
.stDownloadButton > button:hover { background: #000000 !important; color: #ffffff !important; border-color: #ffffff !important; }
.btn-glass > button:hover { background: rgba(255,255,255,0.07) !important; }

/* tooltips row */
.badges { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
.badges .b { font-size:12px; padding:4px 8px; border-radius:10px; border:1px solid var(--border); background: var(--panel-glass); }
.badges .b[title] { cursor: help; }
</style>
""", unsafe_allow_html=True)

# =================== helpers ===================
def human_number(x: float) -> str:
    try: x = float(x)
    except: return str(x)
    absx = abs(x)
    if absx >= 1_000_000_000: return f"{x/1_000_000_000:.1f}B"
    if absx >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if absx >= 1_000: return f"{x/1_000:.1f}K"
    return f"{x:.0f}"

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rolling_volatility(returns: pd.Series, window: int = 30, annualization: int = 365) -> pd.Series:
    return returns.rolling(window).std() * math.sqrt(annualization)

# =================== data ===================
@st.cache_data(show_spinner=False, ttl=600)
def get_market_chart_with_volume(coin_id: str, vs_currency: str, days: int, interval: str = "daily") -> pd.DataFrame:
    """prices + total_volumes (CoinGecko)."""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": interval}
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            prices = data.get("prices", [])
            vols = data.get("total_volumes", [])
            if not prices:
                raise RuntimeError("No prices")
            dfp = pd.DataFrame(prices, columns=["ts_ms","price"])
            dfp["ts"] = pd.to_datetime(dfp["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
            dfp = dfp.set_index("ts").drop(columns=["ts_ms"])
            if vols:
                dfv = pd.DataFrame(vols, columns=["ts_ms","volume"])
                dfv["ts"] = pd.to_datetime(dfv["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
                dfv = dfv.set_index("ts").drop(columns=["ts_ms"])
                df = dfp.join(dfv, how="left")
            else:
                df = dfp
                df["volume"] = np.nan
            return df
        elif r.status_code == 429:
            time.sleep(2 + attempt*2)
        else:
            time.sleep(1 + attempt)
    r.raise_for_status()

@st.cache_data(show_spinner=False, ttl=600)
def get_ohlc(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame:
    valid = [1,7,14,30,90,180,365, "max"]
    d = min(valid, key=lambda x: abs(x - days) if isinstance(x, int) else 10**9)
    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": d}
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            arr = r.json()
            if not arr: raise RuntimeError("No OHLC data")
            df = pd.DataFrame(arr, columns=["ts_ms","open","high","low","close"])
            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
            df = df.drop(columns=["ts_ms"]).set_index("ts")
            return df
        elif r.status_code == 429:
            time.sleep(2 + attempt*2)
        else:
            time.sleep(1 + attempt)
    r.raise_for_status()

@st.cache_data(show_spinner=False, ttl=900)
def get_llama_pools() -> pd.DataFrame:
    r = requests.get(DEFI_LLAMA_POOLS, timeout=60)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data: return pd.DataFrame()
    rows = []
    for it in data:
        rows.append({
            "project": it.get("project"),
            "chain": it.get("chain"),
            "symbol": it.get("symbol"),
            "pool": it.get("pool"),
            "apy": it.get("apy"),
            "apyBase": it.get("apyBase"),
            "apyReward": it.get("apyReward"),
            "tvlUsd": it.get("tvlUsd"),
            "ilRisk": it.get("ilRisk"),
            "exposure": it.get("exposure"),
        })
    df = pd.DataFrame(rows)
    for c in ["apy","apyBase","apyReward","tvlUsd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =================== Sidebar & Hero ===================
st.sidebar.title("CryptoPulse")
section = st.sidebar.radio("Раздел", ["Token Analytics", "DeFi Yields", "DeFi Selector", "Project Health", "Signals"], index=0)

st.markdown('<div class="hero"><h1>CryptoPulse</h1><p>Интерактивные свечи, DeFi-пулы, сигналы и здоровье протоколов. При добавлении монет требуется время на анализ.</p></div>', unsafe_allow_html=True)

# =================== Token Analytics ===================
def token_analytics():
    st.subheader("Token Analytics")
    st.caption("Свечи (Plotly) + объём, EMA20/50, RSI(14), волатильность 30д, корреляции")

    coins = st.sidebar.multiselect("Монеты", ["btc","eth","sol","bnb","ada","xrp","dot","matic","avax","doge"], default=["btc","eth","sol"])
    vs = st.sidebar.selectbox("Валютная пара", ["usd","eur"], index=0)
    days = st.sidebar.slider("Дней истории", 30, 365, 180, step=15)
    show_ema = st.sidebar.checkbox("EMA20/50", value=True)

    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div class="metric">{len(coins)}</div><div class="metric-sub">Выбрано монет</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div class="metric">{days}d</div><div class="metric-sub">Глубина истории</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div class="metric">{vs.upper()}</div><div class="metric-sub">Валюта</div></div>', unsafe_allow_html=True)

    tab_chart, tab_rsi, tab_vol, tab_corr = st.tabs(["График", "RSI(14)", "Волатильность (30д)", "Корреляции"])

    merged = []
    for c in coins:
        coin_id = SYMBOL_TO_ID.get(c.lower(), c.lower())
        try:
            # core series for indicators
            price_vol = get_market_chart_with_volume(coin_id, vs, days)
            ohlc = get_ohlc(coin_id, vs, days)
            # align daily index
            ohlc = ohlc.resample("1D").last().dropna()
            price_vol = price_vol.resample("1D").last().dropna()
            ohlc = ohlc.join(price_vol["volume"], how="left")

            # indicators (from close price)
            close = ohlc["close"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            rs = rsi(close, 14)
            ret = close.pct_change()
            vol30 = ret.rolling(30).std() * math.sqrt(365)

            with tab_chart:
                st.markdown(f"#### {c.upper()} — Свечной график (интерактив)")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                    row_heights=[0.75, 0.25])
                fig.add_trace(go.Candlestick(
                    x=ohlc.index,
                    open=ohlc['open'], high=ohlc['high'], low=ohlc['low'], close=ohlc['close'],
                    name='OHLC'
                ), row=1, col=1)
                if show_ema:
                    fig.add_trace(go.Scatter(x=ohlc.index, y=ema20, name="EMA20", mode="lines"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=ohlc.index, y=ema50, name="EMA50", mode="lines"), row=1, col=1)
                # volume
                fig.add_trace(go.Bar(x=ohlc.index, y=ohlc['volume'], name="Объём", opacity=0.5), row=2, col=1)

                fig.update_layout(
                    height=520,
                    plot_bgcolor="#101214",
                    paper_bgcolor="#0b0d0f",
                    font=dict(color="#ECECEC", family="Plus Jakarta Sans"),
                    hovermode="x unified",
                    xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikecolor="#666"),
                    xaxis2=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikecolor="#666"),
                    dragmode="pan",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_xaxes(gridcolor="#1b1f24", showline=True, linecolor="#1b1f24")
                fig.update_yaxes(gridcolor="#1b1f24", showline=True, linecolor="#1b1f24")

                st.plotly_chart(fig, use_container_width=True, theme=None)

            with tab_rsi:
                st.markdown(f"#### {c.upper()} — RSI(14)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ohlc.index, y=rs, name="RSI14", mode="lines"))
                fig2.add_hline(y=70, line_dash="dash")
                fig2.add_hline(y=30, line_dash="dash")
                fig2.update_layout(height=280, plot_bgcolor="#101214", paper_bgcolor="#0b0d0f",
                                   font=dict(color="#ECECEC", family="Plus Jakarta Sans"),
                                   hovermode="x unified")
                fig2.update_xaxes(gridcolor="#1b1f24"); fig2.update_yaxes(gridcolor="#1b1f24")
                st.plotly_chart(fig2, use_container_width=True, theme=None)

            with tab_vol:
                st.markdown(f"#### {c.upper()} — Волатильность (30д, годовая)")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=ohlc.index, y=vol30, name="Vol30", mode="lines"))
                fig3.update_layout(height=280, plot_bgcolor="#101214", paper_bgcolor="#0b0d0f",
                                   font=dict(color="#ECECEC", family="Plus Jakarta Sans"),
                                   hovermode="x unified")
                fig3.update_xaxes(gridcolor="#1b1f24"); fig3.update_yaxes(gridcolor="#1b1f24")
                st.plotly_chart(fig3, use_container_width=True, theme=None)

            merged.append(ret.rename(f"{c.upper()}_ret"))
        except Exception as e:
            st.warning(f"{c}: {e}")

    if merged:
        all_ret = pd.concat(merged, axis=1).dropna()
        with tab_corr:
            st.markdown("#### Корреляции доходностей")
            corr = all_ret.corr().round(3)
            st.dataframe(corr, use_container_width=True)
            st.download_button("Скачать correlations.csv", corr.to_csv().encode("utf-8"),
                               "correlations.csv", "text/csv", key="dlcorr")

# =================== DeFi Yields ===================
def defi_yields():
    st.subheader("DeFi Yields — YieldSense")
    st.caption("Источник: DefiLlama. Фильтры: сеть, проект, актив, мин. TVL и APY.")

    min_tvl = st.sidebar.number_input("Мин. TVL (USD)", min_value=0, value=100000, step=50000)
    min_apy = st.sidebar.number_input("Мин. APY (%)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
    symbol_query = st.sidebar.text_input("Фильтр по активу (symbol)", "")

    with st.spinner("Загрузка пулов..."):
        try:
            df = get_llama_pools()
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}"); st.stop()

    if df.empty: st.info("Нет данных из DefiLlama."); st.stop()

    chains = ["Все"] + sorted([c for c in df["chain"].dropna().unique()])
    projects = ["Все"] + sorted([p for p in df["project"].dropna().unique()])
    c1, c2, c3 = st.columns(3)
    with c1: chain = st.selectbox("Сеть", chains, index=0)
    with c2: project = st.selectbox("Проект", projects, index=0)
    with c3: pass

    filtered = df.copy()
    if chain != "Все": filtered = filtered[filtered["chain"] == chain]
    if project != "Все": filtered = filtered[filtered["project"] == project]
    if symbol_query: filtered = filtered[filtered["symbol"].fillna("").str.contains(symbol_query, case=False)]
    if min_tvl: filtered = filtered[filtered["tvlUsd"].fillna(0) >= float(min_tvl)]
    if min_apy: filtered = filtered[filtered["apy"].fillna(0) >= float(min_apy)]

    total_pools = int(len(filtered))
    top_apy = float(filtered["apy"].max()) if total_pools else 0.0
    total_tvl = float(filtered["tvlUsd"].sum()) if total_pools else 0.0

    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div class="metric">{total_pools:,}</div><div class="metric-sub">Пулы</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div class="metric">{top_apy:,.2f}%</div><div class="metric-sub">Макс. доходность</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div class="metric">{human_number(total_tvl)}</div><div class="metric-sub">Суммарный TVL</div></div>', unsafe_allow_html=True)

    # tooltips row
    st.markdown('<div class="badges">'
                '<span class="b" title="Суммарная ликвидность в USD.">TVL — Total Value Locked</span>'
                '<span class="b" title="Доходность без учёта наград токенами.">APY база</span>'
                '<span class="b" title="Доп. доходность за счёт токенов/инсентивов.">APY награды</span>'
                '<span class="b" title="Impermanent Loss — потенциальные потери при изменении цены активов в пуле.">Риск IL</span>'
                '<span class="b" title="Состав активов пула: один актив, несколько, или дельта‑нейтральная стратегия.">Экспозиция</span>'
                '</div>', unsafe_allow_html=True)

    view = filtered.sort_values(["apy","tvlUsd"], ascending=[False, False]).head(500).rename(columns={
        "project":"Проект","chain":"Сеть","symbol":"Актив","apy":"APY, %",
        "apyBase":"APY база, %","apyReward":"APY награды, %","tvlUsd":"TVL, $",
        "ilRisk":"Риск IL","exposure":"Экспозиция","pool":"Пул"
    })
    st.dataframe(view, use_container_width=True)
    st.download_button("Скачать выборку (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="defi_yields_selection.csv", mime="text/csv", key="dlyields")

# =================== DeFi Selector ===================
def defi_selector():
    st.subheader("DeFi Project Selector")
    st.caption("Рейтинг пулов по скору: APY, TVL, риск, доля reward (настраиваемые веса).")

    try: pools = get_llama_pools()
    except Exception as e: st.error(f"Ошибка: {e}"); st.stop()
    if pools.empty: st.info("Нет данных."); st.stop()

    df_chains = sorted([c for c in pools["chain"].dropna().unique()])
    df_projects = sorted([p for p in pools["project"].dropna().unique()])

    with st.sidebar.expander("Фильтры", expanded=True):
        min_tvl = st.number_input("Мин. TVL (USD)", min_value=0, value=200000, step=50000)
        min_apy = st.number_input("Мин. APY (%)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
        chains = st.multiselect("Сети", df_chains, default=[])
        projects = st.multiselect("Проекты", df_projects, default=[])
        stables_mode = st.selectbox("Стейблы", ["Все", "Только стейблы", "Без стейблов"], index=0)

    with st.sidebar.expander("Веса скоринга", expanded=True):
        w_apy = st.slider("Вес: APY", 0.0, 2.0, 1.0, 0.05)
        w_tvl = st.slider("Вес: TVL (log)", 0.0, 2.0, 0.6, 0.05)
        w_risk = st.slider("Штраф: риск", 0.0, 2.0, 0.8, 0.05)
        w_reward = st.slider("Вес: доля reward", 0.0, 2.0, 0.3, 0.05)

    pools["reward_share"] = (pools["apyReward"].fillna(0) / (pools["apy"].replace(0, np.nan))).fillna(0)
    pools["tvl_log"] = np.log1p(pools["tvlUsd"].clip(lower=0))
    STABLES = ["USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX", "LUSD", "USDP", "GUSD", "EURS"]
    pools["is_stable"] = pools["symbol"].fillna("").str.upper().str.contains("|".join(STABLES))

    sel = pools.copy()
    sel = sel[sel["tvlUsd"].fillna(0) >= float(min_tvl)]
    sel = sel[sel["apy"].fillna(0) >= float(min_apy)]
    if chains: sel = sel[sel["chain"].isin(chains)]
    if projects: sel = sel[sel["project"].isin(projects)]
    if stables_mode == "Только стейблы": sel = sel[sel["is_stable"]]
    elif stables_mode == "Без стейблов": sel = sel[~sel["is_stable"]]

    risk_map = {"low": 0.1, "medium": 0.5, "high": 1.0}
    exposure_map = {"single": 0.2, "multi": 0.4, "delta neutral": 0.1}
    sel["risk_penalty"] = sel["ilRisk"].map(risk_map).fillna(0.3) + sel["exposure"].map(exposure_map).fillna(0.3)

    def minmax(x):
        x = pd.to_numeric(x, errors="coerce")
        if x.count() == 0: return x
        mn, mx = x.min(), x.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn: return pd.Series(0.0, index=x.index)
        return (x - mn) / (mx - mn)

    sel["s_apy"] = minmax(sel["apy"]); sel["s_tvl"] = minmax(sel["tvl_log"]); sel["s_reward"] = minmax(sel["reward_share"])
    sel["score"] = w_apy*sel["s_apy"] + w_tvl*sel["s_tvl"] - w_risk*sel["risk_penalty"] + w_reward*sel["s_reward"]
    rank = sel.sort_values("score", ascending=False).reset_index(drop=True)

    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div class="metric">{len(rank):,}</div><div class="metric-sub">Кандидатов</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div class="metric">{rank["apy"].median():.2f}%</div><div class="metric-sub">APY медиана</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div class="metric">{human_number(rank["tvlUsd"].sum())}</div><div class="metric-sub">Суммарный TVL</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="badges">'
                '<span class="b" title="Суммарная ликвидность в USD.">TVL</span>'
                '<span class="b" title="Impermanent Loss — потери при изменении цен в пуле.">Риск IL</span>'
                '<span class="b" title="Состав активов: single/multi/delta‑neutral.">Экспозиция</span>'
                '<span class="b" title="Доля наград в общей доходности.">Reward share</span>'
                '</div>', unsafe_allow_html=True)

    view = rank[["score","project","chain","symbol","apy","apyBase","apyReward","tvlUsd","ilRisk","exposure","pool"]].rename(columns={
        "score":"Скор",
        "project":"Проект","chain":"Сеть","symbol":"Актив","apy":"APY, %",
        "apyBase":"APY база, %","apyReward":"APY награды, %","tvlUsd":"TVL, $",
        "ilRisk":"Риск IL","exposure":"Экспозиция","pool":"Пул"
    })
    st.dataframe(view.head(300), use_container_width=True)
    st.download_button("Скачать рейтинг (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "defi_selector_ranking.csv", "text/csv", key="dlselector")

# =================== Project Health ===================
def project_health():
    st.subheader("Project Health")
    st.caption("Флаги аномалий: экстремальный APY, дефицит ликвидности (TVL), риск IL/экспозиции.")

    try: df = get_llama_pools()
    except Exception as e: st.error(f"Ошибка: {e}"); st.stop()
    if df.empty: st.info("Нет данных."); st.stop()

    df["apy_z"] = (df["apy"] - df["apy"].median()) / (df["apy"].std(ddof=0) or 1)
    df["Флаг APY"] = (df["apy"] > df["apy"].quantile(0.99)) | (df["apy_z"] > 3)
    tvl_q10 = df["tvlUsd"].quantile(0.10)
    df["Флаг TVL"] = df["tvlUsd"] < tvl_q10
    df["Флаг риск"] = df["ilRisk"].isin(["high"]) | df["exposure"].isin(["multi"])

    df["Статус"] = np.select(
        [df["Флаг APY"] & df["Флаг риск"], df["Флаг APY"], df["Флаг TVL"] & df["Флаг риск"], df["Флаг TVL"]],
        ["критический (apy+risk)", "внимание (apy)", "внимание (tvl+risk)", "низкий TVL"],
        default="норм"
    )

    view = df.sort_values("apy", ascending=False).head(500).rename(columns={
        "project":"Проект","chain":"Сеть","symbol":"Актив","apy":"APY, %",
        "apyBase":"APY база, %","apyReward":"APY награды, %","tvlUsd":"TVL, $",
        "ilRisk":"Риск IL","exposure":"Экспозиция","pool":"Пул"
    })
    st.dataframe(view[["Статус","Проект","Сеть","Актив","APY, %","APY база, %","APY награды, %","TVL, $","Риск IL","Экспозиция","Пул"]], use_container_width=True)
    st.download_button("Скачать project_health.csv", view.to_csv(index=False).encode("utf-8"),
                       "project_health.csv", "text/csv", key="dlhealth")

# =================== Signals ===================
def signals():
    st.subheader("Signals")
    st.caption("Правила: EMA(20/50) cross, RSI(14) зоны. Экспорт таблицы.")

    coins = st.sidebar.multiselect("Монеты", ["btc","eth","sol","bnb","ada","xrp","dot","matic","avax","doge"], default=["btc","eth","sol"])
    vs = st.sidebar.selectbox("Валютная пара", ["usd","eur"], index=0)
    days = st.sidebar.slider("Дней истории", 30, 365, 180, step=15)

    rows = []
    for c in coins:
        try:
            coin_id = SYMBOL_TO_ID.get(c.lower(), c.lower())
            base = get_market_chart_with_volume(coin_id, vs, days).resample("1D").last().dropna()
            close = base["price"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            rs = rsi(close, 14)
            ret = close.pct_change()
            vol30 = ret.rolling(30).std() * math.sqrt(365)

            last = close.index[-1]
            ema_bull = ema20.iloc[-2] <= ema50.iloc[-2] and ema20.iloc[-1] > ema50.iloc[-1]
            ema_bear = ema20.iloc[-2] >= ema50.iloc[-2] and ema20.iloc[-1] < ema50.iloc[-1]
            rsi_buy = rs.iloc[-1] < 30; rsi_sell = rs.iloc[-1] > 70

            signal = []
            if ema_bull: signal.append("EMA Bull Cross")
            if ema_bear: signal.append("EMA Bear Cross")
            if rsi_buy: signal.append("RSI<30")
            if rsi_sell: signal.append("RSI>70")

            rows.append({"Монета": c.upper(), "Цена": round(close.iloc[-1], 4),
                         "EMA20": round(ema20.iloc[-1],4), "EMA50": round(ema50.iloc[-1],4),
                         "RSI14": round(rs.iloc[-1],2), "Vol30": round(vol30.iloc[-1],4) if not np.isnan(vol30.iloc[-1]) else None,
                         "Сигналы": ", ".join(signal) if signal else "—"})
        except Exception as e:
            rows.append({"Монета": c.upper(), "Сигналы": f"error: {e}"})

    sig_df = pd.DataFrame(rows)
    st.dataframe(sig_df, use_container_width=True)
    st.download_button("Скачать signals.csv", sig_df.to_csv(index=False).encode("utf-8"),
                       "signals.csv", "text/csv", key="dlsignals")

# Route
if section == "Token Analytics":
    token_analytics()
elif section == "DeFi Yields":
    defi_yields()
elif section == "DeFi Selector":
    defi_selector()
elif section == "Project Health":
    project_health()
elif section == "Signals":
    signals()
