"""
Institutional Equity Research, Risk & DCF Workbench
Single-file Streamlit application.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Apex Equity Research | DCF & Risk Workbench",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,18,28,0.6)",
    font=dict(family="IBM Plex Sans, Inter, sans-serif", color="#E8EDF7", size=13),
    margin=dict(l=40, r=24, t=48, b=40),
    coloraxis_colorbar=dict(outlinewidth=0),
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

      html, body, [class*="css"] {
        font-family: "IBM Plex Sans", Inter, sans-serif;
      }
      .stApp {
        background:
          radial-gradient(1200px 500px at 8% -10%, rgba(46, 92, 184, 0.18), transparent 55%),
          radial-gradient(900px 420px at 100% 0%, rgba(196, 154, 82, 0.10), transparent 50%),
          #0b0f17;
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101624 0%, #0c111b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
      }
      .hero-kicker {
        letter-spacing: 0.22em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: #9BB0D3;
        font-weight: 600;
        margin-bottom: 0.25rem;
      }
      .hero-title {
        font-family: "IBM Plex Serif", Georgia, serif;
        font-size: 2.15rem;
        font-weight: 600;
        color: #F4F7FB;
        margin: 0 0 0.35rem 0;
        line-height: 1.15;
      }
      .hero-sub {
        color: #A9B6CC;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
      }
      .panel {
        background: rgba(18, 24, 38, 0.72);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1rem 1.1rem 0.85rem 1.1rem;
      }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
      }
      div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
        color: #F4F7FB;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }
      .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #A9B6CC;
        border-radius: 8px 8px 0 0;
      }
      .stTabs [aria-selected="true"] {
        background: rgba(46, 92, 184, 0.18);
        color: #F4F7FB !important;
      }
      .footnote {
        color: #7E8BA3;
        font-size: 0.78rem;
        margin-top: 1.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_money(value: Optional[float], decimals: int = 2, prefix: str = "$") -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    abs_v = abs(value)
    sign = "-" if value < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}{prefix}{abs_v / 1e12:.{decimals}f}T"
    if abs_v >= 1e9:
        return f"{sign}{prefix}{abs_v / 1e9:.{decimals}f}B"
    if abs_v >= 1e6:
        return f"{sign}{prefix}{abs_v / 1e6:.{decimals}f}M"
    if abs_v >= 1e3:
        return f"{sign}{prefix}{abs_v / 1e3:.{decimals}f}K"
    return f"{sign}{prefix}{abs_v:,.{decimals}f}"


def fmt_num(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    return f"{value:.{decimals}f}%"


def ratio(num: float, den: float, default: float = np.nan) -> float:
    if den in (0, None) or (isinstance(den, float) and abs(den) < 1e-12):
        return default
    try:
        return float(num) / float(den)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def normalize_ticker(raw: str) -> str:
    return (raw or "").strip().upper()


def parse_peer_tickers(raw: str, primary: str) -> list[str]:
    peers: list[str] = []
    seen = {primary}
    for part in (raw or "").replace(";", ",").split(","):
        t = normalize_ticker(part)
        if t and t not in seen:
            peers.append(t)
            seen.add(t)
    return peers[:8]


def _norm_label(label: Any) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def get_row(
    df: Optional[pd.DataFrame],
    candidates: Iterable[str],
    col_idx: int = 0,
    default: float = 0.0,
) -> float:
    """Lookup a statement line with alias matching. Missing lines default safely."""
    if df is None or getattr(df, "empty", True):
        return default
    try:
        series_index = {_norm_label(i): i for i in df.index}
        ordered = list(candidates)
        for name in ordered:
            key = _norm_label(name)
            if key in series_index:
                val = df.loc[series_index[key]].iloc[col_idx]
                parsed = safe_float(val, default=np.nan)
                if not np.isnan(parsed):
                    return parsed
        for name in ordered:
            key = _norm_label(name)
            for idx_key, orig in series_index.items():
                if key in idx_key or idx_key in key:
                    val = df.loc[orig].iloc[col_idx]
                    parsed = safe_float(val, default=np.nan)
                    if not np.isnan(parsed):
                        return parsed
    except Exception:
        return default
    return default


def n_cols(df: Optional[pd.DataFrame]) -> int:
    if df is None or getattr(df, "empty", True):
        return 0
    return int(df.shape[1])


# ---------------------------------------------------------------------------
# Data fetch (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_ticker_bundle(ticker: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ticker": ticker,
        "ok": False,
        "error": None,
        "info": {},
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "price": np.nan,
    }
    try:
        obj = yf.Ticker(ticker)
        info: dict[str, Any] = {}
        try:
            info = obj.info or {}
        except Exception:
            try:
                info = obj.get_info() or {}
            except Exception:
                info = {}

        def _stmt(getter_names: list[str]) -> pd.DataFrame:
            for name in getter_names:
                try:
                    attr = getattr(obj, name, None)
                    df = attr() if callable(attr) else attr
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df
                except Exception:
                    continue
            return pd.DataFrame()

        income = _stmt(["income_stmt", "financials", "get_income_stmt"])
        balance = _stmt(["balance_sheet", "get_balance_sheet"])
        cashflow = _stmt(["cashflow", "cash_flow", "get_cash_flow"])

        price = safe_float(info.get("currentPrice"), np.nan)
        if np.isnan(price):
            price = safe_float(info.get("regularMarketPrice"), np.nan)
        if np.isnan(price):
            price = safe_float(info.get("previousClose"), np.nan)
        if np.isnan(price):
            try:
                hist = obj.history(period="5d")
                if hist is not None and not hist.empty:
                    price = safe_float(hist["Close"].iloc[-1], np.nan)
            except Exception:
                pass

        payload.update(
            {
                "ok": True,
                "info": info,
                "income": income,
                "balance": balance,
                "cashflow": cashflow,
                "price": price,
            }
        )
        return payload
    except Exception as exc:
        payload["error"] = str(exc)
        return payload


# ---------------------------------------------------------------------------
# Financial extractors
# ---------------------------------------------------------------------------
INCOME_ALIASES = {
    "revenue": [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
        "TotalRevenue",
        "Net Sales",
    ],
    "ebit": [
        "EBIT",
        "Operating Income",
        "OperatingIncome",
        "Ebit",
        "Operating Profit",
    ],
    "ebt": [
        "Pretax Income",
        "Income Before Tax",
        "EBT",
        "PretaxIncome",
        "Earnings Before Tax",
    ],
    "ni": [
        "Net Income",
        "Net Income Common Stockholders",
        "NetIncome",
        "Net Income Including Noncontrolling Interests",
    ],
    "cogs": [
        "Cost Of Revenue",
        "Cost Of Goods Sold",
        "Reconciled Cost Of Revenue",
        "CostOfRevenue",
    ],
    "rd": [
        "Research And Development",
        "Research Development",
        "ResearchAndDevelopment",
        "R&D",
    ],
    "gp": [
        "Gross Profit",
        "GrossProfit",
    ],
}

BS_ALIASES = {
    "assets": ["Total Assets", "TotalAssets"],
    "equity": [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
        "StockholdersEquity",
    ],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "CashAndCashEquivalents",
        "Cash Financial",
        "Cash",
    ],
    "debt": [
        "Total Debt",
        "TotalDebt",
        "Net Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
    ],
    "ltd": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
        "LongTermDebt",
        "Total Debt",
    ],
    "ca": ["Current Assets", "CurrentAssets", "Total Current Assets"],
    "cl": ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities"],
    "inventory": ["Inventory", "Inventories"],
    "ar": [
        "Accounts Receivable",
        "Receivables",
        "AccountsReceivable",
        "Gross Accounts Receivable",
    ],
    "ap": [
        "Accounts Payable",
        "Payables And Accrued Expenses",
        "AccountsPayable",
        "Payables",
    ],
    "re": ["Retained Earnings", "RetainedEarnings"],
    "liab": [
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
        "TotalLiabilitiesNetMinorityInterest",
        "Total Liabilities And Stockholders Equity",
    ],
    "shares": [
        "Ordinary Shares Number",
        "Share Issued",
        "Common Stock Shares Outstanding",
        "OrdinarySharesNumber",
    ],
}

CF_ALIASES = {
    "fcf": ["Free Cash Flow", "FreeCashFlow"],
    "ocf": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
        "OperatingCashFlow",
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
        "CapitalExpenditure",
    ],
    "sbc": [
        "Stock Based Compensation",
        "StockBasedCompensation",
        "Share Based Compensation",
    ],
    "issuance": [
        "Issuance Of Capital Stock",
        "Common Stock Issuance",
        "Issuance Of Common Stock",
        "Sale Of Stock",
        "IssuanceOfCapitalStock",
    ],
}


@dataclass
class Fundamentals:
    ticker: str
    price: float
    info: dict[str, Any]
    fcf: float
    debt: float
    cash: float
    shares: float
    ebit: float
    ebt: float
    ni: float
    revenue: float
    cogs: float
    rd: float
    gp: float
    sbc: float
    ocf: float
    assets: float
    equity: float
    ca: float
    cl: float
    inventory: float
    ar: float
    ap: float
    re: float
    liab: float
    ltd: float
    issuance: float
    # prior year
    ni_py: float
    ocf_py: float
    assets_py: float
    revenue_py: float
    gp_py: float
    ltd_py: float
    ca_py: float
    cl_py: float
    shares_bs: float
    shares_bs_py: float
    missing: list[str]


def extract_fundamentals(bundle: dict[str, Any]) -> Fundamentals:
    info = bundle.get("info") or {}
    income: pd.DataFrame = bundle.get("income") if isinstance(bundle.get("income"), pd.DataFrame) else pd.DataFrame()
    balance: pd.DataFrame = bundle.get("balance") if isinstance(bundle.get("balance"), pd.DataFrame) else pd.DataFrame()
    cashflow: pd.DataFrame = bundle.get("cashflow") if isinstance(bundle.get("cashflow"), pd.DataFrame) else pd.DataFrame()
    missing: list[str] = []

    def pick(df, aliases, label, col=0, info_keys=None, default=0.0):
        val = get_row(df, aliases, col_idx=col, default=np.nan)
        if np.isnan(val) and info_keys:
            for k in info_keys:
                val = safe_float(info.get(k), np.nan)
                if not np.isnan(val):
                    break
        if np.isnan(val):
            missing.append(label)
            return default
        return val

    fcf = pick(cashflow, CF_ALIASES["fcf"], "Free Cash Flow", info_keys=["freeCashflow"])
    if fcf == 0:
        ocf_try = get_row(cashflow, CF_ALIASES["ocf"], 0, np.nan)
        capex_try = get_row(cashflow, CF_ALIASES["capex"], 0, np.nan)
        if not np.isnan(ocf_try) and not np.isnan(capex_try):
            # Capex is typically negative in yfinance
            fcf = ocf_try + capex_try if capex_try < 0 else ocf_try - abs(capex_try)
            if "Free Cash Flow" in missing:
                missing.remove("Free Cash Flow")

    debt = pick(balance, BS_ALIASES["debt"], "Total Debt", info_keys=["totalDebt"])
    cash = pick(
        balance,
        BS_ALIASES["cash"],
        "Cash",
        info_keys=["totalCash", "cash"],
    )
    shares = safe_float(info.get("sharesOutstanding"), np.nan)
    if np.isnan(shares) or shares <= 0:
        shares = pick(balance, BS_ALIASES["shares"], "Shares Outstanding", default=np.nan)
    if np.isnan(shares) or shares <= 0:
        missing.append("Shares Outstanding")
        shares = 0.0

    has_py = n_cols(income) >= 2 and n_cols(balance) >= 2 and n_cols(cashflow) >= 2

    return Fundamentals(
        ticker=bundle.get("ticker", ""),
        price=safe_float(bundle.get("price"), np.nan),
        info=info,
        fcf=fcf,
        debt=debt,
        cash=cash,
        shares=shares,
        ebit=pick(income, INCOME_ALIASES["ebit"], "EBIT", info_keys=["ebit"]),
        ebt=pick(income, INCOME_ALIASES["ebt"], "EBT"),
        ni=pick(income, INCOME_ALIASES["ni"], "Net Income", info_keys=["netIncomeToCommon"]),
        revenue=pick(income, INCOME_ALIASES["revenue"], "Revenue", info_keys=["totalRevenue"]),
        cogs=pick(income, INCOME_ALIASES["cogs"], "COGS"),
        rd=pick(income, INCOME_ALIASES["rd"], "R&D"),
        gp=pick(income, INCOME_ALIASES["gp"], "Gross Profit", info_keys=["grossProfits"]),
        sbc=pick(cashflow, CF_ALIASES["sbc"], "SBC"),
        ocf=pick(cashflow, CF_ALIASES["ocf"], "Operating Cash Flow", info_keys=["operatingCashflow"]),
        assets=pick(balance, BS_ALIASES["assets"], "Total Assets"),
        equity=pick(balance, BS_ALIASES["equity"], "Equity"),
        ca=pick(balance, BS_ALIASES["ca"], "Current Assets"),
        cl=pick(balance, BS_ALIASES["cl"], "Current Liabilities"),
        inventory=pick(balance, BS_ALIASES["inventory"], "Inventory"),
        ar=pick(balance, BS_ALIASES["ar"], "Accounts Receivable"),
        ap=pick(balance, BS_ALIASES["ap"], "Accounts Payable"),
        re=pick(balance, BS_ALIASES["re"], "Retained Earnings"),
        liab=pick(balance, BS_ALIASES["liab"], "Total Liabilities"),
        ltd=pick(balance, BS_ALIASES["ltd"], "Long-Term Debt"),
        issuance=pick(cashflow, CF_ALIASES["issuance"], "Share Issuance"),
        ni_py=get_row(income, INCOME_ALIASES["ni"], 1, 0.0) if has_py or n_cols(income) >= 2 else 0.0,
        ocf_py=get_row(cashflow, CF_ALIASES["ocf"], 1, 0.0) if n_cols(cashflow) >= 2 else 0.0,
        assets_py=get_row(balance, BS_ALIASES["assets"], 1, 0.0) if n_cols(balance) >= 2 else 0.0,
        revenue_py=get_row(income, INCOME_ALIASES["revenue"], 1, 0.0) if n_cols(income) >= 2 else 0.0,
        gp_py=get_row(income, INCOME_ALIASES["gp"], 1, 0.0) if n_cols(income) >= 2 else 0.0,
        ltd_py=get_row(balance, BS_ALIASES["ltd"], 1, 0.0) if n_cols(balance) >= 2 else 0.0,
        ca_py=get_row(balance, BS_ALIASES["ca"], 1, 0.0) if n_cols(balance) >= 2 else 0.0,
        cl_py=get_row(balance, BS_ALIASES["cl"], 1, 0.0) if n_cols(balance) >= 2 else 0.0,
        shares_bs=get_row(balance, BS_ALIASES["shares"], 0, 0.0),
        shares_bs_py=get_row(balance, BS_ALIASES["shares"], 1, 0.0) if n_cols(balance) >= 2 else 0.0,
        missing=sorted(set(missing)),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def dcf_equity_per_share(
    fcf0: float,
    wacc: float,
    g: float,
    debt: float,
    cash: float,
    shares: float,
    years: int = 5,
) -> tuple[float, float, pd.DataFrame]:
    """Gordon-growth 5-year DCF. Returns (equity/share, enterprise value, schedule)."""
    if shares <= 0 or wacc <= g or fcf0 == 0:
        return np.nan, np.nan, pd.DataFrame()

    rows = []
    pv_fcf = 0.0
    fcf = fcf0
    for t in range(1, years + 1):
        fcf = fcf0 * ((1 + g) ** t)
        df = (1 + wacc) ** t
        pv = fcf / df
        pv_fcf += pv
        rows.append({"Year": t, "FCF": fcf, "Discount Factor": 1 / df, "PV of FCF": pv})

    tv = fcf * (1 + g) / (wacc - g)
    pv_tv = tv / ((1 + wacc) ** years)
    ev = pv_fcf + pv_tv
    equity = ev - debt + cash
    per_share = equity / shares
    schedule = pd.DataFrame(rows)
    schedule.loc[len(schedule)] = {
        "Year": "TV",
        "FCF": tv,
        "Discount Factor": 1 / ((1 + wacc) ** years),
        "PV of FCF": pv_tv,
    }
    return per_share, ev, schedule


def dcf_grid(fcf0, debt, cash, shares, wacc_center, g_center) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waccs = np.round(np.clip(np.linspace(wacc_center - 0.03, wacc_center + 0.03, 7), 0.04, 0.20), 4)
    gs = np.round(np.clip(np.linspace(g_center - 0.012, g_center + 0.012, 7), 0.005, 0.05), 4)
    # Ensure uniqueness and WACC > g for every cell
    waccs = np.unique(waccs)
    gs = np.unique(gs)
    grid = np.full((len(waccs), len(gs)), np.nan)
    for i, w in enumerate(waccs):
        for j, g in enumerate(gs):
            if w <= g + 0.0025:
                continue
            ps, _, _ = dcf_equity_per_share(fcf0, float(w), float(g), debt, cash, shares)
            grid[i, j] = ps
    return waccs, gs, grid


def z_interpretation(z: float) -> tuple[str, str]:
    if np.isnan(z):
        return "N/A", "Insufficient data to compute Altman Z-Score."
    if z > 2.99:
        return "Safe Zone", "Z > 2.99 — historically associated with a low near-term bankruptcy probability."
    if z >= 1.81:
        return "Grey Zone", "1.81 ≤ Z ≤ 2.99 — mixed credit signal; warrants closer monitoring."
    return "Distress Zone", "Z < 1.81 — historically associated with elevated financial distress risk."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ◆ Apex Research")
    st.caption("Institutional DCF • Risk • Peer Workbench")
    st.divider()

    ticker_in = st.text_input("Primary ticker", value="AAPL", help="US-listed equity ticker.")
    ticker = normalize_ticker(ticker_in) or "AAPL"

    st.markdown("#### DCF assumptions")
    wacc_pct = st.slider("Base WACC", min_value=5.0, max_value=15.0, value=8.5, step=0.1, format="%.1f%%")
    g_pct = st.slider(
        "Terminal growth rate (g)",
        min_value=1.0,
        max_value=4.0,
        value=2.5,
        step=0.1,
        format="%.1f%%",
    )
    wacc = wacc_pct / 100.0
    g = g_pct / 100.0
    if wacc <= g:
        st.error("WACC must exceed terminal growth (g). Raise WACC or lower g.")

    peers_raw = st.text_input(
        "Peer tickers",
        value="MSFT, GOOG, NVDA",
        help="Comma-separated. Used in the Peer Multiples tab.",
    )
    peers = parse_peer_tickers(peers_raw, ticker)

    st.divider()
    refresh = st.button("Refresh market data", use_container_width=True)
    if refresh:
        fetch_ticker_bundle.clear()
        st.rerun()

    st.caption("Data via Yahoo Finance (yfinance). Statements differ by sector; missing lines default to 0 / N/A.")


# ---------------------------------------------------------------------------
# Load primary
# ---------------------------------------------------------------------------
with st.spinner(f"Loading {ticker} fundamentals…"):
    bundle = fetch_ticker_bundle(ticker)

st.markdown(
    f"""
    <div class="hero-kicker">Equity Research Desk</div>
    <h1 class="hero-title">{ticker} — Valuation, Quality & Credit</h1>
    <p class="hero-sub">Five-year Gordon DCF, earnings normalization, working-capital cycle, sell-side consensus, peer multiples, DuPont, Altman Z and Piotroski F.</p>
    """,
    unsafe_allow_html=True,
)

if not bundle.get("ok"):
    st.error(f"Could not load {ticker}: {bundle.get('error') or 'unknown yfinance error'}.")
    st.stop()

fund = extract_fundamentals(bundle)
info = fund.info
name = info.get("longName") or info.get("shortName") or ticker
sector = info.get("sector") or "N/A"
industry = info.get("industry") or "N/A"
currency = info.get("currency") or "USD"

if fund.missing:
    st.info(
        "Some accounting lines were unavailable for this issuer (common for banks, REITs, and non-US filers). "
        f"Defaulted to 0 / N/A: {', '.join(fund.missing[:18])}"
        + ("…" if len(fund.missing) > 18 else "")
        + "."
    )

# ---------------------------------------------------------------------------
# Core valuation
# ---------------------------------------------------------------------------
iv, ev, schedule = dcf_equity_per_share(fund.fcf, wacc, g, fund.debt, fund.cash, fund.shares)
spot = fund.price
upside = ratio(iv - spot, spot) * 100 if not (np.isnan(iv) or np.isnan(spot)) else np.nan
mkt_cap = safe_float(info.get("marketCap"), np.nan)
if np.isnan(mkt_cap) and fund.shares > 0 and not np.isnan(spot):
    mkt_cap = spot * fund.shares

meta_l, meta_r = st.columns([2.2, 1])
with meta_l:
    st.markdown(f"**{name}** · {sector} · {industry} · Reporting currency {currency}")
with meta_r:
    st.caption("Last annual statements via Yahoo Finance. Model is illustrative, not investment advice.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last reported FCF", fmt_money(fund.fcf))
c2.metric("Total debt", fmt_money(fund.debt))
c3.metric("Cash & equivalents", fmt_money(fund.cash))
c4.metric("Shares outstanding", fmt_num(fund.shares / 1e6, 1, "M") if fund.shares else "N/A")
c5.metric("Market cap", fmt_money(mkt_cap, 1))

v1, v2, v3, v4 = st.columns(4)
v1.metric("Spot price", fmt_money(spot) if not np.isnan(spot) else "N/A")
v2.metric(
    "Implied value / share",
    fmt_money(iv) if not np.isnan(iv) else "N/A",
    delta=None if np.isnan(upside) else f"{upside:+.1f}% vs spot",
)
delta_label = "Overvalued vs DCF" if (not np.isnan(upside) and upside < 0) else "Undervalued vs DCF"
v3.metric("WACC / g", f"{wacc_pct:.1f}% / {g_pct:.1f}%")
v4.metric(
    "Enterprise value (DCF)",
    fmt_money(ev) if not np.isnan(ev) else "N/A",
)

if np.isnan(iv):
    st.warning(
        "DCF could not be computed. Typical causes: missing FCF, missing share count, or WACC ≤ g. "
        "Check the sidebar and the issuer’s cash-flow statement."
    )
elif not schedule.empty:
    with st.expander("DCF cash-flow schedule (5-year explicit + terminal value)", expanded=False):
        show = schedule.copy()
        for col in ["FCF", "PV of FCF"]:
            show[col] = show[col].map(lambda x: fmt_money(x))
        show["Discount Factor"] = show["Discount Factor"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "FCF is grown at terminal g through the explicit period (conservative simple model). "
            "TV = FCF₅ × (1+g) / (WACC − g). Equity = EV − Debt + Cash."
        )


# ---------------------------------------------------------------------------
# Research tabs
# ---------------------------------------------------------------------------
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
    [
        "1 · Earnings Normalization",
        "2 · Liquidity & CCC",
        "3 · Sell-Side Consensus",
        "4 · Valuation Matrix",
        "5 · Peer Multiples",
        "6 · 5-Step DuPont",
        "7 · Altman Z-Score",
        "8 · Piotroski F-Score",
    ]
)

# ---- Tab 1 ----
with t1:
    st.subheader("GAAP EBIT → Normalized EBIT")
    st.caption(
        "Stock-based compensation is treated as a cash operating expense (deducted). "
        "R&D is capitalized in-year and amortized on a 1-year linear schedule at 33%."
    )
    sbc = fund.sbc
    rd = fund.rd
    rd_amort = 0.33 * rd
    # Bridge: start GAAP EBIT, deduct SBC, add back R&D, subtract amortization
    norm_ebit = fund.ebit - sbc + rd - rd_amort

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("GAAP EBIT / operating income", fmt_money(fund.ebit))
    b2.metric("(-) Stock-based compensation", fmt_money(sbc))
    b3.metric("(+) R&D add-back", fmt_money(rd))
    b4.metric("(-) R&D amortization (33%)", fmt_money(rd_amort))
    b5.metric("Normalized EBIT", fmt_money(norm_ebit), delta=fmt_money(norm_ebit - fund.ebit))

    if sbc == 0:
        st.info("SBC was not found on the cash-flow statement; treated as 0.")
    if rd == 0:
        st.info("R&D was not found on the income statement (typical for banks, insurers, retailers); treated as 0.")

    bridge = pd.DataFrame(
        {
            "Step": [
                "GAAP EBIT",
                "Less: SBC (cash expense)",
                "Plus: R&D capitalization",
                "Less: 33% 1-year R&D amortization",
                "Normalized EBIT",
            ],
            "Amount": [fund.ebit, -sbc, rd, -rd_amort, norm_ebit],
        }
    )
    fig_bridge = go.Figure(
        go.Waterfall(
            name="Bridge",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=bridge["Step"],
            y=bridge["Amount"],
            connector={"line": {"color": "rgba(255,255,255,0.25)"}},
            decreasing={"marker": {"color": "#C45C5C"}},
            increasing={"marker": {"color": "#3D9B7A"}},
            totals={"marker": {"color": "#2E5CB8"}},
            text=[fmt_money(v, 1) for v in bridge["Amount"]],
            textposition="outside",
        )
    )
    fig_bridge.update_layout(
        **PLOTLY_LAYOUT,
        title="Normalization bridge",
        showlegend=False,
        height=420,
        yaxis_title=f"Amount ({currency})",
    )
    st.plotly_chart(fig_bridge, use_container_width=True)

# ---- Tab 2 ----
with t2:
    st.subheader("Working-capital cycle")
    st.caption("DSO = AR / Revenue × 365 · DIO = Inventory / COGS × 365 · DPO = AP / COGS × 365 · CCC = DSO + DIO − DPO")

    dso = ratio(fund.ar, fund.revenue) * 365 if fund.revenue else np.nan
    dio = ratio(fund.inventory, abs(fund.cogs)) * 365 if fund.cogs else np.nan
    dpo = ratio(fund.ap, abs(fund.cogs)) * 365 if fund.cogs else np.nan
    if fund.inventory == 0:
        dio = 0.0  # service / software / banks: no inventory is a valid 0, not N/A
    ccc = (0 if np.isnan(dso) else dso) + (0 if np.isnan(dio) else dio) - (0 if np.isnan(dpo) else dpo)
    if np.isnan(dso) and np.isnan(dpo):
        ccc = np.nan

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("DSO (days)", fmt_num(dso, 1) if not np.isnan(dso) else "N/A")
    w2.metric("DIO (days)", fmt_num(dio, 1) if not np.isnan(dio) else "N/A")
    w3.metric("DPO (days)", fmt_num(dpo, 1) if not np.isnan(dpo) else "N/A")
    w4.metric("Cash conversion cycle", fmt_num(ccc, 1) if not np.isnan(ccc) else "N/A", "days")

    if fund.ar == 0 or fund.revenue == 0:
        st.info("Receivables and/or revenue missing — DSO may be incomplete (common for some financials).")
    if fund.cogs == 0:
        st.info("COGS / cost of revenue missing — DIO and DPO defaulted using available proxies or 0.")

    wc = fund.ca - fund.cl
    l1, l2, l3 = st.columns(3)
    l1.metric("Current assets", fmt_money(fund.ca))
    l2.metric("Current liabilities", fmt_money(fund.cl))
    cr = ratio(fund.ca, fund.cl)
    l3.metric("Current ratio", fmt_num(cr, 2) if not np.isnan(cr) else "N/A")

    fig_ccc = go.Figure()
    fig_ccc.add_trace(
        go.Bar(
            x=["DSO", "DIO", "DPO", "CCC"],
            y=[dso if not np.isnan(dso) else 0, dio if not np.isnan(dio) else 0, dpo if not np.isnan(dpo) else 0, ccc if not np.isnan(ccc) else 0],
            marker_color=["#2E5CB8", "#C49A52", "#6B7C99", "#3D9B7A"],
            text=[fmt_num(x, 1) if x is not None and not np.isnan(x) else "N/A" for x in [dso, dio, dpo, ccc]],
            textposition="outside",
        )
    )
    fig_ccc.update_layout(**PLOTLY_LAYOUT, title="Liquidity days", height=380, yaxis_title="Days")
    st.plotly_chart(fig_ccc, use_container_width=True)

# ---- Tab 3 ----
with t3:
    st.subheader("Street target distribution")
    mean_t = safe_float(info.get("targetMeanPrice"), np.nan)
    high_t = safe_float(info.get("targetHighPrice"), np.nan)
    low_t = safe_float(info.get("targetLowPrice"), np.nan)
    n_analysts = info.get("numberOfAnalystOpinions")
    rec = info.get("recommendationKey") or info.get("recommendationMean")

    dispersion = np.nan
    if not (np.isnan(mean_t) or np.isnan(high_t) or np.isnan(low_t) or mean_t == 0):
        dispersion = (high_t - low_t) / mean_t * 100.0

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Mean target", fmt_money(mean_t) if not np.isnan(mean_t) else "N/A")
    s2.metric("High target", fmt_money(high_t) if not np.isnan(high_t) else "N/A")
    s3.metric("Low target", fmt_money(low_t) if not np.isnan(low_t) else "N/A")
    s4.metric("Target dispersion", fmt_pct(dispersion) if not np.isnan(dispersion) else "N/A")
    s5.metric("Analysts / rating", f"{n_analysts or 'N/A'} · {rec if rec is not None else 'N/A'}")

    if np.isnan(mean_t):
        st.info("Sell-side targets were not published for this ticker on Yahoo Finance.")
    else:
        fig_tgt = go.Figure()
        fig_tgt.add_trace(
            go.Bar(
                y=[ticker],
                x=[high_t - low_t] if not (np.isnan(high_t) or np.isnan(low_t)) else [0],
                base=[low_t] if not np.isnan(low_t) else [0],
                orientation="h",
                marker=dict(color="rgba(46, 92, 184, 0.55)", line=dict(color="#8EB4FF", width=1)),
                name="Target band (low → high)",
                hovertemplate="Low: %{base:.2f}<br>Width: %{x:.2f}<extra></extra>",
            )
        )
        if not np.isnan(mean_t):
            fig_tgt.add_trace(
                go.Scatter(
                    y=[ticker],
                    x=[mean_t],
                    mode="markers",
                    marker=dict(size=14, color="#C49A52", symbol="diamond"),
                    name="Mean target",
                )
            )
        if not np.isnan(spot):
            fig_tgt.add_vline(
                x=spot,
                line_dash="dash",
                line_color="#F4F7FB",
                annotation_text=f"Spot {fmt_money(spot)}",
                annotation_position="top",
            )
        fig_tgt.update_layout(
            **PLOTLY_LAYOUT,
            title="Analyst target band vs spot",
            height=280,
            xaxis_title=f"Price ({currency})",
            yaxis_title="",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_tgt, use_container_width=True)

# ---- Tab 4 ----
with t4:
    st.subheader("Implied equity value per share — WACC × g")
    st.caption("Grid is centered on sidebar WACC and g. Cells with WACC ≤ g are blank (Gordon growth undefined).")

    if fund.fcf == 0 or fund.shares <= 0:
        st.info("Heatmap requires last-reported FCF and shares outstanding.")
    else:
        waccs, gs, grid = dcf_grid(fund.fcf, fund.debt, fund.cash, fund.shares, wacc, g)
        x_labels = [f"{x * 100:.1f}%" for x in gs]
        y_labels = [f"{y * 100:.1f}%" for y in waccs]
        fig_hm = px.imshow(
            grid,
            x=x_labels,
            y=y_labels,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            origin="lower",
            labels=dict(color="Value / share"),
        )
        fig_hm.update_traces(
            text=np.vectorize(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}")(grid),
            texttemplate="%{text}",
            hovertemplate="g=%{x}<br>WACC=%{y}<br>Value/share=%{z:,.2f}<extra></extra>",
        )
        fig_hm.update_layout(
            **PLOTLY_LAYOUT,
            title="DCF sensitivity heatmap",
            xaxis_title="Terminal growth (g)",
            yaxis_title="WACC",
            height=520,
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        if not np.isnan(iv):
            st.caption(f"Base case (sidebar): WACC {wacc_pct:.1f}% / g {g_pct:.1f}% → {fmt_money(iv)} per share.")

# ---- Tab 5 ----
with t5:
    st.subheader("Relative valuation vs selected peers")
    names = [ticker] + peers
    rows = []
    failed = []
    with st.spinner("Fetching peer multiples…"):
        for tkr in names:
            try:
                b = fetch_ticker_bundle(tkr) if tkr != ticker else bundle
                inf = b.get("info") or {}
                pe = safe_float(inf.get("forwardPE"), np.nan)
                evebitda = safe_float(inf.get("enterpriseToEbitda"), np.nan)
                if np.isnan(pe) and np.isnan(evebitda):
                    failed.append(tkr)
                rows.append(
                    {
                        "Ticker": tkr,
                        "Name": inf.get("shortName") or tkr,
                        "Forward P/E": pe,
                        "EV/EBITDA": evebitda,
                        "Role": "Primary" if tkr == ticker else "Peer",
                    }
                )
            except Exception:
                failed.append(tkr)
                rows.append(
                    {
                        "Ticker": tkr,
                        "Name": tkr,
                        "Forward P/E": np.nan,
                        "EV/EBITDA": np.nan,
                        "Role": "Primary" if tkr == ticker else "Peer",
                    }
                )

    peers_df = pd.DataFrame(rows)
    st.dataframe(
        peers_df.style.format({"Forward P/E": "{:.1f}", "EV/EBITDA": "{:.1f}"}, na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )
    if failed:
        st.info(f"Incomplete multiples (N/A) for: {', '.join(failed)}. Banks and unprofitable names often lack EV/EBITDA.")

    plot_df = peers_df.dropna(subset=["Forward P/E", "EV/EBITDA"], how="any")
    if plot_df.empty:
        st.info("Not enough Forward P/E and EV/EBITDA pairs to draw a scatter.")
    else:
        fig_sc = px.scatter(
            plot_df,
            x="Forward P/E",
            y="EV/EBITDA",
            text="Ticker",
            color="Role",
            color_discrete_map={"Primary": "#C49A52", "Peer": "#8EB4FF"},
            hover_name="Name",
            size_max=18,
        )
        fig_sc.update_traces(textposition="top center", marker=dict(size=14))
        fig_sc.update_layout(
            **PLOTLY_LAYOUT,
            title="Forward P/E vs EV/EBITDA",
            height=460,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ---- Tab 6 ----
with t6:
    st.subheader("Five-step DuPont decomposition of ROE")
    st.caption("ROE = Tax Burden × Interest Burden × Operating Margin × Asset Turnover × Leverage")

    tax_b = ratio(fund.ni, fund.ebt)
    int_b = ratio(fund.ebt, fund.ebit)
    op_m = ratio(fund.ebit, fund.revenue)
    at = ratio(fund.revenue, fund.assets)
    lev = ratio(fund.assets, fund.equity)
    components = [tax_b, int_b, op_m, at, lev]
    roe = float(np.prod([c for c in components if not np.isnan(c)])) if not any(np.isnan(c) for c in components) else np.nan

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Tax burden (NI / EBT)", fmt_num(tax_b, 3) if not np.isnan(tax_b) else "N/A")
    d2.metric("Interest burden (EBT / EBIT)", fmt_num(int_b, 3) if not np.isnan(int_b) else "N/A")
    d3.metric("Operating margin (EBIT / Sales)", fmt_pct(op_m * 100, 1) if not np.isnan(op_m) else "N/A")
    d4.metric("Asset turnover", fmt_num(at, 2) + "x" if not np.isnan(at) else "N/A")
    d5.metric("Leverage (A / E)", fmt_num(lev, 2) + "x" if not np.isnan(lev) else "N/A")
    d6.metric("ROE", fmt_pct(roe * 100, 1) if not np.isnan(roe) else "N/A")

    if any(np.isnan(c) for c in components):
        st.info("One or more DuPont inputs were missing (EBT/EBIT/equity). ROE shown only when the identity is complete.")

    labels = ["Tax burden", "Interest burden", "Op. margin", "Asset turnover", "Leverage"]
    fig_dup = go.Figure(
        go.Bar(
            x=labels,
            y=[c if not np.isnan(c) else 0 for c in components],
            marker_color="#2E5CB8",
            text=[fmt_num(c, 3) if not np.isnan(c) else "N/A" for c in components],
            textposition="outside",
        )
    )
    fig_dup.update_layout(**PLOTLY_LAYOUT, title="DuPont factors", height=380, yaxis_title="Factor")
    st.plotly_chart(fig_dup, use_container_width=True)

# ---- Tab 7 ----
with t7:
    st.subheader("Altman Z-Score (manufacturing specification)")
    st.caption(
        "Z = 1.2·(WC/A) + 1.4·(RE/A) + 3.3·(EBIT/A) + 0.6·(Mkt Cap / Total Liabilities) + 1.0·(Revenue/A). "
        "Interpretation bands are the classic 1968 cut-offs and are less reliable for banks and growth software."
    )
    wc = fund.ca - fund.cl
    a = fund.assets
    z1 = ratio(wc, a)
    z2 = ratio(fund.re, a)
    z3 = ratio(fund.ebit, a)
    z4 = ratio(mkt_cap if not np.isnan(mkt_cap) else 0.0, fund.liab if fund.liab else np.nan)
    z5 = ratio(fund.revenue, a)
    z = np.nan
    if a > 0 and fund.liab > 0:
        z = 1.2 * (z1 if not np.isnan(z1) else 0) + 1.4 * (z2 if not np.isnan(z2) else 0) + 3.3 * (
            z3 if not np.isnan(z3) else 0
        ) + 0.6 * (z4 if not np.isnan(z4) else 0) + 1.0 * (z5 if not np.isnan(z5) else 0)

    zone, note = z_interpretation(z)
    color = {"Safe Zone": "#3D9B7A", "Grey Zone": "#C49A52", "Distress Zone": "#C45C5C"}.get(zone, "#A9B6CC")

    zc1, zc2 = st.columns([1, 2])
    with zc1:
        st.markdown(
            f"""
            <div class="panel">
              <div class="hero-kicker">Altman Z</div>
              <div class="hero-title" style="font-size:2.6rem;color:{color}">{fmt_num(z, 2) if not np.isnan(z) else "N/A"}</div>
              <p style="color:{color};font-weight:600;margin:0.2rem 0 0.6rem 0;">{zone}</p>
              <p style="color:#A9B6CC;font-size:0.9rem;">{note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with zc2:
        k1, k2, k3 = st.columns(3)
        k1.metric("WC / Assets", fmt_num(z1, 3) if not np.isnan(z1) else "N/A")
        k2.metric("RE / Assets", fmt_num(z2, 3) if not np.isnan(z2) else "N/A")
        k3.metric("EBIT / Assets", fmt_num(z3, 3) if not np.isnan(z3) else "N/A")
        k4, k5, k6 = st.columns(3)
        k4.metric("Mkt Cap / Liabilities", fmt_num(z4, 2) if not np.isnan(z4) else "N/A")
        k5.metric("Revenue / Assets", fmt_num(z5, 2) if not np.isnan(z5) else "N/A")
        k6.metric("Working capital", fmt_money(wc))

    if fund.liab == 0 or fund.assets == 0:
        st.info("Total liabilities or assets missing — Z-Score cannot be computed reliably for this issuer.")

# ---- Tab 8 ----
with t8:
    st.subheader("Piotroski F-Score (fundamental momentum)")
    st.caption("Nine binary tests versus the prior annual period. Score range 0–9. High (≥7) historically associated with stronger subsequent quality.")

    roa = ratio(fund.ni, fund.assets)
    roa_py = ratio(fund.ni_py, fund.assets_py)
    cr = ratio(fund.ca, fund.cl)
    cr_py = ratio(fund.ca_py, fund.cl_py)
    gm = ratio(fund.gp, fund.revenue)
    gm_py = ratio(fund.gp_py, fund.revenue_py)
    at_now = ratio(fund.revenue, fund.assets)
    at_py = ratio(fund.revenue_py, fund.assets_py)
    ltd_ratio = ratio(fund.ltd, fund.assets)
    ltd_ratio_py = ratio(fund.ltd_py, fund.assets_py)

    shares_now = fund.shares_bs if fund.shares_bs else fund.shares
    shares_py = fund.shares_bs_py
    no_new_shares = True
    if shares_py and shares_now:
        no_new_shares = shares_now <= shares_py * 1.005  # 0.5% tolerance for rounding / scrip
    elif fund.issuance > 0:
        no_new_shares = False

    tests = [
        ("Positive net income", fund.ni > 0, f"NI = {fmt_money(fund.ni)}"),
        ("Positive operating cash flow", fund.ocf > 0, f"OCF = {fmt_money(fund.ocf)}"),
        ("ROA higher than prior year", (not np.isnan(roa) and not np.isnan(roa_py) and roa > roa_py), f"{fmt_pct(roa * 100, 2)} vs {fmt_pct(roa_py * 100, 2)}"),
        ("CFO > net income (accrual quality)", fund.ocf > fund.ni, f"OCF {fmt_money(fund.ocf)} vs NI {fmt_money(fund.ni)}"),
        ("Lower long-term debt / assets", (not np.isnan(ltd_ratio) and not np.isnan(ltd_ratio_py) and ltd_ratio < ltd_ratio_py), f"{fmt_num(ltd_ratio, 3)} vs {fmt_num(ltd_ratio_py, 3)}"),
        ("Higher current ratio", (not np.isnan(cr) and not np.isnan(cr_py) and cr > cr_py), f"{fmt_num(cr, 2)} vs {fmt_num(cr_py, 2)}"),
        ("No new shares issued", no_new_shares, f"Shares {fmt_num(shares_now / 1e6, 2, 'M') if shares_now else 'N/A'} vs {fmt_num(shares_py / 1e6, 2, 'M') if shares_py else 'N/A'}"),
        ("Higher gross margin", (not np.isnan(gm) and not np.isnan(gm_py) and gm > gm_py), f"{fmt_pct(gm * 100, 1)} vs {fmt_pct(gm_py * 100, 1)}"),
        ("Higher asset turnover", (not np.isnan(at_now) and not np.isnan(at_py) and at_now > at_py), f"{fmt_num(at_now, 2)}x vs {fmt_num(at_py, 2)}x"),
    ]

    f_score = int(sum(1 for _, passed, _ in tests if passed))
    f_color = "#3D9B7A" if f_score >= 7 else ("#C49A52" if f_score >= 4 else "#C45C5C")
    f_label = "Strong" if f_score >= 7 else ("Mixed" if f_score >= 4 else "Weak")

    fc1, fc2 = st.columns([1, 2.4])
    with fc1:
        st.markdown(
            f"""
            <div class="panel">
              <div class="hero-kicker">Piotroski F</div>
              <div class="hero-title" style="font-size:2.6rem;color:{f_color}">{f_score} / 9</div>
              <p style="color:{f_color};font-weight:600;">{f_label} fundamental momentum</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc2:
        check_rows = []
        for i, (label, passed, detail) in enumerate(tests, start=1):
            check_rows.append(
                {
                    "#": i,
                    "Criterion": label,
                    "Result": "Pass" if passed else "Fail",
                    "Detail": detail,
                    "Pts": 1 if passed else 0,
                }
            )
        cdf = pd.DataFrame(check_rows)
        st.dataframe(cdf, use_container_width=True, hide_index=True)

    if n_cols(bundle.get("income")) < 2:
        st.info("Prior-year statements were incomplete. Year-over-year Piotroski tests may be overly conservative (failed).")

st.markdown(
    '<p class="footnote">For research education only. Yahoo Finance line items are not standardized across banks, insurers, REITs and corporates. '
    "This DCF uses a single growth rate for the explicit period and terminal value and does not model fade, ROIC, or capital structure dynamically. "
    "Not a recommendation to buy or sell any security.</p>",
    unsafe_allow_html=True,
)
