from __future__ import annotations

# ======================================================================
# SECTION: ratios
# ======================================================================
"""Financial ratio engine.

Pure Python / Pandas. No API calls and no LLM: the LLM is never the source of
truth for any number. Every function returns None when inputs are missing or a
division by zero would occur (never a guessed value).
Ratios use year-end balances (not averages) - stated so the user is not misled.
"""

import math

import pandas as pd


def _ok(x) -> bool:
    if x is None:
        return False
    try:
        return not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def safe_div(a, b):
    """a / b, or None if either is missing or b == 0."""
    if not _ok(a) or not _ok(b) or float(b) == 0:
        return None
    return float(a) / float(b)


def pct(x):
    return None if x is None else x * 100.0


def growth(prev, cur):
    """Period growth as a fraction; uses abs(prev) so sign changes behave sanely."""
    if not _ok(prev) or not _ok(cur) or float(prev) == 0:
        return None
    return (float(cur) - float(prev)) / abs(float(prev))


def cagr(first, last, years):
    """Compound annual growth rate as a fraction. Needs positive endpoints."""
    if not (_ok(first) and _ok(last)) or years <= 0:
        return None
    if float(first) <= 0 or float(last) <= 0:
        return None
    return (float(last) / float(first)) ** (1.0 / years) - 1.0


# field -> (statement, possible row labels in yfinance statements)
FIELDS = {
    "revenue": ("income", ["Total Revenue", "Operating Revenue", "Revenue"]),
    "cogs": ("income", ["Cost Of Revenue", "Cost of Revenue"]),
    "gross_profit": ("income", ["Gross Profit"]),
    "operating_income": ("income", ["Operating Income", "Total Operating Income As Reported"]),
    "ebit": ("income", ["EBIT"]),
    "ebitda": ("income", ["EBITDA", "Normalized EBITDA"]),
    "net_income": ("income", ["Net Income", "Net Income Common Stockholders"]),
    "eps": ("income", ["Diluted EPS", "Basic EPS"]),
    "interest_expense": ("income", ["Interest Expense", "Interest Expense Non Operating"]),
    "interest_income": ("income", ["Interest Income", "Interest Income Non Operating"]),
    "total_assets": ("balance", ["Total Assets"]),
    "equity": ("balance", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    "current_assets": ("balance", ["Current Assets"]),
    "current_liabilities": ("balance", ["Current Liabilities"]),
    "inventory": ("balance", ["Inventory"]),
    "receivables": ("balance", ["Accounts Receivable", "Receivables"]),
    "cash": ("balance", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    "total_debt": ("balance", ["Total Debt"]),
    "operating_cf": ("cashflow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    "capex": ("cashflow", ["Capital Expenditure"]),
}


def _frame_by_year(df):
    m = {}
    if df is None or getattr(df, "empty", True):
        return m
    for c in df.columns:
        try:
            m[pd.Timestamp(c).year] = c
        except Exception:
            continue
    return m


def _lookup(df, names, col):
    for n in names:
        if n in df.index:
            v = df.loc[n, col]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if _ok(v):
                return float(v)
    return None


def build_periods(income, balance, cashflow) -> pd.DataFrame:
    """Merge the three statements into one table: rows = fiscal year, cols = fields."""
    frames = {"income": income, "balance": balance, "cashflow": cashflow}
    ymaps = {k: _frame_by_year(v) for k, v in frames.items()}
    years = sorted(set().union(*[set(m) for m in ymaps.values()]))
    rows = {}
    for y in years:
        row = {}
        for field, (fr, names) in FIELDS.items():
            col = ymaps[fr].get(y)
            row[field] = _lookup(frames[fr], names, col) if col is not None else None
        rows[y] = row
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def compute_ratios(p: pd.DataFrame) -> pd.DataFrame:
    """Ratios per fiscal year (percent values are in %, others are plain multiples)."""
    if p is None or p.empty:
        return pd.DataFrame()
    out = {}
    prev = None
    for y, r in p.iterrows():
        def g(k):
            return float(r[k]) if k in r.index and _ok(r[k]) else None

        rev, ni, ta, eq = g("revenue"), g("net_income"), g("total_assets"), g("equity")
        gp, cogs, ca, cl = g("gross_profit"), g("cogs"), g("current_assets"), g("current_liabilities")
        inv, rec, cash, debt = g("inventory"), g("receivables"), g("cash"), g("total_debt")
        ebit, ie, eps = g("ebit"), g("interest_expense"), g("eps")
        opi = g("operating_income") if g("operating_income") is not None else ebit
        if gp is None and rev is not None and cogs is not None:
            gp = rev - cogs
        if cogs is None and rev is not None and gp is not None:
            cogs = rev - gp
        quick = safe_div(ca - inv, cl) if (ca is not None and inv is not None) else None
        row = {
            "revenue": rev, "net_income": ni, "eps": eps,
            "gross_margin_pct": pct(safe_div(gp, rev)),
            "operating_margin_pct": pct(safe_div(opi, rev)),
            "net_margin_pct": pct(safe_div(ni, rev)),
            "roe_pct": pct(safe_div(ni, eq)),
            "roa_pct": pct(safe_div(ni, ta)),
            "current_ratio": safe_div(ca, cl),
            "quick_ratio": quick,
            "cash_ratio": safe_div(cash, cl),
            "debt_to_equity": safe_div(debt, eq),
            "debt_to_assets": safe_div(debt, ta),
            "interest_coverage": safe_div(ebit, abs(ie)) if ie is not None else None,
            "asset_turnover": safe_div(rev, ta),
            "inventory_turnover": safe_div(cogs, inv),
            "receivables_turnover": safe_div(rev, rec),
        }
        if prev is not None:
            row["revenue_growth_pct"] = pct(growth(prev.get("revenue"), rev))
            row["net_income_growth_pct"] = pct(growth(prev.get("net_income"), ni))
            row["eps_growth_pct"] = pct(growth(prev.get("eps"), eps))
            row["operating_profit_growth_pct"] = pct(growth(prev.get("opi"), opi))
        out[y] = row
        prev = {"revenue": rev, "net_income": ni, "eps": eps, "opi": opi}
    df = pd.DataFrame.from_dict(out, orient="index").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def cagr_summary(p: pd.DataFrame) -> dict:
    res = {}
    if p is None or p.empty:
        return res
    for k in ("revenue", "net_income", "eps"):
        if k not in p:
            continue
        s = p[k].dropna()
        if len(s) >= 2:
            yrs = int(s.index[-1] - s.index[0])
            res[f"{k}_cagr_pct"] = pct(cagr(s.iloc[0], s.iloc[-1], yrs))
            res[f"{k}_cagr_period"] = f"{s.index[0]}-{s.index[-1]}"
    return res


def valuation(price, market_cap, latest: dict, div_ttm=None) -> dict:
    """Valuation multiples from a price and the latest annual statement row."""
    g = lambda k: latest.get(k) if latest else None
    debt, cash = g("total_debt"), g("cash")
    ev = None
    if _ok(market_cap) and _ok(debt) and _ok(cash):
        ev = float(market_cap) + float(debt) - float(cash)
    fcf = None
    if _ok(g("operating_cf")) and _ok(g("capex")):
        fcf = float(g("operating_cf")) - abs(float(g("capex")))
    return {
        "price": price,
        "market_cap": market_cap,
        "pe": safe_div(price, g("eps")),
        "pb": safe_div(market_cap, g("equity")),
        "ps": safe_div(market_cap, g("revenue")),
        "ev_ebitda": safe_div(ev, g("ebitda")),
        "dividend_yield_pct": pct(safe_div(div_ttm, price)),
        "fcf_yield_pct": pct(safe_div(fcf, market_cap)),
    }


def red_flags(p: pd.DataFrame, r: pd.DataFrame) -> list[str]:
    """Rule-based warning signs computed in Python (not by the LLM)."""
    flags = []
    if p is None or p.empty or r is None or r.empty:
        return ["Insufficient statement data to run red-flag checks."]
    last, y = p.iloc[-1], p.index[-1]
    lr = r.iloc[-1]
    if _ok(last.get("net_income")) and _ok(last.get("operating_cf")) and last["net_income"] > 0 > last["operating_cf"]:
        flags.append(f"{y}: positive net income but negative operating cash flow (earnings quality check).")
    if _ok(lr.get("current_ratio")) and lr["current_ratio"] < 1:
        flags.append(f"{y}: current ratio below 1 ({lr['current_ratio']:.2f}) - liquidity pressure.")
    if _ok(lr.get("debt_to_equity")) and lr["debt_to_equity"] > 2:
        flags.append(f"{y}: debt-to-equity above 2 ({lr['debt_to_equity']:.2f}).")
    if _ok(lr.get("interest_coverage")) and lr["interest_coverage"] < 1.5:
        flags.append(f"{y}: interest coverage below 1.5x ({lr['interest_coverage']:.2f}).")
    if _ok(last.get("equity")) and last["equity"] < 0:
        flags.append(f"{y}: negative shareholders' equity.")
    if "revenue_growth_pct" in r and _ok(lr.get("revenue_growth_pct")) and lr["revenue_growth_pct"] < 0:
        flags.append(f"{y}: revenue declined {lr['revenue_growth_pct']:.1f}% year on year.")
    if len(p) >= 2:
        prv = p.iloc[-2]
        rg = growth(prv.get("revenue"), last.get("revenue"))
        cg = growth(prv.get("receivables"), last.get("receivables"))
        if rg is not None and cg is not None and cg - rg > 0.20:
            flags.append(f"{y}: receivables grew much faster than revenue (possible collection/recognition issue).")
    return flags or ["No rule-based red flags triggered on the available data."]

# ======================================================================
# SECTION: technical
# ======================================================================
"""Technical indicators (Pandas/NumPy only). Descriptive, never predictive."""

import numpy as np
import pandas as pd


def has_ohlc(df: pd.DataFrame) -> bool:
    """True only if real High/Low exist (not just a copy of Close)."""
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        return False
    h, l = df["High"].dropna(), df["Low"].dropna()
    if h.empty or l.empty:
        return False
    return not bool((df["High"] == df["Close"]).all() and (df["Low"] == df["Close"]).all())


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("Date").reset_index(drop=True)
    c = d["Close"].astype(float)
    for n in (20, 50, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_dn = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    d["RSI14"] = 100 - 100 / (1 + rs)
    d.loc[(avg_dn == 0) & (avg_up > 0), "RSI14"] = 100.0
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_mid"], d["BB_upper"], d["BB_lower"] = mid, mid + 2 * sd, mid - 2 * sd
    if has_ohlc(d):
        pc = c.shift(1)
        tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
        d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    else:
        d["ATR14"] = np.nan
    if "Volume" in d:
        d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    return d


def _f(x, nd=2):
    try:
        return None if pd.isna(x) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def summarize(d: pd.DataFrame) -> dict:
    """Facts (observations) separated from signals (interpretations)."""
    if d is None or d.empty:
        return {}
    last = d.iloc[-1]
    close = float(last["Close"])
    ohlc = has_ohlc(d)
    win = d.tail(120)
    support = float(win["Low"].min()) if ohlc else float(win["Close"].min())
    resist = float(win["High"].max()) if ohlc else float(win["Close"].max())
    obs = {
        "data_start": str(d["Date"].iloc[0].date()),
        "last_date": str(d["Date"].iloc[-1].date()),
        "rows": int(len(d)),
        "last_close": _f(close),
        "sma20": _f(last["SMA20"]), "sma50": _f(last["SMA50"]), "sma200": _f(last["SMA200"]),
        "ema20": _f(last["EMA20"]), "ema50": _f(last["EMA50"]), "ema200": _f(last["EMA200"]),
        "rsi14": _f(last["RSI14"]), "macd": _f(last["MACD"], 3), "macd_signal": _f(last["MACD_signal"], 3),
        "bb_upper": _f(last["BB_upper"]), "bb_lower": _f(last["BB_lower"]), "atr14": _f(last["ATR14"]),
        "support_120d": _f(support), "resistance_120d": _f(resist),
        "ohlc_available": ohlc,
    }
    if "Volume" in d and len(d) >= 60:
        v20, v60 = d["Volume"].tail(20).mean(), d["Volume"].tail(60).mean()
        obs["volume_20d_vs_60d_ratio"] = _f(v20 / v60) if v60 else None
    sig = []
    if not pd.isna(last["SMA50"]):
        sig.append("Price above SMA50 (short/medium trend up)" if close > last["SMA50"] else "Price below SMA50 (short/medium trend down)")
    if not pd.isna(last["SMA200"]):
        sig.append("Price above SMA200 (long-term trend up)" if close > last["SMA200"] else "Price below SMA200 (long-term trend down)")
        if not pd.isna(last["SMA50"]):
            sig.append("SMA50 above SMA200 (golden-cross regime)" if last["SMA50"] > last["SMA200"] else "SMA50 below SMA200 (death-cross regime)")
    if not pd.isna(last["RSI14"]):
        r = last["RSI14"]
        sig.append("RSI > 70: overbought zone" if r > 70 else "RSI < 30: oversold zone" if r < 30 else "RSI neutral (30-70)")
    if not pd.isna(last["MACD_hist"]):
        sig.append("MACD histogram positive (bullish momentum)" if last["MACD_hist"] > 0 else "MACD histogram negative (bearish momentum)")
    if not pd.isna(last["BB_upper"]):
        sig.append("Close above upper Bollinger Band" if close > last["BB_upper"] else "Close below lower Bollinger Band" if close < last["BB_lower"] else "Close inside Bollinger Bands")
    return {"observations": obs, "signals": sig}

# ======================================================================
# SECTION: scenarios
# ======================================================================
"""Simple, transparent scenario table. Assumption-driven - NOT a forecast."""

import pandas as pd



def scenarios(price, eps, base_growth_pct, years, base_pe, growth_spread_pct=5.0, pe_spread_pct=20.0) -> pd.DataFrame:
    """Project EPS for `years` and apply an exit P/E. All inputs are user assumptions."""
    if price is None or eps is None or eps <= 0 or base_pe is None or years <= 0:
        return pd.DataFrame()
    rows = []
    for name, dg, dpe in (("Bearish", -1, -1), ("Base", 0, 0), ("Bullish", 1, 1)):
        g = (base_growth_pct + dg * growth_spread_pct) / 100.0
        pe = base_pe * (1 + dpe * pe_spread_pct / 100.0)
        eps_f = eps * (1 + g) ** years
        implied = eps_f * pe
        up = safe_div(implied - price, price)
        rows.append({
            "Scenario": name, "EPS growth %/yr": round(g * 100, 1), "Exit P/E": round(pe, 1),
            f"EPS in {years}y": round(eps_f, 2), "Implied price": round(implied, 2),
            "Change vs current %": None if up is None else round(up * 100, 1),
        })
    return pd.DataFrame(rows)

# ======================================================================
# SECTION: shariah
# ======================================================================
"""Preliminary Shariah screening support. NOT a fatwa or certification."""


METHODS = {
    "Simplified KMI-style (debt/assets < 37%, interest income/revenue < 5%)": {"debt_assets": 37.0, "interest_rev": 5.0},
    "Stricter (debt/assets < 30%, interest income/revenue < 5%)": {"debt_assets": 30.0, "interest_rev": 5.0},
}


def shariah_screen(method: str, total_debt, total_assets, interest_income, revenue, business_ok: bool) -> dict:
    th = METHODS[method]
    checks = []
    da = pct(safe_div(total_debt, total_assets))
    ir = pct(safe_div(interest_income, revenue))

    def status(val, limit):
        if val is None:
            return "Insufficient data"
        return "Pass" if val < limit else "Fail"

    checks.append({"Check": "Core business activity permissible (user-confirmed)", "Value": "Confirmed" if business_ok else "Not confirmed",
                   "Threshold": "Must be permissible", "Result": "Pass" if business_ok else "Not confirmed"})
    checks.append({"Check": "Total debt / total assets", "Value": None if da is None else round(da, 2),
                   "Threshold": f"< {th['debt_assets']}%", "Result": status(da, th["debt_assets"])})
    checks.append({"Check": "Interest income / revenue", "Value": None if ir is None else round(ir, 2),
                   "Threshold": f"< {th['interest_rev']}%", "Result": status(ir, th["interest_rev"])})
    results = [c["Result"] for c in checks]
    if "Fail" in results:
        outcome = "Preliminary: does NOT pass the selected ratio screen"
    elif "Insufficient data" in results or "Not confirmed" in results:
        outcome = "Inconclusive: missing data or unconfirmed business activity"
    else:
        outcome = "Preliminary: passes the selected simplified ratios (NOT a certification)"
    return {"method": method, "checks": checks, "outcome": outcome,
            "limitations": ["Uses annual statements from a public data feed, not audited screening data.",
                            "Other tests (illiquid assets, non-compliant investments, market-price test) are not covered.",
                            "Verify with a qualified Shariah adviser or a recognised screening provider."]}

# ======================================================================
# SECTION: llm
# ======================================================================
"""Groq client wrapper with retries, rate-limit handling and safe key lookup."""

import os
import re
import time

from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def get_api_key() -> str | None:
    """Env var first (local), then Streamlit secrets (Streamlit Cloud)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def _retry_after(msg: str) -> float | None:
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", msg)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def chat(system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 2000, retries: int = 4) -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it in Streamlit Cloud -> Settings -> Secrets.")
    client = Groq(api_key=key, timeout=90.0, max_retries=0)
    use_reasoning, last = True, None
    for attempt in range(1, retries + 1):
        kwargs = dict(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if use_reasoning:
            kwargs["reasoning_effort"] = "low"  # keeps token use low on the free tier
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            if text and text.strip():
                return text.strip()
            last = RuntimeError("Empty response (reasoning may have used all tokens)")
            max_tokens = int(max_tokens * 1.5)
        except Exception as e:  # noqa: BLE001
            msg, last = str(e), e
            low = msg.lower()
            if use_reasoning and "reasoning" in low:
                use_reasoning = False
                continue
            if "401" in msg or "invalid api key" in low:
                raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY.") from e
            if "429" in msg or "rate limit" in low:
                time.sleep(min(_retry_after(msg) or 15 * attempt, 60) + 1)
                continue
            time.sleep(2 * attempt)
    raise RuntimeError(f"Groq request failed after {retries} attempts: {last}")

# ======================================================================
# SECTION: data
# ======================================================================
"""Data providers (modular - swap these functions to change data source).

Sources, in order: uploaded CSV -> Yahoo Finance (<TICKER>.KA) -> PSX data portal
(dps.psx.com.pk, close/volume only). Nothing is ever fabricated: if a source has
no data the dataset is simply marked unavailable.
"""

import io
import re
from datetime import date

import pandas as pd
import requests
import yfinance as yf

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]", "", t or "").upper()[:12]


def _yf_prices(t: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.Ticker(f"{t}.KA").history(period=period, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    first = df.columns[0]
    df = df.rename(columns={first: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends"] if c in df.columns]
    return df[cols].dropna(subset=["Close"])


def _psx_prices(t: str) -> pd.DataFrame:
    """PSX data-portal EOD feed. Assumed row layout [timestamp, close, volume, open];
    only timestamp/close/volume are used. Treat as a fallback."""
    try:
        r = requests.get(f"https://dps.psx.com.pk/timeseries/eod/{t}", timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        rows = r.json().get("data") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).iloc[:, :3]
        df.columns = ["ts", "Close", "Volume"]
        df["Date"] = pd.to_datetime(df["ts"], unit="s")
        df = df.drop(columns="ts")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        return df.dropna(subset=["Close"]).sort_values("Date")[["Date", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


def load_uploaded_prices(file) -> tuple[pd.DataFrame, str | None]:
    """Validate an uploaded CSV. Returns (df, error)."""
    try:
        raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            return pd.DataFrame(), "File larger than 5 MB."
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), f"Could not read CSV: {e}"
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame(), "CSV needs at least 'Date' and 'Close' columns."
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").tail(20000)
    if len(df) < 30:
        return pd.DataFrame(), "Need at least 30 valid rows."
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep], None


def fetch_prices(t: str, uploaded: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    if uploaded is not None and not uploaded.empty:
        return uploaded, "User-uploaded CSV"
    df = _yf_prices(t)
    if not df.empty:
        return df, f"Yahoo Finance ({t}.KA)"
    df = _psx_prices(t)
    if not df.empty:
        return df, "PSX data portal (close/volume only)"
    return pd.DataFrame(), "unavailable"


def _news(tk) -> list[dict]:
    items = []
    try:
        for n in (tk.news or [])[:10]:
            c = n.get("content", n)
            url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""
            items.append({
                "title": c.get("title", ""),
                "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher", ""),
                "date": str(c.get("pubDate") or c.get("providerPublishTime") or ""),
                "url": url,
            })
    except Exception:
        pass
    return [i for i in items if i["title"]]


def fetch_company(t: str, uploaded: pd.DataFrame | None = None) -> dict:
    """Everything fetched for one ticker, with errors recorded instead of raised."""
    out = {"ticker": t, "retrieved": str(date.today()), "errors": [], "info": {}, "news": []}
    tk = yf.Ticker(f"{t}.KA")
    for key, attr in (("income", "financials"), ("balance", "balance_sheet"),
                      ("cashflow", "cashflow"), ("q_income", "quarterly_financials")):
        try:
            df = getattr(tk, attr)
            out[key] = df if df is not None and not df.empty else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            out[key] = pd.DataFrame()
            out["errors"].append(f"{attr}: {e}")
    try:
        info = tk.info or {}
        out["info"] = {k: info.get(k) for k in ("longName", "sector", "industry", "currency", "financialCurrency",
                                                  "marketCap", "sharesOutstanding", "website")}
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"info: {e}")
    out["news"] = _news(tk)
    out["prices"], out["price_source"] = fetch_prices(t, uploaded)
    return out

# ======================================================================
# SECTION: analysis
# ======================================================================
"""Turns raw fetched data into a validated 'bundle' of calculated results."""

import pandas as pd



def _round(o, nd=2):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, float):
        return None if o != o else round(o, nd)
    return o


def build_bundle(company: dict) -> dict:
    t = company["ticker"]
    periods = build_periods(company.get("income"), company.get("balance"), company.get("cashflow"))
    ratios = compute_ratios(periods)
    prices = company.get("prices", pd.DataFrame())
    tech = add_indicators(prices) if prices is not None and not prices.empty else pd.DataFrame()
    tsum = summarize(tech) if not tech.empty else {}
    info = company.get("info", {})

    price = tsum.get("observations", {}).get("last_close") if tsum else None
    mcap = info.get("marketCap")
    if mcap is None and price and info.get("sharesOutstanding"):
        mcap = price * info["sharesOutstanding"]
    div_ttm = None
    if not prices.empty and "Dividends" in prices:
        cutoff = prices["Date"].max() - pd.Timedelta(days=365)
        div_ttm = float(prices.loc[prices["Date"] > cutoff, "Dividends"].sum())
    latest = periods.iloc[-1].to_dict() if not periods.empty else {}
    val = valuation(price, mcap, latest, div_ttm)
    flags = red_flags(periods, ratios)
    cg = cagr_summary(periods)

    availability = {
        "annual_statements": bool(not periods.empty),
        "fiscal_years": [int(y) for y in periods.index] if not periods.empty else [],
        "prices": bool(not prices.empty),
        "price_source": company.get("price_source"),
        "news_items": len(company.get("news", [])),
    }
    last_year = int(periods.index[-1]) if not periods.empty else None
    latest_ratios = ratios.iloc[-1].to_dict() if not ratios.empty else {}
    ccy = info.get("financialCurrency") or info.get("currency")

    views = {
        "data": _round({"ticker": t, "company": info.get("longName"), "sector": info.get("sector"),
                         "industry": info.get("industry"), "currency": ccy, "retrieved": company["retrieved"],
                         "availability": availability, "fetch_errors": company.get("errors", [])[:5]}),
        "fundamental": _round({"company": info.get("longName"), "currency": ccy, "latest_fiscal_year": last_year,
                                "ratios_by_year": ratios.tail(4).to_dict(orient="index") if not ratios.empty else {},
                                "cagr": cg, "valuation": val}),
        "technical": _round(tsum),
        "market": _round({"company": info.get("longName"), "sector": info.get("sector"),
                           "industry": info.get("industry"), "headlines": company.get("news", [])[:8]}),
        "risk": _round({"latest_ratios": latest_ratios, "red_flags": flags, "valuation": val,
                         "technical_signals": tsum.get("signals") if tsum else None}),
    }
    return {
        "ticker": t, "company": info.get("longName") or t, "info": info, "currency": ccy,
        "retrieved": company["retrieved"], "periods": periods, "ratios": ratios, "valuation": val,
        "tech": tech, "tech_summary": tsum, "flags": flags, "cagr": cg, "news": company.get("news", []),
        "availability": availability, "latest_period": latest, "views": views, "errors": company.get("errors", []),
    }

# ======================================================================
# SECTION: db
# ======================================================================
"""SQLite storage. NOTE: on Streamlit Community Cloud the disk is temporary -
reports/watchlist vanish when the app restarts. Swap this module for a hosted DB
(e.g. Supabase/Postgres) if you need permanent storage."""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "research_app.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY, ticker TEXT, created TEXT, status TEXT, markdown TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS watchlist(ticker TEXT PRIMARY KEY, added TEXT)")
    return c


def save_report(ticker: str, status: str, markdown: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO reports(ticker,created,status,markdown) VALUES(?,?,?,?)",
                  (ticker, datetime.now().strftime("%Y-%m-%d %H:%M"), status, markdown))


def list_reports(limit: int = 30) -> list[tuple]:
    with _conn() as c:
        return c.execute("SELECT id,ticker,created,status FROM reports ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def get_report(rid: int) -> str:
    with _conn() as c:
        row = c.execute("SELECT markdown FROM reports WHERE id=?", (rid,)).fetchone()
    return row[0] if row else ""


def add_watch(ticker: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (ticker, datetime.now().strftime("%Y-%m-%d")))


def remove_watch(ticker: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))


def list_watch() -> list[str]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT ticker FROM watchlist ORDER BY ticker")]

# ======================================================================
# SECTION: agents
# ======================================================================
"""Multi-agent research pipeline (8 specialised agents, sequential, explicit dependencies).

Each agent = a role + instructions + a task, run through Groq. Agents only INTERPRET
numbers that Python already calculated; they never calculate or invent data.
(Written without the CrewAI package on purpose: CrewAI + LiteLLM + chromadb often
fail to install on Streamlit Cloud. The Agent/Task structure below maps 1:1 to
CrewAI if you want to migrate later.)
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime


RULES = (
    "STRICT RULES: Use ONLY the data in the DATA block. Never invent numbers, prices, news, dates or sources. "
    "If a value is null or missing, say 'not available'. Do not give Buy/Sell/Hold orders and never promise returns. "
    "Label clearly: **Facts** (from data), **Interpretation** (your analysis), **Limitations**. "
    "Be concise (max ~350 words), plain English, markdown."
)


@dataclass
class Agent:
    key: str
    role: str
    persona: str
    task: str


AGENTS = [
    Agent("data", "Financial Data Researcher", "Financial data collection specialist.",
          "Summarise what data was collected: company, sector, currency, fiscal years available, price history period and source, "
          "retrieval date, and what is MISSING. Give a data-quality verdict."),
    Agent("fundamental", "Fundamental Analyst", "Senior equity research analyst.",
          "Interpret profitability, liquidity, solvency, efficiency, growth (incl. CAGR) and valuation using the ratios given. "
          "Describe trends across years, strengths and weaknesses. Do not recompute ratios."),
    Agent("technical", "Technical Analyst", "Technical market analysis specialist.",
          "Describe short, medium and long-term trend, momentum, volatility, volume and support/resistance. Separate observations from "
          "signals. State that indicators do not predict future prices."),
    Agent("market", "Market & Industry Researcher", "Financial market intelligence analyst.",
          "You have NO web access. Using only the headlines/notes provided, summarise company/industry developments (with publisher and date). "
          "For Pakistan macro factors (inflation, policy rate, PKR, regulation) do NOT state any figures; list what the investor should verify "
          "and the official sources (SBP, PBS, SECP, PSX), and explain generally how each could affect this sector."),
    Agent("risk", "Risk Analyst", "Senior financial risk analyst.",
          "Assess business, financial, liquidity, debt, earnings-volatility, valuation, currency, regulatory and governance risks that the data supports. "
          "Give bullish / base / bearish narrative scenarios (qualitative), key warning signs, information gaps, and what would invalidate a thesis."),
    Agent("shariah", "Shariah Screening Analyst", "Islamic investment screening researcher.",
          "Explain the preliminary screening result, methodology and thresholds, missing information and limitations. Never declare the company Halal or Haram. "
          "Recommend verification with a qualified Shariah adviser."),
    Agent("writer", "Investment Research Writer", "Senior investment research report writer.",
          "Write the final report in markdown with these sections: 1 Company overview, 2 Executive summary, 3 Financial performance & ratios, "
          "4 Valuation, 5 Technical view, 6 Industry & economic outlook, 7 Risk assessment, 8 Bullish/Base/Bearish scenarios, "
          "9 Shariah screening (only if provided), 10 Investment thesis & catalysts, 11 Red flags, 12 Long-term (3-5 yr) considerations, "
          "13 Data limitations, 14 Final research conclusion (balanced; NO buy/sell instruction). Use only the agent notes in DATA."),
    Agent("qa", "Quality Assurance Analyst", "Financial research verification specialist.",
          "Check the REPORT against the FACTS. List: numbers in the report that do not match the facts, unsupported claims, contradictions between sections, "
          "missing information. Be specific and short. END with exactly one line: STATUS: VERIFIED, STATUS: VERIFIED WITH FLAGS, or STATUS: REJECTED."),
]
BY_KEY = {a.key: a for a in AGENTS}


def _j(obj, limit: int = 6500) -> str:
    s = json.dumps(obj, default=str, separators=(",", ":"), ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "...[truncated]"


def _run(agent: Agent, data, extra: str = "") -> str:
    system = f"You are the {agent.role}: {agent.persona}\n{RULES}"
    user = f"TASK: {agent.task}\n{extra}\nDATA:\n{_j(data)}"
    return chat(system, user, max_tokens=1800 if agent.key != "writer" else 3200)


def key_figures_md(b: dict) -> str:
    """Deterministic (Python-only) key figures block appended to the report."""
    lines = []
    if not b["ratios"].empty:
        y = b["ratios"].index[-1]
        r = b["ratios"].iloc[-1]
        lines.append(f"**Latest fiscal year in data: {y}** (currency: {b['currency'] or 'n/a'})")
        for k, lab in (("revenue_growth_pct", "Revenue growth %"), ("net_margin_pct", "Net margin %"),
                       ("roe_pct", "ROE %"), ("roa_pct", "ROA %"), ("current_ratio", "Current ratio"),
                       ("debt_to_equity", "Debt/Equity"), ("eps", "EPS")):
            v = r.get(k)
            lines.append(f"- {lab}: {'n/a' if v is None or v != v else round(float(v), 2)}")
    v = b["valuation"]
    for k, lab in (("price", "Last price"), ("pe", "P/E"), ("pb", "P/B"), ("ps", "P/S"),
                   ("dividend_yield_pct", "Dividend yield %")):
        x = v.get(k)
        lines.append(f"- {lab}: {'n/a' if x is None else round(float(x), 2)}")
    ts = b["tech_summary"].get("observations", {}) if b["tech_summary"] else {}
    if ts:
        lines.append(f"- Price data: {ts['data_start']} to {ts['last_date']} ({ts['rows']} sessions), source: {b['availability']['price_source']}")
    return "\n".join(lines) or "No calculated figures available."


def run_research(b: dict, opts: dict, on_progress=lambda m: None) -> dict:
    sections, errors = {}, []
    pause = float(opts.get("pause", 6))

    def step(key: str, data, extra: str = "") -> None:
        ag = BY_KEY[key]
        on_progress(f"Running: {ag.role}")
        try:
            sections[key] = _run(ag, data, extra)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ag.role}: {str(e)[:200]}")
            sections[key] = f"_{ag.role} failed: {str(e)[:200]}_"
        time.sleep(pause)

    v = b["views"]
    step("data", v["data"])
    step("fundamental", v["fundamental"])
    step("technical", v["technical"])
    notes = opts.get("notes", "").strip()[:1500]
    step("market", v["market"], f"USER-PROVIDED NOTES (unverified): {notes}" if notes else "")
    step("risk", v["risk"])
    if opts.get("shariah_result"):
        step("shariah", opts["shariah_result"])

    brief = {k: sections[k][:2200] for k in ("data", "fundamental", "technical", "market", "risk", "shariah") if k in sections}
    step("writer", {"company": b["company"], "ticker": b["ticker"], "agent_notes": brief},
         "Use ONLY agent_notes below; keep each section short.")

    report = sections.get("writer", "")
    facts = {"ratios_latest": b["views"]["risk"]["latest_ratios"], "valuation": b["views"]["fundamental"]["valuation"],
             "technical": b["views"]["technical"].get("observations") if b["views"]["technical"] else None,
             "flags": b["flags"], "availability": b["availability"]}
    step("qa", {"FACTS": facts, "REPORT": report[:7000]})

    m = re.search(r"STATUS:\s*(VERIFIED WITH FLAGS|VERIFIED|REJECTED)", sections.get("qa", ""), re.I)
    qa_status = m.group(1).upper() if m else "NOT CONFIRMED"
    status = "INCOMPLETE" if errors else qa_status

    md = (
        f"# Investment Research Report: {b['company']} ({b['ticker']})\n\n"
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} | Data retrieved {b['retrieved']} | Verification status: **{status}***\n\n"
        f"> Educational research only. Not investment advice. No returns are guaranteed. AI text may contain errors; verify all figures against official filings.\n\n"
        f"## Key figures (calculated in Python)\n{key_figures_md(b)}\n\n"
        f"## Red flags (rule-based)\n" + "\n".join(f"- {f}" for f in b["flags"]) + "\n\n"
        f"---\n\n{report}\n\n---\n\n## Quality assurance\n{sections.get('qa', '')}\n"
    )
    if errors:
        md += "\n## Incomplete components\n" + "\n".join(f"- {e}" for e in errors) + "\n"
    return {"sections": sections, "errors": errors, "status": status, "markdown": md}

# ======================================================================
# SECTION: pdf_generator
# ======================================================================
"""Markdown-ish text -> simple professional PDF using fpdf2 (core fonts, latin-1 safe)."""

import re

from fpdf import FPDF

_REPL = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
         "\u2022": "-", "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00a0": " "}


def _clean(s: str) -> str:
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = re.sub(r"[*_`]{1,3}", "", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def markdown_to_pdf(md: str, title: str = "Investment Research Report") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    def line(txt: str, size: int = 10, bold: bool = False, h: float = 5.5):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, _clean(txt), new_x="LMARGIN", new_y="NEXT")

    for raw in md.splitlines():
        t = raw.rstrip()
        if not t.strip() or t.strip() == "---":
            pdf.ln(2)
        elif t.startswith("# "):
            line(t[2:], 16, True, 8)
            pdf.ln(1)
        elif t.startswith("## "):
            pdf.ln(2)
            line(t[3:], 13, True, 7)
        elif t.startswith("### "):
            line(t[4:], 11, True, 6)
        elif t.lstrip().startswith(("- ", "* ")):
            line("  - " + t.lstrip()[2:])
        elif t.startswith("|"):
            if set(t.replace("|", "").strip()) <= set("-: "):
                continue
            line("  ".join(c.strip() for c in t.strip("|").split("|")), 9)
        else:
            line(t.lstrip("> "))
    return bytes(pdf.output())

# ======================================================================
# SECTION: app
# ======================================================================
"""AI Investment Research System - PSX (Streamlit + Groq multi-agent)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="AI Investment Research - PSX", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_company(t: str):
    return fetch_company(t)


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:,.{nd}f}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📈 PSX AI Research")
    ticker = clean_ticker(st.text_input("PSX ticker", value="OGDC", help="e.g. OGDC, LUCK, ENGRO, HBL, MARI"))
    uploaded = st.file_uploader("Optional: price CSV (Date, Close, Volume, ...)", type=["csv"])
    use_shariah = st.checkbox("Include Shariah screening")
    method, biz_ok = None, False
    if use_shariah:
        method = st.selectbox("Methodology", list(METHODS))
        biz_ok = st.checkbox("I confirm the core business is permissible")
    notes = st.text_area("Your own notes (optional, treated as unverified)", height=80)
    pause = st.slider("Pause between agents (sec)", 0, 20, 6, help="Helps stay inside Groq free-tier rate limits.")
    run = st.button("🚀 Start multi-agent research", type="primary", use_container_width=True)
    if not get_api_key():
        st.error("GROQ_API_KEY missing. Add it in Settings -> Secrets.")
    st.caption(f"Model: {MODEL}")

# ------------------------------------------------------------------ run
if run:
    if not ticker:
        st.sidebar.error("Enter a ticker.")
    elif not get_api_key():
        st.sidebar.error("Cannot run agents without GROQ_API_KEY.")
    else:
        up_df = None
        if uploaded is not None:
            up_df, err = load_uploaded_prices(uploaded)
            if err:
                st.sidebar.error(err)
                st.stop()
        with st.spinner("Collecting data..."):
            company = cached_company(ticker)
            if up_df is not None and not up_df.empty:
                company = {**company, "prices": up_df, "price_source": "User-uploaded CSV"}
            bundle = build_bundle(company)
        if not bundle["availability"]["annual_statements"] and not bundle["availability"]["prices"]:
            st.error(f"No data found for '{ticker}'. Check the ticker or upload a price CSV.")
            st.stop()
        sh = None
        if use_shariah:
            p = bundle["latest_period"]
            sh = shariah_screen(method, p.get("total_debt"), p.get("total_assets"),
                                p.get("interest_income"), p.get("revenue"), biz_ok)
        with st.status("Multi-agent research in progress...", expanded=True) as status:
            result = run_research(bundle, {"pause": pause, "notes": notes, "shariah_result": sh},
                                  on_progress=lambda m: status.write(m))
            status.update(label=f"Finished - status: {result['status']}", state="complete")
        st.session_state.update(bundle=bundle, result=result, shariah=sh)
        save_report(ticker, result["status"], result["markdown"])

b, res, sh = st.session_state.get("bundle"), st.session_state.get("result"), st.session_state.get("shariah")

tabs = st.tabs(["Dashboard", "Report", "Fundamentals", "Technical", "Market", "Risk & Scenarios",
                "Shariah", "Compare & Watchlist", "Saved Reports"])

# ------------------------------------------------------------------ dashboard
with tabs[0]:
    st.header("AI Investment Research System")
    st.write("A multi-agent platform that researches PSX-listed companies: data collection, fundamentals, technicals, "
             "market context, risk, optional Shariah screening, report writing and quality assurance. "
             "**All numbers are calculated in Python; the AI only interprets them.**")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recent reports")
        rows = list_reports(5)
        for rid, t, created, s in rows:
            st.write(f"**{t}** - {created} - {s}")
        if not rows:
            st.caption("None yet. Enter a ticker in the sidebar and press Start.")
    with c2:
        st.subheader("Watchlist")
        w = list_watch()
        st.write(", ".join(w) if w else "Empty")
    st.info("Educational research only. Not investment advice. Data from public feeds can be incomplete or delayed.")

if not b:
    for i in range(1, 7):
        with tabs[i]:
            st.info("Run research from the sidebar first.")
else:
    ratios, periods = b["ratios"], b["periods"]
    # -------------------------------------------------------------- report
    with tabs[1]:
        stt = res["status"]
        (st.success if stt.startswith("VERIFIED") else st.warning)(f"Verification status: {stt}")
        st.markdown(res["markdown"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download PDF", markdown_to_pdf(res["markdown"]), f"{b['ticker']}_research.pdf", "application/pdf")
        d2.download_button("⬇️ Download Markdown", res["markdown"], f"{b['ticker']}_research.md")
        with st.expander("Individual agent outputs"):
            for k, v in res["sections"].items():
                st.markdown(f"**{k.upper()}**\n\n{v}")
                st.divider()
    # -------------------------------------------------------------- fundamentals
    with tabs[2]:
        st.subheader(f"{b['company']} - fundamentals ({b['currency'] or 'currency n/a'})")
        st.caption(f"Source: Yahoo Finance annual statements, retrieved {b['retrieved']}. Years available: {b['availability']['fiscal_years']}")
        if ratios.empty:
            st.warning("No annual statements available for this ticker. Ratios were not calculated (nothing is estimated).")
        else:
            disp = ratios.round(2).T
            disp.columns = [str(c) for c in disp.columns]
            st.dataframe(disp, use_container_width=True)
            x = [str(i) for i in ratios.index]
            fig = go.Figure()
            fig.add_bar(x=x, y=ratios["revenue"], name="Revenue")
            fig.add_bar(x=x, y=ratios["net_income"], name="Net income")
            fig.update_layout(barmode="group", title="Revenue vs net income", height=350)
            st.plotly_chart(fig, use_container_width=True)
            fig = go.Figure()
            for c in ("gross_margin_pct", "operating_margin_pct", "net_margin_pct", "roe_pct"):
                fig.add_scatter(x=x, y=ratios[c], name=c.replace("_pct", " %"), mode="lines+markers")
            fig.update_layout(title="Profitability trends", height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(go.Figure(go.Bar(x=x, y=ratios["eps"])).update_layout(title="EPS", height=300), use_container_width=True)
        st.subheader("Valuation")
        st.dataframe(pd.DataFrame([b["valuation"]]).T.rename(columns={0: "value"}), use_container_width=True)
        if b["cagr"]:
            st.write("CAGR:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in b["cagr"].items()})
    # -------------------------------------------------------------- technical
    with tabs[3]:
        d = b["tech"]
        if d.empty:
            st.warning("No price history available. Upload a CSV in the sidebar.")
        else:
            rng = st.radio("Range", ["6M", "1Y", "3Y", "All"], horizontal=True, index=1)
            n = {"6M": 126, "1Y": 252, "3Y": 756, "All": len(d)}[rng]
            v = d.tail(n)
            o = b["tech_summary"]["observations"]
            st.caption(f"Data: {o['data_start']} to {o['last_date']} | last close {o['last_close']} on {o['last_date']} | source: {b['availability']['price_source']}")
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.17, 0.18], vertical_spacing=0.03)
            if has_ohlc(d):
                fig.add_trace(go.Candlestick(x=v["Date"], open=v["Open"], high=v["High"], low=v["Low"], close=v["Close"], name="Price"), 1, 1)
            else:
                fig.add_trace(go.Scatter(x=v["Date"], y=v["Close"], name="Close"), 1, 1)
            for c in ("SMA20", "SMA50", "SMA200", "EMA20", "BB_upper", "BB_lower"):
                fig.add_trace(go.Scatter(x=v["Date"], y=v[c], name=c, line=dict(width=1), visible=True if c in ("SMA50", "SMA200", "BB_upper", "BB_lower") else "legendonly"), 1, 1)
            if "Volume" in v:
                fig.add_trace(go.Bar(x=v["Date"], y=v["Volume"], name="Volume"), 2, 1)
            fig.add_trace(go.Scatter(x=v["Date"], y=v["RSI14"], name="RSI14"), 3, 1)
            fig.add_hline(y=70, line_dash="dot", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", row=3, col=1)
            fig.add_trace(go.Scatter(x=v["Date"], y=v["MACD"], name="MACD"), 4, 1)
            fig.add_trace(go.Scatter(x=v["Date"], y=v["MACD_signal"], name="Signal"), 4, 1)
            fig.add_trace(go.Bar(x=v["Date"], y=v["MACD_hist"], name="Hist"), 4, 1)
            fig.update_layout(height=850, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Observations")
            st.json(o)
            st.subheader("Signals (interpretation, not prediction)")
            for s_ in b["tech_summary"]["signals"]:
                st.write("- " + s_)
            if not o["ohlc_available"]:
                st.caption("This source has close/volume only, so candlesticks and ATR are not shown.")
    # -------------------------------------------------------------- market
    with tabs[4]:
        st.subheader("Latest available company news")
        if not b["news"]:
            st.info("No headlines returned by the data feed. Check PSX announcements, SBP, PBS and SECP directly.")
        for n_ in b["news"]:
            st.markdown(f"- [{n_['title']}]({n_['url']}) - {n_['publisher']} {n_['date'][:10]}")
        st.caption("The app has no live web-search tool. For inflation, policy rate and PKR data use sbp.org.pk, pbs.gov.pk, secp.gov.pk and dps.psx.com.pk.")
        if res:
            st.markdown(res["sections"].get("market", ""))
    # -------------------------------------------------------------- risk
    with tabs[5]:
        st.subheader("Rule-based warning indicators")
        for f in b["flags"]:
            st.write("- " + f)
        st.subheader("Scenario table (your assumptions - not a forecast)")
        eps = b["latest_period"].get("eps") if b["latest_period"] else None
        price = b["valuation"].get("price")
        c1, c2, c3 = st.columns(3)
        g = c1.number_input("Base EPS growth %/yr", value=8.0)
        yrs = c2.slider("Years", 1, 5, 3)
        pe0 = c3.number_input("Base exit P/E", value=float(round(b["valuation"]["pe"], 1)) if b["valuation"].get("pe") else 8.0)
        sc = scenarios(price, eps, g, yrs, pe0)
        if sc.empty:
            st.info("Needs a positive latest EPS and a price. Not available for this company.")
        else:
            st.dataframe(sc, use_container_width=True, hide_index=True)
        if res:
            st.markdown(res["sections"].get("risk", ""))
    # -------------------------------------------------------------- shariah
    with tabs[6]:
        if not sh:
            st.info("Enable 'Include Shariah screening' in the sidebar and run research.")
        else:
            st.write(f"**Methodology:** {sh['method']}")
            st.dataframe(pd.DataFrame(sh["checks"]), hide_index=True, use_container_width=True)
            st.warning(sh["outcome"])
            for l in sh["limitations"]:
                st.write("- " + l)
            st.markdown(res["sections"].get("shariah", ""))

# ------------------------------------------------------------------ compare & watchlist
with tabs[7]:
    st.subheader("Watchlist")
    wc1, wc2 = st.columns(2)
    if wc1.button(f"Add {ticker} to watchlist") and ticker:
        add_watch(ticker)
    rm = wc2.selectbox("Remove", [""] + list_watch())
    if rm and st.button("Remove selected"):
        remove_watch(rm)
        st.rerun()
    st.write("Saved:", ", ".join(list_watch()) or "none")
    st.subheader("Compare 2-5 companies (no AI calls)")
    raw = st.text_input("Tickers (comma separated)", value=", ".join(list_watch()[:5]) or "OGDC, PPL")
    if st.button("Compare"):
        tl = [clean_ticker(x) for x in raw.split(",") if clean_ticker(x)][:5]
        if len(tl) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            rows = []
            for t in tl:
                with st.spinner(f"Loading {t}..."):
                    bb = build_bundle(cached_company(t))
                lr = bb["ratios"].iloc[-1] if not bb["ratios"].empty else {}
                ob = bb["tech_summary"].get("observations", {}) if bb["tech_summary"] else {}
                g_ = lambda k: lr.get(k) if len(lr) else None
                rows.append({"Ticker": t, "FY": int(bb["ratios"].index[-1]) if not bb["ratios"].empty else None,
                             "Rev growth %": g_("revenue_growth_pct"), "EPS growth %": g_("eps_growth_pct"),
                             "Net margin %": g_("net_margin_pct"), "ROE %": g_("roe_pct"), "ROA %": g_("roa_pct"),
                             "D/E": g_("debt_to_equity"), "P/E": bb["valuation"]["pe"], "P/B": bb["valuation"]["pb"],
                             "Div yield %": bb["valuation"]["dividend_yield_pct"],
                             "Price > SMA200": (ob.get("last_close") > ob["sma200"]) if ob.get("sma200") else None,
                             "Red flags": sum(1 for f in bb["flags"] if not f.startswith(("No rule", "Insufficient")))})
            cdf = pd.DataFrame(rows)
            st.dataframe(cdf.round(2), hide_index=True, use_container_width=True)
            if cdf["FY"].nunique() > 1:
                st.warning("Companies report different latest fiscal years - comparison is not like-for-like. No ranking is given.")
            metric = st.selectbox("Chart metric", [c for c in cdf.columns if c not in ("Ticker", "FY", "Price > SMA200")])
            st.plotly_chart(go.Figure(go.Bar(x=cdf["Ticker"], y=cdf[metric])).update_layout(title=metric, height=350), use_container_width=True)

# ------------------------------------------------------------------ saved reports
with tabs[8]:
    st.subheader("Saved reports")
    st.caption("Stored in a temporary SQLite file: on Streamlit Cloud they disappear when the app restarts. Download what you need.")
    rows = list_reports()
    if not rows:
        st.info("No saved reports yet.")
    else:
        pick = st.selectbox("Report", rows, format_func=lambda r: f"#{r[0]} {r[1]} - {r[2]} - {r[3]}")
        md = get_report(pick[0])
        st.download_button("⬇️ Download PDF", markdown_to_pdf(md), f"{pick[1]}_{pick[0]}.pdf", "application/pdf", key="dlsaved")
        st.markdown(md)
