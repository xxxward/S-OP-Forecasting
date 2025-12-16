"""
Unified Sales, Demand & Operations Forecasting Dashboard (Streamlit)

Reads these Google Sheets tabs (exact names, unchanged):
- _NS_Invoices_Data
- Invoice Line Item
- Sales Order Line Item
- _NS_SalesOrders_Data
- _NS_Customer_List
- Raw_Items
- Raw_Vendors
- Average Leadtimes
- Deals
(Optional) Raw_Inventory

Provides 5 tabs:
1) Sales Rep View
2) Operations / Supply Chain
3) Scenario Planning (S&OP) with persistence
4) Purchase Order Forecast + Cashflow timeline
5) Upcoming Deliveries & Tracking (PO API)

Forecasting models (user-selectable):
- Exponential Smoothing (Holt-Winters)
- ARIMA / SARIMA (SARIMAX) with auto AIC search + manual override
- ML (RandomForest / GradientBoosting) with explainability (feature importance)

Notes on robustness:
- Column names vary across ERPs/exports; this app uses a fuzzy column resolver.
- If a critical field cannot be detected, the UI shows a clear error explaining what's missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import altair as alt
import requests

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Unified Forecasting Dashboard", layout="wide")
st.title("Unified Sales, Demand & Operations Forecasting Dashboard")

# Google Sheet tab names (mandatory, do not change)
TABS = {
    "invoices_header": "_NS_Invoices_Data",
    "invoice_line": "Invoice Line Item",
    "so_line": "Sales Order Line Item",
    "so_header": "_NS_SalesOrders_Data",
    "customers": "_NS_Customer_List",
    "items": "Raw_Items",
    "vendors": "Raw_Vendors",
    "avg_leadtimes": "Average Leadtimes",
    "deals": "Deals",
    "inventory": "Raw_Inventory",  # optional
}

DEFAULT_FREQ = "MS"  # Month start
SCENARIO_DIR = Path(".scenario_store")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
SCENARIO_INDEX = SCENARIO_DIR / "index.json"
SCENARIO_APPROVED = SCENARIO_DIR / "approved.json"


# -------------------------
# Small utilities
# -------------------------
def _now() -> datetime:
    return datetime.utcnow()


def _safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        s = re.sub(r"[\$,]", "", s)
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    return int(round(_safe_float(x)))


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _human_money(x: float) -> str:
    return "${:,.0f}".format(float(x))


def _human_qty(x: float) -> str:
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{x:.0f}"


def _winsorize(y: pd.Series, lo_q=0.01, hi_q=0.99) -> pd.Series:
    if y.dropna().empty:
        return y
    lo = y.quantile(lo_q)
    hi = y.quantile(hi_q)
    return y.clip(lower=lo, upper=hi)


def _seasonal_period(freq: str) -> Optional[int]:
    if freq in ("MS", "M"):
        return 12
    if freq.startswith("W"):
        return 52
    return None


def _as_ts(y: pd.Series, freq: str) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    y.index = pd.to_datetime(y.index)
    return y.asfreq(freq).fillna(0.0)


def _date_to_period_start(d: pd.Series, freq: str) -> pd.Series:
    d = pd.to_datetime(d, errors="coerce")
    if freq in ("MS", "M"):
        return d.dt.to_period("M").dt.to_timestamp()
    return d


# -------------------------
# Column resolver
# -------------------------
ALIASES: Dict[str, Tuple[str, ...]] = {
    "date": ("date", "trandate", "transactiondate", "createddate", "orderdate", "invoicedate", "closedate", "shipdate"),
    "sku": ("sku", "item", "itemname", "itemid", "product", "productcode"),
    "customer": ("customer", "customername", "entity", "company", "account", "client"),
    "sales_rep": ("salesrep", "salesperson", "owner", "rep", "accountmanager"),
    "qty": ("quantity", "qty", "quantityordered", "quantityfulfilled", "units"),
    "amount": ("amount", "total", "totalamount", "revenue", "subtotal", "netamount"),
    "status": ("status", "stage", "invoicestatus", "orderstatus"),
    "due_date": ("duedate", "due", "paymentduedate"),
    "paid_date": ("paiddate", "paymentdate", "datepaid"),
    "category": ("productcategory", "category", "calyx", "producttype", "type"),
    "vendor": ("vendor", "supplier"),
    "terms": ("terms", "paymentterms"),
    "purchase_price": ("purchaseprice", "unitcost", "cost", "buyprice"),
    "lead_time_days": ("leadtime", "leadtime_days", "avgleadtime", "leadtimeindays"),
    "probability": ("probability", "dealprobability", "prob"),
}


def resolve_col(df: pd.DataFrame, role: str) -> Optional[str]:
    if df.empty:
        return None
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    for a in ALIASES.get(role, ()):
        if a in norm_map:
            return norm_map[a]
    for c in cols:
        nc = _norm(c)
        for a in ALIASES.get(role, ()):
            if a in nc:
                return c
    if role == "date":
        # last resort: any column containing "date"
        for c in cols:
            if "date" in _norm(c):
                return c
    return None


def require(df: pd.DataFrame, roles: List[str], context: str) -> Optional[Dict[str, str]]:
    out, missing = {}, []
    for r in roles:
        c = resolve_col(df, r)
        if c is None:
            missing.append(r)
        else:
            out[r] = c
    if missing:
        st.error(
            f"Missing required fields for **{context}**: {', '.join(missing)}.\n\n"
            f"Detected columns: {list(df.columns)[:40]}{'...' if len(df.columns)>40 else ''}"
        )
        return None
    return out


# -------------------------
# Google Sheets ingestion
# -------------------------
def _get_spreadsheet_id() -> str:
    # Common patterns
    if "gsheets" in st.secrets and isinstance(st.secrets["gsheets"], dict):
        sid = st.secrets["gsheets"].get("spreadsheet_id") or st.secrets["gsheets"].get("sheet_id")
        if sid:
            return str(sid)
    sid = st.secrets.get("SPREADSHEET_ID") or st.secrets.get("SHEET_ID")
    if not sid:
        raise RuntimeError("Missing Spreadsheet ID in secrets (SPREADSHEET_ID or gsheets.spreadsheet_id).")
    return str(sid)


def _get_service_account_info():
    for k in ("service_account", "gcp_service_account", "google_service_account"):
        if k in st.secrets:
            return dict(st.secrets[k])
    raise RuntimeError(
        f"Missing service account dict in Streamlit secrets. "
        f"Available keys: {list(st.secrets.keys())}"
    )

def _gsheets_client():
    """
    Tries:
    - gspread + google-auth
    - google-api-python-client + google-auth
    """
    sa = _get_service_account_info()
    # gspread first
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(
            sa,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.readonly",
            ],
        )
        return ("gspread", gspread.authorize(creds))
    except Exception:
        pass

    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_service_account_info(sa, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        return ("googleapiclient", service)
    except Exception as e:
        raise RuntimeError(
            "Google Sheets client could not be initialized. "
            "Install `gspread` + `google-auth` or `google-api-python-client` + `google-auth`."
        ) from e


def _dedup_column_names(columns):
    """
    Deduplicate column names by adding .1, .2, etc. to duplicates
    This replaces the internal pandas API that was breaking in pandas 2.0+
    """
    seen = {}
    new_columns = []
    
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}.{seen[col]}")
    
    return new_columns


def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safely coerce dataframe columns without assuming shape."""
    if df is None:
        return pd.DataFrame()

    # If a Series somehow sneaks through, convert to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Ensure columns are strings and unique
    df.columns = _dedup_column_names(
        pd.Series(df.columns)
        .astype(str)
        .str.strip()
        .fillna("unknown")
        .tolist()
    )

    for c in df.columns:
        try:
            col = df[c]

            # If duplicate column -> DataFrame, skip dtype coercion
            if isinstance(col, pd.DataFrame):
                continue

            if col.dtype == object:
                # Try numeric
                coerced = pd.to_numeric(col, errors="ignore")
                df[c] = coerced

        except Exception:
            # Never hard-fail during coercion
            continue

    return df



@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_tab(sheet_id: str, tab: str) -> pd.DataFrame:
    mode, client = _gsheets_client()
    
    # Special handling for Deals tab - headers are in row 2
    header_row_idx = 1 if tab == "Deals" else 0
    data_start_idx = header_row_idx + 1
    
    if mode == "gspread":
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab)
        vals = ws.get_all_values()
        if not vals:
            return pd.DataFrame()
        if len(vals) <= header_row_idx:
            return pd.DataFrame()
        df = pd.DataFrame(vals[data_start_idx:], columns=vals[header_row_idx])
        return _coerce_df(df)
    # googleapiclient
    rng = f"'{tab}'"
    res = client.spreadsheets().values().get(spreadsheetId=sheet_id, range=rng).execute()
    vals = res.get("values", [])
    if not vals:
        return pd.DataFrame()
    if len(vals) <= header_row_idx:
        return pd.DataFrame()
    df = pd.DataFrame(vals[data_start_idx:], columns=vals[header_row_idx])
    return _coerce_df(df)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_all(sheet_id: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for key, tab in TABS.items():
        try:
            out[key] = load_tab(sheet_id, tab)
        except Exception as e:
            if key == "inventory":
                out[key] = pd.DataFrame()
            else:
                raise
    return out


def clear_cache():
    load_tab.clear()
    load_all.clear()


# -------------------------
# Data modeling
# -------------------------
def build_items(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=["sku", "category", "vendor", "purchase_price", "lead_time_days"])
    sku = resolve_col(items, "sku") or items.columns[0]
    cat = resolve_col(items, "category")
    ven = resolve_col(items, "vendor")
    cost = resolve_col(items, "purchase_price")
    lt = resolve_col(items, "lead_time_days")

    out = pd.DataFrame(
        {
            "sku": items[sku].astype(str).str.strip(),
            "category": items[cat].astype(str).str.strip() if cat else "Uncategorized",
            "vendor": items[ven].astype(str).str.strip() if ven else "Unknown Vendor",
            "purchase_price": items[cost].apply(_safe_float) if cost else np.nan,
            "lead_time_days": items[lt].apply(_safe_float) if lt else np.nan,
        }
    )
    out["category"] = out["category"].replace({"": "Uncategorized"}).fillna("Uncategorized")
    out["vendor"] = out["vendor"].replace({"": "Unknown Vendor"}).fillna("Unknown Vendor")
    return out.drop_duplicates(subset=["sku"])


def build_vendors(vendors: pd.DataFrame) -> pd.DataFrame:
    if vendors.empty:
        return pd.DataFrame(columns=["vendor", "terms"])
    ven = resolve_col(vendors, "vendor") or vendors.columns[0]
    terms = resolve_col(vendors, "terms")
    out = pd.DataFrame(
        {
            "vendor": vendors[ven].astype(str).str.strip(),
            "terms": vendors[terms].astype(str).str.strip() if terms else np.nan,
        }
    ).drop_duplicates(subset=["vendor"])
    out["terms"] = out["terms"].replace({"": np.nan})
    return out


def build_avg_lead_map(avg: pd.DataFrame) -> pd.DataFrame:
    if avg.empty:
        return pd.DataFrame(columns=["key", "key_type", "avg_lead_time_days"])
    lt = resolve_col(avg, "lead_time_days")
    ven = resolve_col(avg, "vendor")
    cat = resolve_col(avg, "category")
    sku = resolve_col(avg, "sku")

    if lt is None:
        # try any column containing "lead"
        cands = [c for c in avg.columns if "lead" in _norm(c)]
        lt = cands[0] if cands else None
    if lt is None:
        return pd.DataFrame(columns=["key", "key_type", "avg_lead_time_days"])

    rows = []
    for _, r in avg.iterrows():
        v = _safe_float(r.get(lt))
        if v <= 0:
            continue
        if sku and pd.notna(r.get(sku)):
            rows.append({"key": str(r.get(sku)).strip(), "key_type": "sku", "avg_lead_time_days": v})
        elif ven and pd.notna(r.get(ven)):
            rows.append({"key": str(r.get(ven)).strip(), "key_type": "vendor", "avg_lead_time_days": v})
        elif cat and pd.notna(r.get(cat)):
            rows.append({"key": str(r.get(cat)).strip(), "key_type": "category", "avg_lead_time_days": v})
    return pd.DataFrame(rows).drop_duplicates(subset=["key", "key_type"])


def demand_history(so_line: pd.DataFrame, items: pd.DataFrame, freq: str) -> pd.DataFrame:
    if so_line.empty:
        return pd.DataFrame(columns=["ds", "sku", "customer", "qty", "category", "vendor", "purchase_price", "lead_time_days"])

    cols = require(so_line, ["date", "sku", "qty"], "Sales Order Line Item")
    if cols is None:
        return pd.DataFrame()

    cust = resolve_col(so_line, "customer")
    df = so_line.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"]])

    df["ds"] = _date_to_period_start(df[cols["date"]], freq)
    df["sku"] = df[cols["sku"]].astype(str).str.strip()
    df["qty"] = df[cols["qty"]].apply(_safe_float)
    df["customer"] = df[cust].astype(str).str.strip() if cust else "All Customers"

    out = df.groupby(["ds", "sku", "customer"], as_index=False)["qty"].sum().sort_values(["ds", "sku", "customer"])
    if not items.empty:
        out = out.merge(items, on="sku", how="left")
    return out


def invoice_line_history(inv_line: pd.DataFrame, items: pd.DataFrame, freq: str) -> pd.DataFrame:
    if inv_line.empty:
        return pd.DataFrame(columns=["ds", "sku", "customer", "revenue", "qty", "category"])
    cols = require(inv_line, ["date", "sku", "amount"], "Invoice Line Item")
    if cols is None:
        return pd.DataFrame()

    qty = resolve_col(inv_line, "qty")
    cust = resolve_col(inv_line, "customer")

    df = inv_line.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"]])
    df["ds"] = _date_to_period_start(df[cols["date"]], freq)
    df["sku"] = df[cols["sku"]].astype(str).str.strip()
    df["revenue"] = df[cols["amount"]].apply(_safe_float)
    df["qty"] = df[qty].apply(_safe_float) if qty else np.nan
    df["customer"] = df[cust].astype(str).str.strip() if cust else "All Customers"

    out = df.groupby(["ds", "sku", "customer"], as_index=False)[["revenue", "qty"]].sum().sort_values(["ds", "sku"])
    if not items.empty:
        out = out.merge(items[["sku", "category"]], on="sku", how="left")
    return out


def unit_price_map(inv_hist: pd.DataFrame) -> pd.DataFrame:
    if inv_hist.empty or "qty" not in inv_hist.columns:
        return pd.DataFrame(columns=["sku", "avg_unit_price"])
    df = inv_hist.dropna(subset=["sku"]).copy()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df[df["qty"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["sku", "avg_unit_price"])
    g = df.groupby("sku", as_index=False).agg(revenue=("revenue", "sum"), qty=("qty", "sum"))
    g["avg_unit_price"] = g["revenue"] / g["qty"]
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_unit_price"])
    return g[["sku", "avg_unit_price"]]


def pipeline_history(deals: pd.DataFrame, freq: str) -> pd.DataFrame:
    if deals.empty:
        return pd.DataFrame(columns=["ds", "category", "pipeline_amount"])
    cols = require(deals, ["date", "amount"], "Deals (pipeline)")
    if cols is None:
        return pd.DataFrame()

    cat = resolve_col(deals, "category")
    prob = resolve_col(deals, "probability")
    stage = resolve_col(deals, "status")

    df = deals.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"]])
    df["ds"] = _date_to_period_start(df[cols["date"]], freq)
    df["category"] = df[cat].astype(str).str.strip() if cat else "All Categories"
    df["amount"] = df[cols["amount"]].apply(_safe_float)

    if prob:
        p = df[prob].apply(_safe_float)
        p = np.where(p > 1.0, p / 100.0, p)
        df["prob"] = np.clip(p, 0, 1)
    else:
        # conservative transparent stage->prob mapping (shown in UI)
        s = df[stage].astype(str).str.lower() if stage else ""
        df["prob"] = np.where(s.str.contains("closed won|won", na=False), 1.0,
                              np.where(s.str.contains("contract|negotiat|proposal", na=False), 0.5,
                                       np.where(s.str.contains("qualified|discovery|demo", na=False), 0.25, 0.15)))

    df["pipeline_amount"] = df["amount"] * df["prob"]
    return df.groupby(["ds", "category"], as_index=False)["pipeline_amount"].sum().sort_values(["ds", "category"])


def invoice_payment_metrics(inv_hdr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if inv_hdr.empty:
        return pd.DataFrame(), pd.DataFrame()

    cols = require(inv_hdr, ["date", "amount"], "Invoice Header")
    if cols is None:
        return pd.DataFrame(), pd.DataFrame()

    due = resolve_col(inv_hdr, "due_date")
    paid = resolve_col(inv_hdr, "paid_date")
    status = resolve_col(inv_hdr, "status")
    cust = resolve_col(inv_hdr, "customer")

    df = inv_hdr.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df = df.dropna(subset=[cols["date"]])
    df["invoice_date"] = df[cols["date"]]
    df["amount"] = df[cols["amount"]].apply(_safe_float)
    df["customer"] = df[cust].astype(str).str.strip() if cust else "Unknown Customer"

    df["due_date"] = pd.to_datetime(df[due], errors="coerce") if due else pd.NaT
    df["paid_date"] = pd.to_datetime(df[paid], errors="coerce") if paid else pd.NaT

    is_paid = pd.Series(False, index=df.index)
    if status:
        s = df[status].astype(str).str.lower()
        is_paid = is_paid | s.str.contains("paid|closed|settled", na=False)
    if paid:
        is_paid = is_paid | df["paid_date"].notna()
    df["is_paid"] = is_paid

    today = pd.Timestamp(date.today())
    df["days_to_pay"] = (df["paid_date"] - df["invoice_date"]).dt.days
    df["days_past_due"] = np.where(
        df["is_paid"],
        np.where(df["due_date"].notna(), (df["paid_date"] - df["due_date"]).dt.days, np.nan),
        np.where(df["due_date"].notna(), (today - df["due_date"]).dt.days, np.nan),
    )

    open_inv = df.loc[~df["is_paid"]].copy()
    open_inv["days_past_due"] = pd.to_numeric(open_inv["days_past_due"], errors="coerce").fillna(0).astype(int)
    open_inv = open_inv.sort_values(["days_past_due", "amount"], ascending=[False, False])

    paid_df = df.loc[df["is_paid"]].copy()
    summary = paid_df.groupby("customer", as_index=False).agg(
        invoices_paid=("amount", "count"),
        total_paid=("amount", "sum"),
        avg_days_to_pay=("days_to_pay", "mean"),
        p90_days_to_pay=("days_to_pay", lambda x: float(np.nanpercentile(x.dropna(), 90)) if x.dropna().size else np.nan),
    )
    unpaid = df.loc[~df["is_paid"]].groupby("customer", as_index=False).agg(
        invoices_open=("amount", "count"),
        total_open=("amount", "sum"),
        max_days_past_due=("days_past_due", "max"),
    )
    cust_summary = summary.merge(unpaid, on="customer", how="outer").fillna(0)
    return open_inv, cust_summary


# -------------------------
# Forecasting models
# -------------------------
@dataclass
class ForecastConfig:
    model: str  # "exp" | "arima" | "ml"
    horizon: int
    freq: str = DEFAULT_FREQ

    # volatility
    winsorize: bool = True
    lo_q: float = 0.01
    hi_q: float = 0.99

    # Exp smoothing
    es_trend: str = "add"  # add|mul|none
    es_seasonal: str = "auto"  # auto|add|mul|none
    es_damped: bool = True
    es_alpha: Optional[float] = None
    es_beta: Optional[float] = None
    es_gamma: Optional[float] = None

    # ARIMA/SARIMA
    arima_auto: bool = True
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 12)
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False

    # ML
    ml_model: str = "random_forest"  # random_forest|gradient_boosting
    ml_min_hist: int = 18
    ml_lags: Tuple[int, ...] = (1, 2, 3)
    ml_rolls: Tuple[int, ...] = (3, 6)
    rf_n_estimators: int = 400
    rf_max_depth: Optional[int] = None
    random_state: int = 42


def forecast_exp(y: pd.Series, cfg: ForecastConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    
    # Minimum data check - need at least 2 points for exponential smoothing
    if len(y) < 2:
        # Return naive forecast (repeat last value)
        last_val = y.iloc[-1] if len(y) > 0 else 0.0
        
        # Create proper future date range
        if len(y) > 0:
            last_date = y.index[-1]
            if isinstance(last_date, pd.Period):
                future_dates = pd.period_range(start=last_date + 1, periods=cfg.horizon, freq=cfg.freq)
            else:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=cfg.horizon, freq=cfg.freq)
        else:
            # No data at all, use today as start
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=cfg.horizon, freq=cfg.freq)
        
        fc = pd.Series([last_val] * cfg.horizon, index=future_dates)
        return fc, {"method": "naive", "reason": "insufficient_data"}
    
    if cfg.winsorize:
        y = _winsorize(y, cfg.lo_q, cfg.hi_q)
    y = _as_ts(y, cfg.freq)

    sp = _seasonal_period(cfg.freq)
    allow_seasonal = sp is not None and y.shape[0] >= (sp * 2)

    trend = None if cfg.es_trend == "none" else cfg.es_trend
    seasonal = cfg.es_seasonal
    if seasonal == "auto":
        seasonal = "add" if allow_seasonal else "none"
    seasonal = None if seasonal == "none" else seasonal

    # multiplicative requires positive values
    if trend == "mul" and (y <= 0).any():
        trend = "add"
    if seasonal == "mul" and (y <= 0).any():
        seasonal = "add"

    model = ExponentialSmoothing(
        y,
        trend=trend,
        damped_trend=cfg.es_damped if trend else False,
        seasonal=seasonal,
        seasonal_periods=sp if seasonal else None,
        initialization_method="estimated",
    )

    fit_kwargs = {}
    if cfg.es_alpha is not None:
        fit_kwargs["smoothing_level"] = float(cfg.es_alpha)
    if cfg.es_beta is not None and trend:
        fit_kwargs["smoothing_trend"] = float(cfg.es_beta)
    if cfg.es_gamma is not None and seasonal:
        fit_kwargs["smoothing_seasonal"] = float(cfg.es_gamma)

    res = model.fit(optimized=(len(fit_kwargs) == 0), **fit_kwargs)
    fc = res.forecast(cfg.horizon)
    meta = {"model": "ExponentialSmoothing", "trend": trend, "seasonal": seasonal, "seasonal_periods": sp if seasonal else None, "aic": getattr(res, "aic", None)}
    return fc, meta


def _auto_sarima(y: pd.Series, sp: Optional[int], max_p=2, max_q=2, max_P=1, max_Q=1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float]:
    y = _as_ts(y, DEFAULT_FREQ)
    n = y.shape[0]
    if n < 12:
        return (0, 1, 0), (0, 0, 0, sp or 0), np.inf
    d = 1 if n >= 24 else 0
    D = 1 if (sp and n >= sp * 2) else 0

    best_aic = np.inf
    best_order = (1, d, 1)
    best_season = (0, D, 0, sp or 0)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range((max_P + 1) if sp else 1):
                for Q in range((max_Q + 1) if sp else 1):
                    order = (p, d, q)
                    season = (P, D, Q, sp) if sp else (0, 0, 0, 0)
                    try:
                        res = SARIMAX(y, order=order, seasonal_order=season, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        if np.isfinite(res.aic) and res.aic < best_aic:
                            best_aic = res.aic
                            best_order, best_season = order, season
                    except Exception:
                        continue
    return best_order, best_season, best_aic


def forecast_arima(y: pd.Series, cfg: ForecastConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    if cfg.winsorize:
        y = _winsorize(y, cfg.lo_q, cfg.hi_q)
    y = _as_ts(y, cfg.freq)

    sp = _seasonal_period(cfg.freq)
    if cfg.arima_auto:
        order, season, _ = _auto_sarima(y, sp)
    else:
        order, season = cfg.order, cfg.seasonal_order

    res = SARIMAX(
        y,
        order=order,
        seasonal_order=season,
        enforce_stationarity=cfg.enforce_stationarity,
        enforce_invertibility=cfg.enforce_invertibility,
    ).fit(disp=False)
    fc = res.get_forecast(steps=cfg.horizon).predicted_mean
    meta = {"model": "SARIMAX", "order": order, "seasonal_order": season, "aic": res.aic}
    return fc, meta


def _time_features(ds: pd.Series) -> pd.DataFrame:
    t = pd.to_datetime(ds)
    return pd.DataFrame(
        {
            "year": t.dt.year.astype(int),
            "month": t.dt.month.astype(int),
            "quarter": t.dt.quarter.astype(int),
            "month_sin": np.sin(2 * np.pi * t.dt.month / 12.0),
            "month_cos": np.cos(2 * np.pi * t.dt.month / 12.0),
        }
    )


def _ml_supervised(hist: pd.DataFrame, target: str, lags: Tuple[int, ...], rolls: Tuple[int, ...], cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    df = hist.sort_values("ds").copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0.0)

    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    for w in rolls:
        df[f"roll_mean_{w}"] = df[target].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df[target].shift(1).rolling(w).std()

    df = pd.concat([df.reset_index(drop=True), _time_features(df["ds"]).reset_index(drop=True)], axis=1)
    feature_cols = [c for c in df.columns if c not in {target, "ds"}]
    X = df[feature_cols]
    y = df[target]
    keep = ~X[[f"lag_{lag}" for lag in lags]].isna().any(axis=1)
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


def _fit_tree_model(X: pd.DataFrame, y: pd.Series, cfg: ForecastConfig, cat_cols: List[str]) -> Pipeline:
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
    )
    if cfg.ml_model == "gradient_boosting":
        reg = GradientBoostingRegressor(random_state=cfg.random_state)
    else:
        reg = RandomForestRegressor(
            n_estimators=int(cfg.rf_n_estimators),
            max_depth=cfg.rf_max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    pipe = Pipeline([("prep", pre), ("model", reg)])
    pipe.fit(X, y)
    return pipe


def _feature_importance(pipe: Pipeline, cat_cols: List[str]) -> pd.DataFrame:
    m = pipe.named_steps["model"]
    if not hasattr(m, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    prep = pipe.named_steps["prep"]
    try:
        num_cols = prep.transformers_[0][2]
        cat_cols_ = prep.transformers_[1][2]
        ohe = prep.named_transformers_["cat"].named_steps["ohe"]
        names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols_))
    except Exception:
        names = [f"f{i}" for i in range(len(m.feature_importances_))]
    imp = pd.Series(m.feature_importances_, index=names)

    # aggregate one-hot back to base categorical col
    agg: Dict[str, float] = {}
    for k, v in imp.items():
        base = next((c for c in cat_cols if str(k).startswith(c)), str(k))
        agg[base] = agg.get(base, 0.0) + float(v)
    out = pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def forecast_ml(hist: pd.DataFrame, target: str, cfg: ForecastConfig, cat_cols: List[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    if hist["ds"].nunique() < cfg.ml_min_hist:
        y = hist.set_index("ds")[target]
        fc, meta = forecast_exp(y, cfg)
        meta["note"] = f"ML fallback -> ExponentialSmoothing (history < {cfg.ml_min_hist})"
        return fc, meta

    X, y = _ml_supervised(hist, target, cfg.ml_lags, cfg.ml_rolls, cat_cols)
    pipe = _fit_tree_model(X, y, cfg, cat_cols)
    fi = _feature_importance(pipe, cat_cols)

    last_ds = pd.to_datetime(hist["ds"].max())
    future_ds = pd.date_range(start=last_ds + pd.tseries.frequencies.to_offset(cfg.freq), periods=cfg.horizon, freq=cfg.freq)

    ext = hist[["ds", target] + cat_cols].copy().sort_values("ds")
    preds = []
    for ds in future_ds:
        # append empty row and rebuild features to get the last row
        row = {c: ext[c].iloc[-1] for c in cat_cols}
        row.update({"ds": ds, target: np.nan})
        temp = pd.concat([ext, pd.DataFrame([row])], ignore_index=True)
        X_i, _ = _ml_supervised(temp, target, cfg.ml_lags, cfg.ml_rolls, cat_cols)
        x_last = X_i.iloc[[-1]]
        y_hat = float(pipe.predict(x_last)[0])
        y_hat = max(0.0, y_hat)
        preds.append(y_hat)
        row[target] = y_hat
        ext = pd.concat([ext, pd.DataFrame([row])], ignore_index=True)

    fc = pd.Series(preds, index=future_ds)
    meta = {"model": f"ML ({'GBR' if cfg.ml_model=='gradient_boosting' else 'RF'})", "feature_importance": fi.to_dict(orient="records")}
    return fc, meta


def forecast_univariate(y: pd.Series, cfg: ForecastConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    if cfg.model == "exp":
        return forecast_exp(y, cfg)
    if cfg.model == "arima":
        return forecast_arima(y, cfg)
    # fallback: ml isn't univariate here
    return forecast_exp(y, cfg)


# -------------------------
# Allocation + scenario math
# -------------------------
def sku_mix(demand: pd.DataFrame, category: Optional[str], lookback_m: int) -> pd.DataFrame:
    if demand.empty:
        return pd.DataFrame(columns=["category", "sku", "mix_share"])
    df = demand.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    cutoff = df["ds"].max() - pd.DateOffset(months=lookback_m)
    df = df[df["ds"] >= cutoff]
    if category:
        df = df[df["category"] == category]
    if df.empty:
        return pd.DataFrame(columns=["category", "sku", "mix_share"])
    g = df.groupby(["category", "sku"], as_index=False)["qty"].sum()
    g["mix_share"] = g.groupby("category")["qty"].transform(lambda x: x / x.sum() if x.sum() else 0.0)
    return g[["category", "sku", "mix_share"]]


def allocate_category_to_skus(cat_series: pd.DataFrame, mix_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if cat_series.empty or mix_df.empty:
        return pd.DataFrame(columns=["ds", "category", "sku", value_col])
    df = cat_series.merge(mix_df, on="category", how="left")
    df["mix_share"] = df["mix_share"].fillna(0.0)
    df[value_col] = df[value_col].apply(_safe_float) * df["mix_share"]
    return df[["ds", "category", "sku", value_col]]


def blend_forecasts(demand_fc: pd.DataFrame, sales_fc: pd.DataFrame, w_demand: float, growth: float) -> pd.DataFrame:
    if demand_fc.empty and sales_fc.empty:
        return pd.DataFrame(columns=["ds", "sku", "qty_final"])
    base = demand_fc[["ds", "sku", "qty_forecast"]].copy() if not demand_fc.empty else pd.DataFrame(columns=["ds", "sku", "qty_forecast"])
    out = base.merge(sales_fc[["ds", "sku", "qty_sales"]], on=["ds", "sku"], how="outer") if not sales_fc.empty else base.assign(qty_sales=0.0)
    out["qty_forecast"] = out["qty_forecast"].fillna(0.0)
    out["qty_sales"] = out["qty_sales"].fillna(0.0)
    w_d = float(np.clip(w_demand, 0.0, 1.0))
    out["qty_final"] = (w_d * out["qty_forecast"]) + ((1 - w_d) * out["qty_sales"])
    out["qty_final"] = out["qty_final"] * (1 + float(growth))
    out["qty_final"] = out["qty_final"].clip(lower=0.0)
    return out[["ds", "sku", "qty_final"]].sort_values(["ds", "sku"])


# -------------------------
# Scenario store (local, persistent)
# -------------------------
def _load_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, default=str))


def list_scenarios() -> List[Dict[str, Any]]:
    return _load_json(SCENARIO_INDEX, [])


def save_scenario(name: str, assumptions: Dict[str, Any], forecast_df: pd.DataFrame, cash_df: pd.DataFrame) -> str:
    sid = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "scenario"
    sid = f"{sid}-{int(_now().timestamp())}"
    f_path = SCENARIO_DIR / f"{sid}__forecast.parquet"
    c_path = SCENARIO_DIR / f"{sid}__cash.parquet"
    forecast_df.to_parquet(f_path, index=False)
    cash_df.to_parquet(c_path, index=False)
    idx = list_scenarios()
    idx.append({"scenario_id": sid, "name": name, "created_at_utc": _now().isoformat(), "assumptions": assumptions, "forecast_path": str(f_path), "cash_path": str(c_path)})
    _save_json(SCENARIO_INDEX, idx)
    return sid


def load_scenario(sid: str) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    rec = next((r for r in list_scenarios() if r.get("scenario_id") == sid), None)
    if not rec:
        return None, pd.DataFrame(), pd.DataFrame()
    f = Path(rec["forecast_path"])
    c = Path(rec["cash_path"])
    return rec, (pd.read_parquet(f) if f.exists() else pd.DataFrame()), (pd.read_parquet(c) if c.exists() else pd.DataFrame())


def approve_scenario(sid: str):
    _save_json(SCENARIO_APPROVED, {"scenario_id": sid, "approved_at_utc": _now().isoformat()})


def approved_scenario_id() -> Optional[str]:
    return _load_json(SCENARIO_APPROVED, {}).get("scenario_id")


# -------------------------
# Purchase order forecast + cashflow
# -------------------------
def _parse_terms_days(terms: Any) -> Optional[int]:
    if terms is None or (isinstance(terms, float) and np.isnan(terms)):
        return None
    s = str(terms).strip().lower()
    if not s:
        return None
    m = re.search(r"net\s*(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s*days?\b", s)
    if m:
        return int(m.group(1))
    if "cod" in s or "due on receipt" in s or "cash on delivery" in s or "prepay" in s or "upfront" in s:
        return 0
    return None


def build_po_plan(
    forecast_units: pd.DataFrame,  # ds, sku, qty_forecast
    items: pd.DataFrame,
    vendors: pd.DataFrame,
    avg_leads: pd.DataFrame,
    terms_basis: str = "po_date",  # po_date|arrival_date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if forecast_units.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = forecast_units.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["sku"] = df["sku"].astype(str).str.strip()
    df["qty_forecast"] = df["qty_forecast"].apply(_safe_float)
    df["po_qty"] = df["qty_forecast"].apply(_safe_int)

    if not items.empty:
        df = df.merge(items, on="sku", how="left")
    else:
        df["vendor"] = "Unknown Vendor"
        df["category"] = "Uncategorized"
        df["purchase_price"] = np.nan
        df["lead_time_days"] = np.nan

    # fill lead times from avg map (sku->vendor->category)
    if not avg_leads.empty:
        sku_map = avg_leads[avg_leads["key_type"] == "sku"].set_index("key")["avg_lead_time_days"].to_dict()
        ven_map = avg_leads[avg_leads["key_type"] == "vendor"].set_index("key")["avg_lead_time_days"].to_dict()
        cat_map = avg_leads[avg_leads["key_type"] == "category"].set_index("key")["avg_lead_time_days"].to_dict()
        lt = df["lead_time_days"].apply(_safe_float)
        lt = np.where((lt <= 0) | np.isnan(lt), df["sku"].map(sku_map), lt)
        lt = np.where((lt <= 0) | np.isnan(lt), df["vendor"].map(ven_map), lt)
        lt = np.where((lt <= 0) | np.isnan(lt), df["category"].map(cat_map), lt)
        df["lead_time_days"] = pd.to_numeric(lt, errors="coerce")

    df["lead_time_days"] = df["lead_time_days"].fillna(0).apply(_safe_int)
    df["expected_arrival_date"] = df["ds"]
    df["planned_po_date"] = df["expected_arrival_date"] - pd.to_timedelta(df["lead_time_days"], unit="D")
    df["unit_cost"] = df["purchase_price"].apply(_safe_float)
    df["po_cost"] = df["po_qty"] * df["unit_cost"]

    if not vendors.empty:
        df = df.merge(vendors, on="vendor", how="left")
    else:
        df["terms"] = np.nan

    df["terms_days"] = df["terms"].apply(_parse_terms_days)
    df["terms_unparsed"] = df["terms_days"].isna()
    df["terms_days"] = df["terms_days"].fillna(0).astype(int)  # conservative worst-case if unknown

    base = df["planned_po_date"] if terms_basis == "po_date" else df["expected_arrival_date"]
    df["payment_date"] = base + pd.to_timedelta(df["terms_days"], unit="D")
    df["payment_month"] = df["payment_date"].dt.to_period("M").dt.to_timestamp()

    po = df[
        [
            "planned_po_date",
            "expected_arrival_date",
            "sku",
            "category",
            "vendor",
            "po_qty",
            "unit_cost",
            "po_cost",
            "lead_time_days",
            "terms",
            "terms_unparsed",
            "payment_date",
            "payment_month",
        ]
    ].sort_values(["planned_po_date", "vendor", "sku"])

    cash = po.groupby("payment_month", as_index=False)["po_cost"].sum().rename(columns={"po_cost": "cash_out"})
    return po, cash


# -------------------------
# PO API (deliveries & tracking)
# -------------------------
@dataclass
class POApiConfig:
    base_url: str
    token: str
    timeout_s: int = 20


def po_api_config() -> Optional[POApiConfig]:
    base = st.secrets.get("PO_API_BASE_URL")
    token = st.secrets.get("PO_API_TOKEN")
    if "po_api" in st.secrets and isinstance(st.secrets["po_api"], dict):
        base = base or st.secrets["po_api"].get("base_url")
        token = token or st.secrets["po_api"].get("token")
    if not base or not token:
        return None
    return POApiConfig(str(base).rstrip("/"), str(token))


@st.cache_data(ttl=10 * 60, show_spinner=False)
def fetch_po_api(cfg: POApiConfig) -> pd.DataFrame:
    url = f"{cfg.base_url}/purchase-orders"
    resp = requests.get(url, headers={"Authorization": f"Bearer {cfg.token}", "Accept": "application/json"}, timeout=cfg.timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for po in data:
        if not isinstance(po, dict):
            continue
        rows.append(
            {
                "po_number": po.get("po_number") or po.get("number") or po.get("id"),
                "vendor": po.get("vendor") or po.get("supplier"),
                "status": po.get("status"),
                "tracking_number": po.get("tracking_number") or po.get("tracking") or po.get("trackingNo"),
                "carrier": po.get("carrier"),
                "expected_delivery_date": po.get("expected_delivery_date") or po.get("eta") or po.get("expected_arrival_date"),
                "delivered_date": po.get("delivered_date") or po.get("deliveredAt"),
                "received_date": po.get("received_date") or po.get("receivedAt"),
                "last_update": po.get("last_update") or po.get("updated_at") or po.get("updatedAt"),
            }
        )
    df = pd.DataFrame(rows)
    for c in ("expected_delivery_date", "delivered_date", "received_date", "last_update"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# -------------------------
# Visualization helpers
# -------------------------
def kpis(items: List[Tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (k, v) in zip(cols, items):
        col.metric(k, v)


def layered_line(df: pd.DataFrame, title: str, height: int = 340):
    if df.empty:
        st.info("No data to display.")
        return
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("ds:T", title=""),
            y=alt.Y("value:Q", title=""),
            color=alt.Color("series:N", legend=alt.Legend(title="")),
            tooltip=["ds:T", "series:N", "value:Q"],
        )
        .properties(title=title, height=height)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def bar(df: pd.DataFrame, x: str, y: str, title: str, height: int = 280):
    if df.empty:
        st.info("No data to display.")
        return
    chart = alt.Chart(df).mark_bar().encode(x=alt.X(x, title=""), y=alt.Y(y, title=""), tooltip=[x, y]).properties(title=title, height=height).interactive()
    st.altair_chart(chart, use_container_width=True)


# -------------------------
# Forecast builders (bottom-up + top-down)
# -------------------------
def bottom_up_forecast(demand: pd.DataFrame, scope: Dict[str, Any], cfg: ForecastConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = demand.copy()
    if scope.get("customer"):
        df = df[df["customer"] == scope["customer"]]
    if scope.get("category"):
        df = df[df["category"] == scope["category"]]
    if scope.get("sku"):
        df = df[df["sku"] == scope["sku"]]
    if df.empty:
        return pd.DataFrame(), {"note": "no demand in scope"}

    group_cols = ["sku"] + (["customer"] if scope.get("customer") else [])
    out_rows, metas = [], []
    skipped_count = 0
    
    for keys, g in df.groupby(group_cols):
        g = g.groupby("ds", as_index=False)["qty"].sum().sort_values("ds")
        
        # Skip items with insufficient data (need at least 2 data points for forecasting)
        if len(g) < 2:
            skipped_count += 1
            continue
        
        if cfg.model == "ml":
            hist = g.copy()
            if isinstance(keys, tuple):
                hist["sku"] = keys[0]
                if "customer" in group_cols:
                    hist["customer"] = keys[1]
            else:
                hist["sku"] = keys
            fc, meta = forecast_ml(hist.assign(ds=pd.to_datetime(hist["ds"])), "qty", cfg, cat_cols=[c for c in ("sku", "customer") if c in hist.columns])
        else:
            y = g.set_index("ds")["qty"]
            fc, meta = forecast_univariate(y, cfg)

        fc_df = fc.reset_index()
        fc_df.columns = ["ds", "qty_forecast"]
        if isinstance(keys, tuple):
            for k, c in zip(keys, group_cols):
                fc_df[c] = k
        else:
            fc_df[group_cols[0]] = keys
        out_rows.append(fc_df)
        metas.append({"keys": keys, "meta": meta})

    if skipped_count > 0:
        st.info(f"ℹ️ Skipped {skipped_count} items with insufficient data (< 2 data points)")

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    return out, {"model": cfg.model, "series": metas, "skipped": skipped_count}


def top_down_sales_forecast_units(
    inv_hist: pd.DataFrame,
    pipe_hist: pd.DataFrame,
    demand: pd.DataFrame,
    unit_prices: pd.DataFrame,
    category: Optional[str],
    cfg: ForecastConfig,
    mix_lookback: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if inv_hist.empty:
        return pd.DataFrame(), {"note": "invoice line history empty"}
    df = inv_hist.copy()
    if category and "category" in df.columns:
        df = df[df["category"] == category]
    if df.empty:
        return pd.DataFrame(), {"note": "no invoice history in scope"}

    # forecast revenue at category level
    if "category" not in df.columns:
        df["category"] = "All Categories"
    rev = df.groupby(["ds", "category"], as_index=False)["revenue"].sum().sort_values("ds")
    y = rev.set_index("ds")["revenue"]
    # keep revenue model stable even if user selected ML
    stable_cfg = cfg if cfg.model in ("exp", "arima") else ForecastConfig(model="exp", horizon=cfg.horizon, freq=cfg.freq, winsorize=cfg.winsorize, lo_q=cfg.lo_q, hi_q=cfg.hi_q)
    rev_fc, rev_meta = forecast_univariate(y, stable_cfg)
    rev_fc_df = rev_fc.reset_index()
    rev_fc_df.columns = ["ds", "revenue_fc"]
    rev_fc_df["category"] = category or rev["category"].iloc[0]

    # add pipeline amounts (category-level)
    pipe = pipe_hist.copy()
    if not pipe.empty:
        if category:
            pipe = pipe[pipe["category"] == category]
        else:
            pipe = pipe.groupby("ds", as_index=False)["pipeline_amount"].sum().assign(category="All Categories")
    else:
        pipe = pd.DataFrame(columns=["ds", "category", "pipeline_amount"])
    rev_fc_df = rev_fc_df.merge(pipe, on=["ds", "category"], how="left")
    rev_fc_df["pipeline_amount"] = rev_fc_df["pipeline_amount"].fillna(0.0)
    rev_fc_df["sales_revenue_total"] = rev_fc_df["revenue_fc"] + rev_fc_df["pipeline_amount"]

    # allocate revenue to SKUs using recent mix
    mix = sku_mix(demand, category, mix_lookback)
    alloc = allocate_category_to_skus(rev_fc_df[["ds", "category", "sales_revenue_total"]].rename(columns={"sales_revenue_total": "alloc_revenue"}), mix, "alloc_revenue")

    # convert revenue -> units (if possible)
    if unit_prices.empty:
        # if no unit price, return revenue-proxy units and label
        alloc["qty_sales"] = alloc["alloc_revenue"]
        return alloc[["ds", "sku", "qty_sales"]], {"note": "Unit price unavailable; sales forecast returned in revenue units (proxy).", "revenue_model": rev_meta}

    alloc = alloc.merge(unit_prices, on="sku", how="left")
    alloc["avg_unit_price"] = pd.to_numeric(alloc["avg_unit_price"], errors="coerce")
    # fallback ASP = category mean
    if alloc["avg_unit_price"].isna().any():
        asp_cat = unit_prices.merge(df[["sku", "category"]].drop_duplicates(), on="sku", how="left").groupby("category")["avg_unit_price"].mean().to_dict()
        alloc["avg_unit_price"] = alloc["avg_unit_price"].fillna((category and asp_cat.get(category)) or np.nan)
    alloc["avg_unit_price"] = alloc["avg_unit_price"].replace([0, np.inf, -np.inf], np.nan)
    alloc["qty_sales"] = (alloc["alloc_revenue"] / alloc["avg_unit_price"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return alloc[["ds", "sku", "qty_sales"]], {"revenue_model": rev_meta}


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Global Controls")
    try:
        SHEET_ID = _get_spreadsheet_id()
        st.success("Google Sheets configured")
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.button("🔄 Refresh data (clear cache)"):
        clear_cache()
        st.rerun()

    freq = st.selectbox("Frequency", options=["MS"], index=0)
    horizon = st.selectbox("Forecast horizon (months)", options=[3, 6, 9, 12, 18, 24], index=3)

    model = st.selectbox(
        "Forecast model",
        options=[("exp", "Exponential Smoothing"), ("arima", "ARIMA/SARIMA"), ("ml", "Machine Learning")],
        format_func=lambda x: x[1],
    )[0]

    wins = st.checkbox("Clip extreme outliers (winsorize)", value=True)
    lo_q = st.slider("Lower quantile", 0.0, 0.10, 0.01, 0.005) if wins else 0.01
    hi_q = st.slider("Upper quantile", 0.90, 1.00, 0.99, 0.005) if wins else 0.99

    cfg = ForecastConfig(model=model, horizon=int(horizon), freq=freq, winsorize=wins, lo_q=float(lo_q), hi_q=float(hi_q))

    if model == "exp":
        st.subheader("Exponential Smoothing")
        cfg.es_trend = st.selectbox("Trend", ["add", "mul", "none"], index=0)
        cfg.es_seasonal = st.selectbox("Seasonality", ["auto", "add", "mul", "none"], index=0)
        cfg.es_damped = st.checkbox("Damped trend", value=True)
        st.caption("Manual α/β/γ optional (blank = auto-fit)")
        a, b, g = st.columns(3)
        with a:
            alpha = st.text_input("α", value="")
        with b:
            beta = st.text_input("β", value="")
        with g:
            gamma = st.text_input("γ", value="")
        cfg.es_alpha = float(alpha) if alpha.strip() else None
        cfg.es_beta = float(beta) if beta.strip() else None
        cfg.es_gamma = float(gamma) if gamma.strip() else None

    if model == "arima":
        st.subheader("ARIMA/SARIMA")
        cfg.arima_auto = st.checkbox("Auto-select (AIC grid)", value=True)
        if not cfg.arima_auto:
            p = st.number_input("p", 0, 5, 1)
            d = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 5, 1)
            P = st.number_input("P", 0, 2, 0)
            D = st.number_input("D", 0, 1, 0)
            Q = st.number_input("Q", 0, 2, 0)
            s = st.number_input("s (seasonal period)", 0, 52, 12)
            cfg.order = (int(p), int(d), int(q))
            cfg.seasonal_order = (int(P), int(D), int(Q), int(s))
        cfg.enforce_stationarity = st.checkbox("Enforce stationarity", value=False)
        cfg.enforce_invertibility = st.checkbox("Enforce invertibility", value=False)

    if model == "ml":
        st.subheader("Machine Learning")
        cfg.ml_model = st.selectbox("Tree model", ["random_forest", "gradient_boosting"], index=0)
        cfg.ml_min_hist = st.slider("Min history periods before ML", 6, 36, 18, 1)
        cfg.rf_n_estimators = st.slider("RF n_estimators", 100, 1200, 400, 50)
        cfg.rf_max_depth = st.selectbox("RF max_depth", [None, 5, 8, 12, 16, 24], index=0)
        cfg.ml_lags = tuple(st.multiselect("Lag months", [1, 2, 3, 4, 6, 9, 12], default=[1, 2, 3]))
        cfg.ml_rolls = tuple(st.multiselect("Rolling windows", [2, 3, 4, 6, 9, 12], default=[3, 6]))

# Load data
data = load_all(SHEET_ID)

items = build_items(data["items"])
vendors = build_vendors(data["vendors"])
avg_leads = build_avg_lead_map(data["avg_leadtimes"])

dem = demand_history(data["so_line"], items, freq=freq)
inv = invoice_line_history(data["invoice_line"], items, freq=freq)
prices = unit_price_map(inv)
pipe = pipeline_history(data["deals"], freq=freq)
open_inv, cust_pay = invoice_payment_metrics(data["invoices_header"])

# Dimensions
categories = sorted(items["category"].dropna().unique()) if not items.empty else sorted(dem["category"].dropna().unique())
skus = sorted(items["sku"].dropna().unique()) if not items.empty else sorted(dem["sku"].dropna().unique())
customers = sorted(dem["customer"].dropna().unique()) if not dem.empty else []
rep_map = {}
if not data["customers"].empty:
    rep_col = resolve_col(data["customers"], "sales_rep")
    cust_col = resolve_col(data["customers"], "customer")
    if rep_col and cust_col:
        rep_map = (
            data["customers"]
            .assign(_cust=data["customers"][cust_col].astype(str).str.strip(), _rep=data["customers"][rep_col].astype(str).str.strip())
            .dropna(subset=["_cust", "_rep"])
            .drop_duplicates(subset=["_cust"])
            .set_index("_cust")["_rep"]
            .to_dict()
        )
sales_reps = sorted(set(rep_map.values())) if rep_map else []

tabs = st.tabs(["Sales Rep View", "Operations / Supply Chain", "Scenario Planning", "PO Forecast", "Deliveries & Tracking"])


# -------------------------
# TAB 1: Sales Rep View
# -------------------------
with tabs[0]:
    st.subheader("Sales Rep View (customer-safe)")

    c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
    with c1:
        rep = st.selectbox("Sales Rep", ["(All)"] + sales_reps, index=0)
    with c2:
        if rep != "(All)" and rep_map:
            custs = sorted([c for c, r in rep_map.items() if r == rep])
            customer = st.selectbox("Customer", custs if custs else ["(None)"])
        else:
            customer = st.selectbox("Customer", ["(All)"] + customers if customers else ["(None)"])
    with c3:
        pass

    if customer in ("(None)",):
        st.info("No customers available in demand history.")
        st.stop()

    # Restrict SKUs to customer's history if customer selected
    if customer not in ("(All)", "(None)") and not dem.empty:
        cust_dem = dem[dem["customer"] == customer].copy()
        cust_skus = sorted(cust_dem["sku"].unique())
    else:
        cust_dem = dem.copy()
        cust_skus = skus

    sku = st.selectbox("SKU (optional)", ["(All customer SKUs)"] + cust_skus, index=0)

    scope = {"customer": None if customer == "(All)" else customer, "sku": None}
    if sku != "(All customer SKUs)":
        scope["sku"] = sku

    if cust_dem.empty:
        st.info("No demand history for selection.")
    else:
        # Cadence by SKU (median days between non-zero months)
        cadence_rows = []
        for s, g in cust_dem.groupby("sku"):
            if sku != "(All customer SKUs)" and s != sku:
                continue
            g2 = g.groupby("ds", as_index=False)["qty"].sum().sort_values("ds")
            g2 = g2[g2["qty"] > 0]
            if g2.shape[0] < 2:
                continue
            deltas = g2["ds"].diff().dt.days.dropna()
            cadence_rows.append(
                {
                    "sku": s,
                    "orders_count": int(g2.shape[0]),
                    "median_days_between_orders": float(deltas.median()) if not deltas.empty else np.nan,
                    "last_order_date": g2["ds"].max().date(),
                    "last_order_qty": float(g2.loc[g2["ds"].idxmax(), "qty"]),
                }
            )
        cadence = pd.DataFrame(cadence_rows).sort_values(["orders_count", "median_days_between_orders"], ascending=[False, True])

        # Forecast (bottom-up)
        fc, meta = bottom_up_forecast(dem, scope, cfg)
        if not fc.empty:
            fc = fc.groupby("ds", as_index=False)["qty_forecast"].sum()

        hist = cust_dem.copy()
        if scope["customer"]:
            hist = hist[hist["customer"] == scope["customer"]]
        if scope["sku"]:
            hist = hist[hist["sku"] == scope["sku"]]
        hist = hist.groupby("ds", as_index=False)["qty"].sum().sort_values("ds")

        total_12m = hist[hist["ds"] >= (hist["ds"].max() - pd.DateOffset(months=12))]["qty"].sum() if not hist.empty else 0.0
        open_cnt = int(open_inv[open_inv["customer"] == customer].shape[0]) if customer not in ("(All)", "(None)") else int(open_inv.shape[0])
        kpis(
            [
                ("Customer", customer),
                ("12-mo units (actual)", _human_qty(total_12m)),
                ("SKUs in scope", str(int(cust_dem["sku"].nunique()))),
                ("Open invoices", str(open_cnt)),
            ]
        )

        st.markdown("### Historical SKU ordering cadence")
        if cadence.empty:
            st.info("Not enough repeat orders to compute cadence.")
        else:
            st.dataframe(cadence, use_container_width=True, hide_index=True)

        st.markdown("### Invoice payment behavior")
        left, right = st.columns([1.3, 1.7])
        with left:
            if customer not in ("(All)", "(None)"):
                row = cust_pay[cust_pay["customer"] == customer]
            else:
                row = cust_pay
            if row.empty:
                st.info("No payment summary available from invoice header data.")
            else:
                r = row.iloc[0].to_dict()
                kpis([("Avg days to pay", f"{r.get('avg_days_to_pay', 0):.1f}"), ("P90 days to pay", f"{r.get('p90_days_to_pay', 0):.1f}"), ("Open balance", _human_money(r.get("total_open", 0)))])
        with right:
            oi = open_inv[open_inv["customer"] == customer] if customer not in ("(All)", "(None)") else open_inv
            if oi.empty:
                st.success("No open invoices in scope.")
            else:
                st.dataframe(oi[["invoice_date", "due_date", "amount", "days_past_due"]].head(25), use_container_width=True, hide_index=True)

        st.markdown("### Demand forecast (historical + forecast)")
        plot = []
        if not hist.empty:
            plot.append(hist.assign(series="Actual", value=hist["qty"])[["ds", "series", "value"]])
        if not fc.empty:
            plot.append(fc.assign(series="Forecast", value=fc["qty_forecast"])[["ds", "series", "value"]])
        plot_df = pd.concat(plot, ignore_index=True) if plot else pd.DataFrame()
        layered_line(plot_df, "Historical vs Forecast demand (Units)")

        if not fc.empty:
            fc2 = fc.copy()
            fc2["quarter"] = pd.to_datetime(fc2["ds"]).dt.to_period("Q").astype(str)
            rec = fc2.groupby("quarter", as_index=False)["qty_forecast"].sum().sort_values("quarter")
            st.markdown("### Quarterly recommendations (next 4 quarters)")
            st.dataframe(rec, use_container_width=True, hide_index=True)

        if cfg.model == "ml":
            st.markdown("### Explainability (ML feature importance)")
            fi = None
            for s in meta.get("series", []):
                m = s.get("meta", {})
                if m and "feature_importance" in m:
                    fi = pd.DataFrame(m["feature_importance"])
                    break
            if fi is None or fi.empty:
                st.info("Feature importance not available (ML may have fallen back to smoothing).")
            else:
                st.dataframe(fi.head(15), use_container_width=True, hide_index=True)


# -------------------------
# TAB 2: Operations / Supply Chain
# -------------------------
with tabs[1]:
    st.subheader("Operations / Supply Chain View")

    a, b, c, d = st.columns([1.3, 1.4, 1.1, 1.5])
    with a:
        cat = st.selectbox("Product Category", ["(All)"] + categories, index=0)
    with b:
        sku_list = skus if cat == "(All)" else sorted(items[items["category"] == cat]["sku"].dropna().unique())
        sku_sel = st.selectbox("SKU", sku_list if sku_list else ["(None)"])
    with c:
        metric = st.selectbox("Metric", ["Units", "Revenue (approx)"], index=0)
    with d:
        mix_lb = st.slider("SKU mix lookback (months)", 3, 24, 6, 1)

    if sku_sel == "(None)":
        st.info("No SKUs for this selection.")
    else:
        # actual demand
        df_hist = dem.copy()
        if cat != "(All)":
            df_hist = df_hist[df_hist["category"] == cat]
        df_hist = df_hist[df_hist["sku"] == sku_sel].groupby("ds", as_index=False)["qty"].sum().sort_values("ds")

        if df_hist.empty:
            st.info("No demand history found for this SKU/category.")
        else:
            # bottom-up forecast
            scope = {"category": None if cat == "(All)" else cat, "sku": sku_sel}
            bu_fc, _ = bottom_up_forecast(dem, scope, cfg)
            bu_fc = bu_fc.groupby("ds", as_index=False)["qty_forecast"].sum() if not bu_fc.empty else pd.DataFrame()

            # pipeline trend (category-level) forecast + allocation
            pipe_cat = pipe.copy()
            if cat != "(All)":
                pipe_cat = pipe_cat[pipe_cat["category"] == cat]
            else:
                pipe_cat = pipe_cat.groupby("ds", as_index=False)["pipeline_amount"].sum().assign(category="All Categories")
            y_pipe = pipe_cat.set_index("ds")["pipeline_amount"] if not pipe_cat.empty else pd.Series(dtype=float)
            pipe_fc, _ = forecast_exp(y_pipe, ForecastConfig(model="exp", horizon=int(horizon), freq=freq, winsorize=wins, lo_q=float(lo_q), hi_q=float(hi_q)))
            pipe_fc_df = pipe_fc.reset_index()
            pipe_fc_df.columns = ["ds", "pipeline_fc"]
            pipe_fc_df["category"] = cat if cat != "(All)" else "All Categories"
            mix = sku_mix(dem, None if cat == "(All)" else cat, mix_lb)
            pipe_alloc = allocate_category_to_skus(pipe_fc_df[["ds", "category", "pipeline_fc"]], mix, "pipeline_fc")
            pipe_alloc_sku = pipe_alloc[pipe_alloc["sku"] == sku_sel].rename(columns={"pipeline_fc": "pipeline_alloc"})

            # top-down sales forecast (revenue+pipeline) allocated -> units for SKU
            sales_fc, sales_meta = top_down_sales_forecast_units(inv, pipe, dem, prices, None if cat == "(All)" else cat, cfg, mix_lb)
            sales_fc_sku = sales_fc[sales_fc["sku"] == sku_sel].copy()

            # metric conversion
            sku_price = None
            if not prices.empty and sku_sel in set(prices["sku"]):
                sku_price = float(prices[prices["sku"] == sku_sel]["avg_unit_price"].iloc[0])

            def to_metric(series: pd.Series) -> pd.Series:
                if metric == "Units":
                    return series
                if sku_price and sku_price > 0:
                    return series * sku_price
                return series  # if no price, may already be revenue-proxy

            plot_parts = []
            plot_parts.append(df_hist.assign(series="Actual", value=to_metric(df_hist["qty"]))[["ds", "series", "value"]])
            if not bu_fc.empty:
                plot_parts.append(bu_fc.assign(series="Bottom-up forecast", value=to_metric(bu_fc["qty_forecast"]))[["ds", "series", "value"]])
            if not pipe_alloc_sku.empty:
                plot_parts.append(pipe_alloc_sku.assign(series="Pipeline allocated", value=to_metric(pipe_alloc_sku["pipeline_alloc"]))[["ds", "series", "value"]])
            if not sales_fc_sku.empty:
                plot_parts.append(sales_fc_sku.assign(series="Top-down sales+pipeline (allocated)", value=to_metric(sales_fc_sku["qty_sales"]))[["ds", "series", "value"]])

            plot_df = pd.concat(plot_parts, ignore_index=True)
            layered_line(plot_df, f"Demand vs Forecast vs Pipeline ({metric})", height=380)

            # gap table
            comp = (
                (bu_fc.rename(columns={"qty_forecast": "bottom_up"}).merge(pipe_alloc_sku[["ds", "pipeline_alloc"]] if not pipe_alloc_sku.empty else pd.DataFrame(columns=["ds", "pipeline_alloc"]), on="ds", how="outer"))
                .merge(sales_fc_sku.rename(columns={"qty_sales": "top_down"})[["ds", "top_down"]] if not sales_fc_sku.empty else pd.DataFrame(columns=["ds", "top_down"]), on="ds", how="outer")
                .fillna(0.0)
                .sort_values("ds")
            )
            comp["pipeline_minus_bottom"] = comp.get("pipeline_alloc", 0.0) - comp["bottom_up"]
            comp["top_down_minus_bottom"] = comp.get("top_down", 0.0) - comp["bottom_up"]

            kpis(
                [
                    ("Bottom-up total", _human_money(to_metric(comp["bottom_up"]).sum()) if metric != "Units" else _human_qty(comp["bottom_up"].sum())),
                    ("Pipeline - Bottom-up", _human_money(to_metric(comp["pipeline_minus_bottom"]).sum()) if metric != "Units" else _human_qty(comp["pipeline_minus_bottom"].sum())),
                    ("Top-down - Bottom-up", _human_money(to_metric(comp["top_down_minus_bottom"]).sum()) if metric != "Units" else _human_qty(comp["top_down_minus_bottom"].sum())),
                ]
            )
            st.markdown("### Monthly comparison")
            st.dataframe(comp, use_container_width=True, hide_index=True)
            if "note" in sales_meta:
                st.info(sales_meta["note"])


# -------------------------
# TAB 3: Scenario Planning
# -------------------------
with tabs[2]:
    st.subheader("Scenario Planning (S&OP)")
    st.caption("Blend bottom-up demand and top-down sales forecasts, adjust growth, save/load/compare, and approve a final scenario.")

    left, right = st.columns([1.1, 1.9])
    with left:
        scope_mode = st.selectbox("Scenario scope", ["All SKUs", "Single Category"], index=1)
        scen_cat = None
        if scope_mode == "Single Category":
            scen_cat = st.selectbox("Category", categories if categories else ["Uncategorized"])
        growth = st.slider("Growth adjustment", -0.5, 0.5, 0.0, 0.01, format="%.2f")
        w_demand = st.slider("Blend: % Demand forecast", 0, 100, 70, 5)
        mix_lb = st.slider("SKU mix lookback (months)", 3, 24, 6, 1)
        scen_name = st.text_input("Scenario name", value=f"Scenario {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    cat_scope = scen_cat if scope_mode == "Single Category" else None

    # Compute forecasts
    bu, bu_meta = bottom_up_forecast(dem, {"category": cat_scope, "sku": None, "customer": None}, cfg)
    bu = bu.groupby(["ds", "sku"], as_index=False)["qty_forecast"].sum() if not bu.empty else pd.DataFrame(columns=["ds", "sku", "qty_forecast"])

    td, td_meta = top_down_sales_forecast_units(inv, pipe, dem, prices, cat_scope, cfg, mix_lb)
    td = td.groupby(["ds", "sku"], as_index=False)["qty_sales"].sum() if not td.empty else pd.DataFrame(columns=["ds", "sku", "qty_sales"])

    scen = blend_forecasts(bu, td, float(w_demand) / 100.0, float(growth))

    # Cash implications from PO plan
    po_preview, cash_preview = build_po_plan(scen.rename(columns={"qty_final": "qty_forecast"}), items, vendors, avg_leads, terms_basis="po_date")

    with right:
        hist = dem.copy()
        if cat_scope:
            hist = hist[hist["category"] == cat_scope]
        hist = hist.groupby("ds", as_index=False)["qty"].sum().assign(series="Actual", value=lambda d: d["qty"])
        scen_agg = scen.groupby("ds", as_index=False)["qty_final"].sum().assign(series="Scenario", value=lambda d: d["qty_final"])
        plot = pd.concat([hist[["ds", "series", "value"]], scen_agg[["ds", "series", "value"]]], ignore_index=True)
        layered_line(plot, "Actual vs Scenario forecast (Units)", height=320)

        kpis(
            [
                ("Scope", "All SKUs" if scope_mode == "All SKUs" else scen_cat),
                ("Scenario units (sum)", _human_qty(scen["qty_final"].sum() if not scen.empty else 0)),
                ("Est. PO cash out (sum)", _human_money(cash_preview["cash_out"].sum() if not cash_preview.empty else 0)),
                ("SKUs", str(int(scen["sku"].nunique())) if not scen.empty else "0"),
            ]
        )
        if not cash_preview.empty:
            bar(cash_preview, "payment_month:T", "cash_out:Q", "Cash out timeline (payments by month)")
        else:
            st.info("Cash preview unavailable (check item costs / vendor mapping).")

        st.markdown("### Scenario preview tables")
        t1, t2 = st.columns(2)
        with t1:
            st.caption("SKU-level scenario forecast (head)")
            st.dataframe(scen.head(200), use_container_width=True, hide_index=True)
        with t2:
            st.caption("PO plan preview (head)")
            st.dataframe(po_preview.head(200), use_container_width=True, hide_index=True)

    st.markdown("---")
    s1, s2, s3 = st.columns([1.2, 1.4, 1.4])
    assumptions = {
        "scope_mode": scope_mode,
        "category": scen_cat,
        "growth_rate": float(growth),
        "blend_w_demand": float(w_demand) / 100.0,
        "blend_w_sales": 1.0 - (float(w_demand) / 100.0),
        "forecast_model": cfg.model,
        "horizon": cfg.horizon,
        "mix_lookback_months": int(mix_lb),
        "created_at_utc": _now().isoformat(),
    }

    with s1:
        if st.button("💾 Save scenario"):
            if scen.empty:
                st.error("Scenario is empty; cannot save.")
            else:
                sid = save_scenario(scen_name.strip() or "Scenario", assumptions, scen.rename(columns={"qty_final": "qty_forecast"}), cash_preview)
                st.success(f"Saved scenario: {sid}")

    with s2:
        idx = list_scenarios()
        if idx:
            sid_sel = st.selectbox("Load scenario", [r["scenario_id"] for r in idx], format_func=lambda sid: next((r["name"] for r in idx if r["scenario_id"] == sid), sid))
            if st.button("📂 Load selected"):
                rec, fdf, cdf = load_scenario(sid_sel)
                st.session_state["loaded_rec"] = rec
                st.session_state["loaded_forecast"] = fdf
                st.session_state["loaded_cash"] = cdf
                st.success(f"Loaded: {rec['name'] if rec else sid_sel}")
        else:
            st.info("No saved scenarios yet.")

    with s3:
        appr = approved_scenario_id()
        st.caption(f"Approved scenario: {appr or 'None'}")
        idx = list_scenarios()
        if idx:
            sid_appr = st.selectbox("Select scenario to approve", [r["scenario_id"] for r in idx], format_func=lambda sid: next((r["name"] for r in idx if r["scenario_id"] == sid), sid))
            if st.button("✅ Approve scenario"):
                approve_scenario(sid_appr)
                st.success(f"Approved scenario set: {sid_appr}")

    st.markdown("### Compare scenarios")
    idx = list_scenarios()
    if len(idx) >= 2:
        sids = [r["scenario_id"] for r in idx]
        a = st.selectbox("Scenario A", sids, index=max(0, len(sids) - 1), key="cmpA")
        b = st.selectbox("Scenario B", sids, index=max(0, len(sids) - 2), key="cmpB")

        _, fa, ca = load_scenario(a)
        _, fb, cb = load_scenario(b)
        if not fa.empty and not fb.empty:
            ua, ub = fa["qty_forecast"].sum(), fb["qty_forecast"].sum()
            coa, cob = ca["cash_out"].sum() if not ca.empty else 0.0, cb["cash_out"].sum() if not cb.empty else 0.0
            kpis([("A units", _human_qty(ua)), ("B units", _human_qty(ub)), ("Δ units (B-A)", _human_qty(ub - ua)), ("Δ cash out (B-A)", _human_money(cob - coa))])

            a_ts = fa.groupby("ds", as_index=False)["qty_forecast"].sum().rename(columns={"qty_forecast": "A"})
            b_ts = fb.groupby("ds", as_index=False)["qty_forecast"].sum().rename(columns={"qty_forecast": "B"})
            comp = a_ts.merge(b_ts, on="ds", how="outer").fillna(0.0)
            comp_long = pd.concat(
                [comp[["ds", "A"]].rename(columns={"A": "value"}).assign(series="Scenario A"), comp[["ds", "B"]].rename(columns={"B": "value"}).assign(series="Scenario B")],
                ignore_index=True,
            )
            layered_line(comp_long, "Scenario comparison (Units)", height=320)


# -------------------------
# TAB 4: PO Forecast
# -------------------------
with tabs[3]:
    st.subheader("Purchase Order Forecast")
    appr = approved_scenario_id()
    if not appr:
        st.warning("No approved scenario set. Approve one in **Scenario Planning**.")
    else:
        rec, fdf, _ = load_scenario(appr)
        if rec is None or fdf.empty:
            st.error("Approved scenario could not be loaded (missing files).")
        else:
            st.caption(f"Using approved scenario: **{rec['name']}** ({appr})")
            basis = st.selectbox("Payment terms basis", [("po_date", "From PO date (conservative)"), ("arrival_date", "From arrival date (optimistic)")], format_func=lambda x: x[1])[0]
            po, cash = build_po_plan(fdf.rename(columns={"qty_forecast": "qty_forecast"}), items, vendors, avg_leads, terms_basis=basis)

            if po.empty:
                st.error("PO plan is empty. Check item master mapping (vendor/cost/lead time).")
            else:
                kpis(
                    [
                        ("PO lines", str(int(po.shape[0]))),
                        ("Total PO cost", _human_money(po["po_cost"].sum())),
                        ("Vendors", str(int(po["vendor"].nunique()))),
                        ("Avg lead time (days)", f"{po['lead_time_days'].mean():.1f}"),
                    ]
                )
                st.markdown("### PO Forecast Table (export-ready)")
                st.dataframe(po, use_container_width=True, hide_index=True)

                st.download_button("⬇️ Download PO forecast (CSV)", po.to_csv(index=False).encode("utf-8"), file_name=f"po_forecast__{appr}.csv", mime="text/csv")

                st.markdown("### Cash flow timeline (payments)")
                if cash.empty:
                    st.info("No cash timeline available.")
                else:
                    bar(cash, "payment_month:T", "cash_out:Q", "Projected cash out by month")
                    st.dataframe(cash, use_container_width=True, hide_index=True)


# -------------------------
# TAB 5: Deliveries & Tracking
# -------------------------
with tabs[4]:
    st.subheader("Upcoming Deliveries & Tracking (PO API)")
    cfg_po = po_api_config()
    if not cfg_po:
        st.error(
            "PO API is not configured.\n\n"
            "Add Streamlit secrets:\n"
            "- PO_API_BASE_URL\n"
            "- PO_API_TOKEN\n\n"
            "Expected endpoint: GET {base_url}/purchase-orders returning JSON."
        )
    else:
        try:
            po_df = fetch_po_api(cfg_po)
        except Exception as e:
            st.error(f"Failed to fetch PO API data: {e}")
            po_df = pd.DataFrame()

        if po_df.empty:
            st.info("No PO records returned from API.")
        else:
            today = pd.Timestamp(date.today())
            po_df = po_df.copy()
            po_df["is_delayed"] = po_df["expected_delivery_date"].notna() & (po_df["expected_delivery_date"] < today) & po_df["delivered_date"].isna()
            po_df["delivered_not_received"] = po_df["delivered_date"].notna() & po_df["received_date"].isna()

            f1, f2, f3 = st.columns([1.2, 1.2, 1.4])
            with f1:
                ven_opts = sorted([v for v in po_df["vendor"].dropna().unique()])
                ven_sel = st.selectbox("Vendor", ["(All)"] + ven_opts, index=0)
            with f2:
                st_opts = sorted([s for s in po_df["status"].dropna().unique()])
                st_sel = st.selectbox("Status", ["(All)"] + st_opts, index=0)
            with f3:
                exceptions_only = st.checkbox("Exceptions only", value=True)

            view = po_df.copy()
            if ven_sel != "(All)":
                view = view[view["vendor"] == ven_sel]
            if st_sel != "(All)":
                view = view[view["status"] == st_sel]
            if exceptions_only:
                view = view[view["is_delayed"] | view["delivered_not_received"]]

            kpis(
                [
                    ("POs", str(int(po_df.shape[0]))),
                    ("Exceptions", str(int((po_df["is_delayed"] | po_df["delivered_not_received"]).sum()))),
                    ("Delayed", str(int(po_df["is_delayed"].sum()))),
                    ("Delivered not received", str(int(po_df["delivered_not_received"].sum()))),
                ]
            )

            st.markdown("### Tracking table")
            st.dataframe(view.sort_values(["is_delayed", "expected_delivery_date"], ascending=[False, True]), use_container_width=True, hide_index=True)

            st.markdown("### Flags")
            st.write(
                "- **Delayed**: expected delivery date is in the past and shipment is not delivered.\n"
                "- **Delivered but not received**: carrier says delivered, but receipt is missing."
            )
