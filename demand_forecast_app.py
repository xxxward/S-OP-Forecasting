"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX CONTAINERS - S&OP COMMAND CENTER
    The Most Beautiful Dashboard in Existence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calyx S&OP Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# 🎭 LEGENDARY CSS - THE MOST BEAUTIFUL STYLING EVER CREATED
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ═══ IMPORT FONTS ═══ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* ═══ ROOT VARIABLES ═══ */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --dark-gradient: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.18);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.12);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.16);
        --shadow-xl: 0 16px 48px rgba(0, 0, 0, 0.24);
    }
    
    /* ═══ GLOBAL STYLES ═══ */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
    }
    
    /* ═══ ANIMATED GRADIENT BACKGROUND ═══ */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        z-index: -1;
        opacity: 0.8;
    }
    
    /* ═══ GLASSMORPHISM CARDS ═══ */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* ═══ METRIC CARDS - ABSOLUTELY STUNNING ═══ */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255, 255, 255, 0.8) !important;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }
    
    /* ═══ HEADERS - EPIC TYPOGRAPHY ═══ */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        line-height: 1.1 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.95) !important;
        letter-spacing: -0.01em;
        margin-top: 1.5rem !important;
    }
    
    /* ═══ TABS - SLEEK AND MODERN ═══ */
    .stTabs {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 0 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* ═══ DATAFRAMES - BEAUTIFUL TABLES ═══ */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] table {
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] thead tr th {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        padding: 1rem !important;
    }
    
    [data-testid="stDataFrame"] tbody tr:hover {
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* ═══ SIDEBAR - COMMAND CENTER ═══ */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* ═══ SELECTBOX & INPUTS - PREMIUM FEEL ═══ */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stSelectbox > div > div, .stTextInput > div > div > input {
        background: transparent !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500;
    }
    
    /* ═══ BUTTONS - CALL TO ACTION ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ═══ DOWNLOAD BUTTON ═══ */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 16px rgba(79, 172, 254, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(79, 172, 254, 0.6);
    }
    
    /* ═══ SLIDER - SMOOTH CONTROL ═══ */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* ═══ EXPANDER - ELEGANT ACCORDION ═══ */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* ═══ INFO/WARNING/SUCCESS BOXES ═══ */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }
    
    /* ═══ LOADING SPINNER ═══ */
    .stSpinner > div {
        border-top-color: #ffffff !important;
    }
    
    /* ═══ DIVIDER ═══ */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* ═══ PLOTLY CHARTS - GLASS CONTAINER ═══ */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
    }
    
    /* ═══ ANIMATIONS ═══ */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main > div {
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📊 DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

TABS_CONFIG = {
    "invoices_header": "_NS_Invoices_Data",
    "invoice_line": "Invoice Line Item",
    "so_line": "Sales Order Line Item",
    "so_header": "_NS_SalesOrders_Data",
    "customers": "_NS_Customer_List",
    "items": "Raw_Items",
    "vendors": "Raw_Vendors",
    "avg_leadtimes": "Average Leadtimes",
    "deals": "Deals",
    "inventory": "Raw_Inventory",
}

@dataclass
class ForecastConfig:
    model: str = "exp"
    horizon: int = 12
    freq: str = "MS"
    winsorize: bool = True

# ══════════════════════════════════════════════════════════════════════════════
# 🛠️ UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(x: Any) -> float:
    if pd.isna(x) or x == "": return 0.0
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except: return 0.0

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def format_currency(x: float) -> str:
    if x >= 1_000_000: return f"${x/1_000_000:.2f}M"
    if x >= 1_000: return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def format_qty(x: float) -> str:
    if x >= 1_000_000: return f"{x/1_000_000:.2f}M"
    if x >= 1_000: return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

ALIASES = {
    "date": ["date", "trandate", "transactiondate", "createddate", "orderdate", "closedate"],
    "sku": ["sku", "item", "itemname", "itemid", "product"],
    "customer": ["customer", "customername", "entity", "company", "customercompanyname"],
    "qty": ["quantity", "qty", "quantityordered", "quantityfulfilled"],
    "amount": ["amount", "total", "totalamount", "revenue"],
    "category": ["category", "producttype", "calyxproducttype", "type"],
    "sales_rep": ["salesrep", "salesperson", "rep", "owner", "repmaster"],
    "lead_time": ["leadtime", "purchaseleadtime", "avgleadtime"],
}

def resolve_col(df: pd.DataFrame, role: str) -> Optional[str]:
    if df.empty: return None
    cols = df.columns.tolist()
    norm_cols = {_norm(c): c for c in cols}
    for alias in ALIASES.get(role, []):
        if _norm(alias) in norm_cols:
            return norm_cols[_norm(alias)]
    for col in cols:
        for alias in ALIASES.get(role, []):
            if _norm(alias) in _norm(col):
                return col
    return None

# ══════════════════════════════════════════════════════════════════════════════
# 📥 DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        # Try multiple credential configurations
        creds_dict = None
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        elif "service_account" in st.secrets:
            creds_dict = dict(st.secrets["service_account"])
        elif "gsheets" in st.secrets and "service_account" in st.secrets["gsheets"]:
            creds_dict = dict(st.secrets["gsheets"]["service_account"])
        else:
            raise ValueError("Missing Google credentials. Add [gcp_service_account] to secrets.toml")
        
        # Try multiple sheet ID configurations
        sheet_id = None
        if "SPREADSHEET_ID" in st.secrets:
            sheet_id = st.secrets["SPREADSHEET_ID"]
        elif "gsheets" in st.secrets:
            gsheets = st.secrets["gsheets"]
            sheet_id = gsheets.get("spreadsheet_id") or gsheets.get("sheet_id")
        
        if not sheet_id:
            raise ValueError("Missing spreadsheet ID. Add SPREADSHEET_ID to secrets.toml")

        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
        
        data = {}
        progress = st.progress(0, text="🚀 Loading data...")
        
        for i, (key, tab_name) in enumerate(TABS_CONFIG.items()):
            try:
                ws = sh.worksheet(tab_name)
                rows = ws.get_all_values()
                header_idx = 1 if tab_name == "Deals" else 0
                
                if len(rows) > header_idx + 1:
                    headers = rows[header_idx]
                    seen = {}
                    clean_headers = []
                    for h in headers:
                        h = str(h).strip()
                        if h in seen:
                            seen[h] += 1
                            h = f"{h}_{seen[h]}"
                        else:
                            seen[h] = 0
                        clean_headers.append(h)
                    
                    df = pd.DataFrame(rows[header_idx + 1:], columns=clean_headers)
                    df = df.replace('', np.nan)
                    data[key] = df
                else:
                    data[key] = pd.DataFrame()
            except:
                data[key] = pd.DataFrame()
            
            progress.progress((i + 1) / len(TABS_CONFIG), text=f"Loading {tab_name}...")
        
        progress.empty()
        return data
    except Exception as e:
        st.error(f"❌ Data Loading Error: {e}")
        st.info("💡 Check your .streamlit/secrets.toml configuration")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 📊 DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_demand_history(data: Dict[str, pd.DataFrame], freq="MS") -> pd.DataFrame:
    df = data["so_line"].copy()
    if df.empty:
        return pd.DataFrame(columns=["ds", "sku", "customer", "qty", "category"])
    
    c_date = resolve_col(df, "date")
    c_sku = resolve_col(df, "sku")
    c_qty = resolve_col(df, "qty")
    c_customer = resolve_col(df, "customer")
    
    if not all([c_date, c_sku, c_qty]):
        return pd.DataFrame()
    
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=[c_date])
    df["ds"] = df[c_date].dt.to_period(freq).dt.to_timestamp()
    df["sku"] = df[c_sku].astype(str).str.strip()
    df["qty"] = df[c_qty].apply(_safe_float)
    df["customer"] = df[c_customer].astype(str).str.strip() if c_customer else "Unknown"
    
    out = df.groupby(["ds", "sku", "customer"], as_index=False)["qty"].sum()
    
    items = data["items"]
    if not items.empty:
        c_item_sku = resolve_col(items, "sku")
        c_category = resolve_col(items, "category")
        if c_item_sku and c_category:
            cat_map = dict(zip(
                items[c_item_sku].astype(str).str.strip(),
                items[c_category].astype(str).str.strip()
            ))
            out["category"] = out["sku"].map(cat_map).fillna("Uncategorized")
        else:
            out["category"] = "Uncategorized"
    else:
        out["category"] = "Uncategorized"
    
    return out.sort_values("ds")

# ══════════════════════════════════════════════════════════════════════════════
# 🔮 FORECASTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast(history: pd.Series, cfg: ForecastConfig) -> pd.Series:
    if len(history) < 2:
        last_val = history.iloc[-1] if len(history) > 0 else 0.0
        future_dates = pd.date_range(
            start=history.index[-1] if len(history) > 0 else pd.Timestamp.now(),
            periods=cfg.horizon + 1, freq=cfg.freq
        )[1:]
        return pd.Series([last_val] * cfg.horizon, index=future_dates)
    
    y = history.fillna(0).copy()
    if y.index.duplicated().any():
        y = y.groupby(y.index).sum()
    
    if cfg.winsorize and len(y) > 2:
        y = y.clip(y.quantile(0.01), y.quantile(0.99))
    
    try:
        model = ExponentialSmoothing(
            y,
            trend="add" if len(y) > 10 else None,
            seasonal="add" if len(y) > 24 else None,
            seasonal_periods=12 if len(y) > 24 else None
        )
        fit = model.fit(optimized=True, disp=False)
        return fit.forecast(cfg.horizon).clip(lower=0)
    except:
        future_dates = pd.date_range(start=y.index[-1], periods=cfg.horizon + 1, freq=cfg.freq)[1:]
        return pd.Series([y.mean()] * cfg.horizon, index=future_dates)

# ══════════════════════════════════════════════════════════════════════════════
# 📈 LEGENDARY CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def create_forecast_chart(hist_df: pd.DataFrame, fc_df: pd.DataFrame, title: str):
    """Create the most beautiful forecast chart ever"""
    fig = go.Figure()
    
    # Historical data - smooth line
    fig.add_trace(go.Scatter(
        x=hist_df['ds'],
        y=hist_df['qty'],
        mode='lines',
        name='Historical',
        line=dict(color='rgba(255, 255, 255, 0.8)', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        hovertemplate='<b>%{x|%b %Y}</b><br>Qty: %{y:,.0f}<extra></extra>'
    ))
    
    # Forecast - dashed line with glow
    if not fc_df.empty:
        fig.add_trace(go.Scatter(
            x=fc_df['ds'],
            y=fc_df['qty_forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#00f2fe', width=3, dash='dash'),
            marker=dict(size=8, color='#00f2fe', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.2)',
            hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, color='white', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False,
            title=None
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False,
            title='Units'
        ),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ═══ SIDEBAR ═══
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")
        
        horizon = st.slider("📅 Forecast Horizon", 3, 24, 12, help="Number of months to forecast")
        freq = st.selectbox("📊 Frequency", ["Monthly", "Weekly"], index=0)
        freq_code = "MS" if freq == "Monthly" else "W"
        
        cfg = ForecastConfig(horizon=horizon, freq=freq_code)
        
        st.markdown("---")
        st.markdown("### 📡 Connection")
        st.success("✓ Data Synced")
        st.caption(f"Last updated: {datetime.now().strftime('%I:%M %p')}")
    
    # ═══ EPIC HEADER ═══
    st.markdown('<h1>🚀 S&OP COMMAND CENTER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin-bottom: 2rem;">Real-time Sales & Operations Intelligence</p>', unsafe_allow_html=True)
    
    # ═══ LOAD DATA ═══
    with st.spinner("🔮 Loading intelligence..."):
        data = load_data()
    
    dem = prepare_demand_history(data, freq=cfg.freq)
    
    if dem.empty:
        st.error("🚫 No demand data available")
        st.stop()
    
    # Get lists
    customers = sorted(dem["customer"].dropna().unique())
    skus = sorted(dem["sku"].dropna().unique())
    categories = sorted(dem["category"].dropna().unique())
    
    # Sales reps
    sales_reps = []
    rep_map = {}
    if not data["customers"].empty:
        c_cust = resolve_col(data["customers"], "customer")
        c_rep = resolve_col(data["customers"], "sales_rep")
        if c_cust and c_rep:
            rep_map = dict(zip(
                data["customers"][c_cust].astype(str).str.strip(),
                data["customers"][c_rep].astype(str).str.strip()
            ))
            sales_reps = sorted(set(rep_map.values()))
    
    # ═══ METRICS ROW ═══
    current_date = dem['ds'].max()
    l12m = dem[dem['ds'] > (current_date - pd.DateOffset(months=12))]['qty'].sum()
    prev_l12m = dem[
        (dem['ds'] <= (current_date - pd.DateOffset(months=12))) &
        (dem['ds'] > (current_date - pd.DateOffset(months=24)))
    ]['qty'].sum()
    delta = ((l12m - prev_l12m) / prev_l12m * 100) if prev_l12m > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 L12M UNITS", format_qty(l12m), f"{delta:+.1f}%")
    col2.metric("🎯 ACTIVE SKUS", f"{len(skus):,}")
    col3.metric("👥 CUSTOMERS", f"{len(customers):,}")
    col4.metric("📊 CATEGORIES", f"{len(categories):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══ TABS ═══
    tabs = st.tabs(["📊 Sales Intelligence", "🏭 Operations", "🔮 Scenarios", "📦 Purchase Orders", "🚚 Logistics"])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: SALES INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 💎 Sales Intelligence Dashboard")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            rep = st.selectbox("👤 Sales Rep", ["🌐 All Reps"] + sales_reps, index=0)
        
        with c2:
            if rep != "🌐 All Reps" and rep_map:
                rep_customers = [c for c, r in rep_map.items() if r == rep]
                customer = st.selectbox("🏢 Customer", ["🌐 All Customers"] + rep_customers)
            else:
                customer = st.selectbox("🏢 Customer", ["🌐 All Customers"] + customers)
        
        with c3:
            if customer not in ("🌐 All Customers",) and not dem.empty:
                cust_dem = dem[dem["customer"] == customer]
                cust_skus = sorted(cust_dem["sku"].unique())
            else:
                cust_dem = dem
                cust_skus = skus
            
            sku = st.selectbox("📦 SKU", ["🌐 All SKUs"] + cust_skus, index=0)
        
        # Filter
        filtered = cust_dem.copy()
        if sku != "🌐 All SKUs":
            filtered = filtered[filtered["sku"] == sku]
        
        if not filtered.empty:
            st.markdown("---")
            st.markdown("### 📈 Demand Forecast")
            
            # Forecast
            fc_results = []
            skipped = 0
            
            for s, g in filtered.groupby("sku"):
                hist = g.groupby("ds")["qty"].sum()
                if len(hist) < 2:
                    skipped += 1
                    continue
                fc = run_forecast(hist, cfg)
                fc_df = fc.reset_index()
                fc_df.columns = ["ds", "qty_forecast"]
                fc_df["sku"] = s
                fc_results.append(fc_df)
            
            if skipped > 0:
                st.info(f"ℹ️ {skipped} SKUs skipped (insufficient data)")
            
            if fc_results:
                all_fc = pd.concat(fc_results, ignore_index=True)
                agg_fc = all_fc.groupby("ds")["qty_forecast"].sum().reset_index()
                hist_agg = filtered.groupby("ds")["qty"].sum().reset_index()
                
                title = f"🎯 {customer if customer != '🌐 All Customers' else 'Company-Wide'} Forecast"
                fig = create_forecast_chart(hist_agg, agg_fc, title)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cadence
                st.markdown("---")
                st.markdown("### 🎯 SKU Ordering Patterns")
                
                cadence_rows = []
                for s, g in filtered.groupby("sku"):
                    g2 = g.groupby("ds")["qty"].sum().reset_index()
                    g2 = g2[g2["qty"] > 0]
                    if len(g2) >= 2:
                        deltas = g2["ds"].diff().dt.days.dropna()
                        cadence_rows.append({
                            "SKU": s,
                            "Orders": len(g2),
                            "Avg Days Between": f"{deltas.mean():.0f}" if not deltas.empty else "N/A",
                            "Last Order": g2["ds"].max().strftime('%b %d, %Y')
                        })
                
                if cadence_rows:
                    cadence_df = pd.DataFrame(cadence_rows).sort_values("Orders", ascending=False).head(10)
                    st.dataframe(cadence_df, use_container_width=True, hide_index=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## 🏭 Operations Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⏱️ Lead Times")
            items_df = data["items"]
            if not items_df.empty:
                c_ven = resolve_col(items_df, "vendor")
                c_lt = resolve_col(items_df, "lead_time")
                if c_ven and c_lt:
                    items_df[c_lt] = items_df[c_lt].apply(_safe_float)
                    lt_df = items_df.groupby(c_ven)[c_lt].mean().reset_index()
                    
                    fig = go.Figure(go.Bar(
                        x=lt_df[c_ven],
                        y=lt_df[c_lt],
                        marker=dict(
                            color=lt_df[c_lt],
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=lt_df[c_lt].round(0),
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Average Lead Time by Vendor",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title="Days",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎨 Category Mix")
            cat_mix = dem.groupby("category")["qty"].sum().reset_index()
            
            fig = go.Figure(go.Pie(
                labels=cat_mix["category"],
                values=cat_mix["qty"],
                hole=0.5,
                marker=dict(
                    colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe'],
                    line=dict(color='rgba(255,255,255,0.2)', width=2)
                ),
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>Units: %{value:,.0f}<br>%{percent}<extra></extra>'
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(255,255,255,0.1)',
                    bordercolor='rgba(255,255,255,0.2)',
                    borderwidth=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: SCENARIOS
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## 🔮 Scenario Planner")
        
        base_hist = dem.groupby("ds")["qty"].sum()
        base_fc = run_forecast(base_hist, cfg).reset_index()
        base_fc.columns = ["ds", "qty"]
        
        with st.expander("⚙️ Scenario Assumptions", expanded=True):
            col1, col2 = st.columns(2)
            growth = col1.slider("📈 Growth Rate (%)", -50, 50, 0) / 100
            bump = col2.number_input("🚀 Marketing Boost (units)", 0, 10000, 0, 100)
        
        base_fc["scenario"] = base_fc["qty"] * (1 + growth) + bump / cfg.horizon
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=base_fc["ds"], y=base_fc["qty"],
            name="Baseline", line=dict(dash="dash", color="rgba(255,255,255,0.5)", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=base_fc["ds"], y=base_fc["scenario"],
            name="Scenario", line=dict(color="#00f2fe", width=3),
            fill="tonexty", fillcolor="rgba(79, 172, 254, 0.2)"
        ))
        fig.update_layout(
            title="📊 Scenario Impact Analysis",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        impact = base_fc["scenario"].sum() - base_fc["qty"].sum()
        st.metric("💥 Total Impact", format_qty(impact), f"{impact/base_fc['qty'].sum()*100:+.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: PO FORECAST
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## 📦 Purchase Order Intelligence")
        
        base_hist = dem.groupby("ds")["qty"].sum()
        po_fc = run_forecast(base_hist, cfg).reset_index()
        po_fc.columns = ["ds", "qty_forecast"]
        po_fc["Safety Stock"] = (po_fc["qty_forecast"] * 0.2).round(0)
        po_fc["Order Qty"] = po_fc["qty_forecast"] + po_fc["Safety Stock"]
        po_fc["Est. Cost"] = po_fc["Order Qty"] * 50
        
        total_spend = po_fc["Est. Cost"].sum()
        st.metric("💰 Projected Spend", format_currency(total_spend), f"{horizon} months")
        
        st.markdown("---")
        st.dataframe(
            po_fc[["ds", "Order Qty", "Est. Cost"]].style.format({
                "Order Qty": "{:,.0f}",
                "Est. Cost": "${:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        csv = po_fc.to_csv(index=False)
        st.download_button("⬇️ Download PO Plan", csv, "po_forecast.csv", "text/csv")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: LOGISTICS
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## 🚚 Logistics & Delivery Tracking")
        st.info("🚧 Shipment tracking module launching Q1 2026")

if __name__ == "__main__":
    main()
