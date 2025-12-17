"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX - SALES REP ACTION CENTER (v5.0)
    
    Clear, No-Nonsense View for Sales Reps:
    1. Historical Performance (2024 & 2025)
    2. Current Quarter Status (Q4 2025)
    3. Pending Sales Orders (Backlog)
    4. Q4 2025 Pipeline (HubSpot)
    5. 2026 Outlook (Pipeline + Future SOs)
    6. 2026 Forecast by Customer & Product
    7. Client-Level Forecast Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calyx Sales Rep Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_YEAR = 2025
NEXT_YEAR = 2026
CURRENT_QUARTER = 4  # Q4 2025

# Clean CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .big-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    .big-metric-label { font-size: 0.8rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
    .big-metric-value { font-size: 2rem; font-weight: 700; margin-top: 5px; }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 0.75rem; color: #6b7280; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.5rem; color: #111827; font-weight: 700; margin-top: 4px; }
    .metric-sub { font-size: 0.7rem; color: #9ca3af; margin-top: 2px; }
    
    .section-header {
        background-color: #f8fafc;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 20px 0 10px 0;
        border-left: 4px solid #3b82f6;
    }
    .section-header h3 { margin: 0; color: #1e40af; font-size: 1.1rem; }
    
    h1, h2, h3 { color: #111827; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

SHEET_SO_INV = "SO & invoice Data merged"
SHEET_DEALS = "Deals"

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_headers(headers):
    """Ensures all headers are unique."""
    deduped = []
    counts = {}
    for h in headers:
        h_clean = str(h).strip()
        if h_clean in counts:
            counts[h_clean] += 1
            deduped.append(f"{h_clean}_{counts[h_clean]}")
        else:
            counts[h_clean] = 0
            deduped.append(h_clean)
    return deduped

def safe_get_col(df, candidates, default_val='Unknown'):
    for col in candidates:
        matches = [c for c in df.columns if c.strip().lower() == col.strip().lower()]
        if matches: return df[matches[0]]
    return pd.Series([default_val] * len(df), index=df.index)

def clean_money(s):
    return pd.to_numeric(s.astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)

def format_currency(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"${val/1_000:.0f}K"
    else:
        return f"${val:,.0f}"

def render_metric_card(label, value, subtitle=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_big_metric(label, value):
    st.markdown(f"""
    <div class="big-metric">
        <div class="big-metric-label">{label}</div>
        <div class="big-metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(title):
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_and_prep_data():
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        creds_dict = None
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        elif "service_account" in st.secrets:
            creds_dict = dict(st.secrets["service_account"])
        else:
            return None, None

        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        
        sheet_id = st.secrets.get("SPREADSHEET_ID") or st.secrets["gsheets"]["spreadsheet_id"]
        sh = client.open_by_key(sheet_id)

        # Load SO/Invoice Data
        try:
            ws = sh.worksheet(SHEET_SO_INV)
            rows = ws.get_all_values()
            df_main = pd.DataFrame(rows[1:], columns=deduplicate_headers(rows[0])) if len(rows) > 1 else pd.DataFrame()
        except:
            df_main = pd.DataFrame()

        # Load Deals
        try:
            ws_d = sh.worksheet(SHEET_DEALS)
            rows_d = ws_d.get_all_values()
            df_deals = pd.DataFrame(rows_d[2:], columns=deduplicate_headers(rows_d[1])) if len(rows_d) > 2 else pd.DataFrame()
        except:
            df_deals = pd.DataFrame()

        # ── PREP MAIN DATA ──
        if not df_main.empty:
            df_main['Rep'] = safe_get_col(df_main, ['Inv - Rep Master', 'SO - Rep Master']).fillna('Unassigned').astype(str).str.strip()
            df_main['Customer'] = safe_get_col(df_main, ['Inv - Correct Customer', 'SO - Customer Companyname']).fillna('Unknown').astype(str).str.strip()
            df_main['Item'] = safe_get_col(df_main, ['SO - Item', 'Item']).astype(str).str.strip()
            df_main['Product Type'] = safe_get_col(df_main, ['SO - Calyx || Product Type', 'Product Type']).astype(str).str.strip()
            df_main['SO_Num'] = safe_get_col(df_main, ['SO - Document Number', 'Document Number']).astype(str).str.strip()
            
            df_main['Inv_Date'] = pd.to_datetime(safe_get_col(df_main, ['Inv - Date']), errors='coerce')
            df_main['SO_Date'] = pd.to_datetime(safe_get_col(df_main, ['SO - Date Created', 'Date Created']), errors='coerce')
            df_main['Fulfill_Date'] = pd.to_datetime(safe_get_col(df_main, ['SO - Pending Fulfillment Date']), errors='coerce')
            
            df_main['Amount'] = clean_money(safe_get_col(df_main, ['Inv - Amount', 'Amount']))
            
            # Type: Invoiced vs Pending SO
            df_main['Type'] = np.where(df_main['Inv_Date'].notnull(), 'Invoiced', 'Pending SO')
            
            df_main['Effective_Date'] = np.where(
                df_main['Type'] == 'Invoiced', 
                df_main['Inv_Date'], 
                df_main['Fulfill_Date'].combine_first(df_main['SO_Date'])
            )
            df_main['Effective_Date'] = pd.to_datetime(df_main['Effective_Date'], errors='coerce')
            
            df_main = df_main[df_main['Amount'] > 0]

        # ── PREP DEALS ──
        if not df_deals.empty:
            inc = safe_get_col(df_deals, ['Include?', 'Include'])
            df_deals = df_deals[inc.astype(str).str.upper().isin(['TRUE', 'YES', '1'])].copy()
            
            first = safe_get_col(df_deals, ['Deal Owner First Name']).astype(str)
            last = safe_get_col(df_deals, ['Deal Owner Last Name']).astype(str)
            df_deals['Rep'] = (first + " " + last).str.strip().str.replace('None None', 'Unassigned')
            
            df_deals['Amount'] = clean_money(safe_get_col(df_deals, ['Amount']))
            df_deals['Close Date'] = pd.to_datetime(safe_get_col(df_deals, ['Close Date']), errors='coerce')
            df_deals['Stage'] = safe_get_col(df_deals, ['Deal Stage', 'Stage'])
            df_deals['Deal Name'] = safe_get_col(df_deals, ['Deal Name'])
            
            # Match customers
            unique_cust = df_main['Customer'].unique() if not df_main.empty else []
            def match(name):
                name_up = str(name).upper()
                for c in sorted(unique_cust, key=len, reverse=True):
                    if len(c) > 3 and c.upper() in name_up: return c
                return "Unassigned"
            df_deals['Matched_Customer'] = df_deals['Deal Name'].apply(match)

        return df_main, df_deals
        
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# CLIENT FORECAST GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_client_forecast(df_hist, df_pipe, customer, year_target=2026):
    """
    Generate quarterly forecast for a specific customer based on:
    - Historical quarterly patterns (2024/2025)
    - Current pipeline deals
    """
    results = {
        'customer': customer,
        'historical_2024': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0, 'Total': 0},
        'historical_2025': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0, 'Total': 0},
        'pipeline_2026': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0, 'Total': 0},
        'forecast_2026': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0, 'Total': 0},
        'method': 'No History'
    }
    
    cust_hist = df_hist[(df_hist['Customer'] == customer) & (df_hist['Type'] == 'Invoiced')]
    
    # 2024 Quarterly
    for q in [1, 2, 3, 4]:
        val = cust_hist[(cust_hist['Inv_Date'].dt.year == 2024) & (cust_hist['Inv_Date'].dt.quarter == q)]['Amount'].sum()
        results['historical_2024'][f'Q{q}'] = val
    results['historical_2024']['Total'] = sum(results['historical_2024'][f'Q{q}'] for q in [1,2,3,4])
    
    # 2025 Quarterly
    for q in [1, 2, 3, 4]:
        val = cust_hist[(cust_hist['Inv_Date'].dt.year == 2025) & (cust_hist['Inv_Date'].dt.quarter == q)]['Amount'].sum()
        results['historical_2025'][f'Q{q}'] = val
    results['historical_2025']['Total'] = sum(results['historical_2025'][f'Q{q}'] for q in [1,2,3,4])
    
    # Pipeline for 2026
    cust_pipe = df_pipe[(df_pipe['Matched_Customer'] == customer) & (df_pipe['Close Date'].dt.year == year_target)]
    for q in [1, 2, 3, 4]:
        val = cust_pipe[cust_pipe['Close Date'].dt.quarter == q]['Amount'].sum()
        results['pipeline_2026'][f'Q{q}'] = val
    results['pipeline_2026']['Total'] = cust_pipe['Amount'].sum()
    
    # Generate Forecast
    total_2024 = results['historical_2024']['Total']
    total_2025 = results['historical_2025']['Total']
    
    if total_2025 > 0 and total_2024 > 0:
        # Use YoY growth rate applied to quarterly pattern
        growth_rate = (total_2025 - total_2024) / total_2024 if total_2024 > 0 else 0
        results['method'] = f"YoY Pattern (+{growth_rate:.0%} growth)" if growth_rate >= 0 else f"YoY Pattern ({growth_rate:.0%})"
        
        for q in [1, 2, 3, 4]:
            base = results['historical_2025'][f'Q{q}']
            pipeline = results['pipeline_2026'][f'Q{q}']
            # Forecast = 2025 quarterly * (1 + growth) + any extra pipeline
            projected = base * (1 + max(growth_rate, 0))
            results['forecast_2026'][f'Q{q}'] = max(projected, pipeline)
            
    elif total_2025 > 0:
        # No 2024 data, use flat 2025
        results['method'] = "Flat from 2025"
        for q in [1, 2, 3, 4]:
            base = results['historical_2025'][f'Q{q}']
            pipeline = results['pipeline_2026'][f'Q{q}']
            results['forecast_2026'][f'Q{q}'] = max(base, pipeline)
            
    elif results['pipeline_2026']['Total'] > 0:
        # New customer with pipeline only
        results['method'] = "Pipeline Only (New Customer)"
        for q in [1, 2, 3, 4]:
            results['forecast_2026'][f'Q{q}'] = results['pipeline_2026'][f'Q{q}']
    
    results['forecast_2026']['Total'] = sum(results['forecast_2026'][f'Q{q}'] for q in [1,2,3,4])
    
    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    
    # Load Data
    with st.spinner("Loading Sales Data..."):
        df_main, df_deals = load_and_prep_data()

    if df_main is None or df_main.empty:
        st.error("No data found. Please check your Google Sheets connection.")
        st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR FILTERS
    # ══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.header("🎯 My View")
        
        # Rep Filter
        all_reps = sorted(list(set(df_main['Rep'].unique()) | set(df_deals['Rep'].unique())))
        sel_rep = st.selectbox("Select Your Name", ["All Reps"] + all_reps)
        
        if sel_rep != "All Reps":
            df_main = df_main[df_main['Rep'] == sel_rep]
            df_deals = df_deals[df_deals['Rep'] == sel_rep]
        
        st.markdown("---")
        
        # Customer Filter
        cust_list = sorted(df_main['Customer'].unique())
        sel_cust = st.selectbox("Filter by Customer", ["All Customers"] + cust_list)
        
        if sel_cust != "All Customers":
            df_main = df_main[df_main['Customer'] == sel_cust]
            df_deals = df_deals[df_deals['Matched_Customer'] == sel_cust]
        
        st.markdown("---")
        st.caption("📊 Data refreshes hourly")
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    
    if sel_rep != "All Reps":
        st.title(f"📊 {sel_rep}'s Sales Dashboard")
    else:
        st.title("📊 Calyx Sales Dashboard")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEVEL KPIs
    # ══════════════════════════════════════════════════════════════════════════
    
    # Calculate key metrics
    inv_2024 = df_main[(df_main['Type'] == 'Invoiced') & (df_main['Inv_Date'].dt.year == 2024)]['Amount'].sum()
    inv_2025 = df_main[(df_main['Type'] == 'Invoiced') & (df_main['Inv_Date'].dt.year == 2025)]['Amount'].sum()
    inv_q4_2025 = df_main[(df_main['Type'] == 'Invoiced') & (df_main['Inv_Date'].dt.year == 2025) & (df_main['Inv_Date'].dt.quarter == 4)]['Amount'].sum()
    
    pending_so = df_main[df_main['Type'] == 'Pending SO']['Amount'].sum()
    
    pipe_q4_2025 = df_deals[(df_deals['Close Date'].dt.year == 2025) & (df_deals['Close Date'].dt.quarter == 4)]['Amount'].sum()
    pipe_2026 = df_deals[df_deals['Close Date'].dt.year == 2026]['Amount'].sum()
    
    # Display top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        render_metric_card("2024 Total", format_currency(inv_2024), "Full Year Invoiced")
    with col2:
        render_metric_card("2025 YTD", format_currency(inv_2025), "Invoiced Revenue")
    with col3:
        render_metric_card("Q4 2025", format_currency(inv_q4_2025), "This Quarter Invoiced")
    with col4:
        render_metric_card("Pending Orders", format_currency(pending_so), "Active Sales Orders")
    with col5:
        render_metric_card("2026 Pipeline", format_currency(pipe_2026), "HubSpot Deals")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TABBED SECTIONS
    # ══════════════════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Historical Performance",
        "🎯 Current Quarter (Q4)",
        "📦 Pending Orders & Pipeline",
        "🔮 2026 Outlook",
        "🧮 Forecast Generator"
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: HISTORICAL PERFORMANCE (2024 & 2025)
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Historical Revenue Performance")
        
        col_24, col_25 = st.columns(2)
        
        with col_24:
            section_header("📅 2024 Performance")
            hist_2024 = df_main[(df_main['Type'] == 'Invoiced') & (df_main['Inv_Date'].dt.year == 2024)]
            
            if not hist_2024.empty:
                # Quarterly breakdown
                q_data = []
                for q in [1, 2, 3, 4]:
                    q_val = hist_2024[hist_2024['Inv_Date'].dt.quarter == q]['Amount'].sum()
                    q_data.append({'Quarter': f'Q{q} 2024', 'Revenue': q_val})
                q_df = pd.DataFrame(q_data)
                st.dataframe(q_df, use_container_width=True, hide_index=True,
                    column_config={"Revenue": st.column_config.NumberColumn(format="$%,.0f")})
                
                # By Customer
                st.markdown("**By Customer:**")
                cust_24 = hist_2024.groupby('Customer')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
                st.dataframe(cust_24, use_container_width=True, hide_index=True,
                    column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})
            else:
                st.info("No 2024 invoiced data.")
        
        with col_25:
            section_header("📅 2025 Performance (YTD)")
            hist_2025 = df_main[(df_main['Type'] == 'Invoiced') & (df_main['Inv_Date'].dt.year == 2025)]
            
            if not hist_2025.empty:
                # Quarterly breakdown
                q_data = []
                for q in [1, 2, 3, 4]:
                    q_val = hist_2025[hist_2025['Inv_Date'].dt.quarter == q]['Amount'].sum()
                    q_data.append({'Quarter': f'Q{q} 2025', 'Revenue': q_val})
                q_df = pd.DataFrame(q_data)
                st.dataframe(q_df, use_container_width=True, hide_index=True,
                    column_config={"Revenue": st.column_config.NumberColumn(format="$%,.0f")})
                
                # By Customer
                st.markdown("**By Customer:**")
                cust_25 = hist_2025.groupby('Customer')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
                st.dataframe(cust_25, use_container_width=True, hide_index=True,
                    column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})
            else:
                st.info("No 2025 invoiced data.")

        # YoY Comparison
        if inv_2024 > 0 and inv_2025 > 0:
            st.markdown("---")
            section_header("📊 Year-over-Year Comparison")
            yoy_growth = ((inv_2025 - inv_2024) / inv_2024) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("2024 Total", f"${inv_2024:,.0f}")
            m2.metric("2025 YTD", f"${inv_2025:,.0f}")
            m3.metric("YoY Change", f"{yoy_growth:+.1f}%", delta=f"${inv_2025 - inv_2024:,.0f}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: CURRENT QUARTER (Q4 2025)
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Q4 2025 - Current Quarter Performance")
        
        q4_invoiced = df_main[
            (df_main['Type'] == 'Invoiced') & 
            (df_main['Inv_Date'].dt.year == 2025) & 
            (df_main['Inv_Date'].dt.quarter == 4)
        ]
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            render_big_metric("Q4 2025 Invoiced", format_currency(inv_q4_2025))
            
            # Compare to Q4 2024
            q4_2024 = df_main[
                (df_main['Type'] == 'Invoiced') & 
                (df_main['Inv_Date'].dt.year == 2024) & 
                (df_main['Inv_Date'].dt.quarter == 4)
            ]['Amount'].sum()
            
            if q4_2024 > 0:
                q4_yoy = ((inv_q4_2025 - q4_2024) / q4_2024) * 100
                st.metric("vs Q4 2024", f"${q4_2024:,.0f}", delta=f"{q4_yoy:+.1f}%")
        
        with col_b:
            section_header("Q4 2025 Invoiced by Customer")
            if not q4_invoiced.empty:
                q4_by_cust = q4_invoiced.groupby('Customer')['Amount'].sum().sort_values(ascending=False).reset_index()
                st.dataframe(q4_by_cust, use_container_width=True, hide_index=True,
                    column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})
            else:
                st.info("No Q4 2025 invoiced revenue yet.")
        
        st.markdown("---")
        section_header("Q4 2025 Invoiced by Product Type")
        if not q4_invoiced.empty:
            q4_by_prod = q4_invoiced.groupby('Product Type')['Amount'].sum().sort_values(ascending=False).reset_index()
            st.dataframe(q4_by_prod, use_container_width=True, hide_index=True,
                column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: PENDING ORDERS & PIPELINE
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Active Orders & Pipeline")
        
        col_so, col_pipe = st.columns(2)
        
        with col_so:
            section_header("📦 Pending Sales Orders (Backlog)")
            pending = df_main[df_main['Type'] == 'Pending SO'].copy()
            
            st.metric("Total Pending", f"${pending_so:,.0f}")
            
            if not pending.empty:
                pending_view = pending[['SO_Date', 'Customer', 'SO_Num', 'Item', 'Amount']].sort_values('Amount', ascending=False)
                st.dataframe(
                    pending_view,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "SO_Date": st.column_config.DateColumn("Order Date", format="MMM DD"),
                        "Amount": st.column_config.NumberColumn(format="$%,.0f")
                    }
                )
                
                # Summary by customer
                st.markdown("**Pending by Customer:**")
                pending_cust = pending.groupby('Customer')['Amount'].sum().sort_values(ascending=False).reset_index()
                st.dataframe(pending_cust, use_container_width=True, hide_index=True,
                    column_config={"Amount": st.column_config.NumberColumn(format="$%,.0f")})
            else:
                st.success("✅ No pending orders - all caught up!")
        
        with col_pipe:
            section_header("🎯 Q4 2025 Pipeline (HubSpot)")
            pipe_q4 = df_deals[
                (df_deals['Close Date'].dt.year == 2025) & 
                (df_deals['Close Date'].dt.quarter == 4)
            ]
            
            st.metric("Q4 2025 Pipeline", f"${pipe_q4_2025:,.0f}")
            
            if not pipe_q4.empty:
                st.dataframe(
                    pipe_q4[['Close Date', 'Deal Name', 'Matched_Customer', 'Stage', 'Amount']].sort_values('Close Date'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Close Date": st.column_config.DateColumn(format="MMM DD"),
                        "Amount": st.column_config.NumberColumn(format="$%,.0f")
                    }
                )
            else:
                st.info("No Q4 2025 pipeline deals.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: 2026 OUTLOOK
    # ──────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("2026 Outlook")
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        
        # Pipeline deals for 2026
        pipe_2026_deals = df_deals[df_deals['Close Date'].dt.year == 2026]
        
        # Pending SOs with 2026 fulfillment dates
        pending_2026 = df_main[
            (df_main['Type'] == 'Pending SO') & 
            (df_main['Fulfill_Date'].dt.year == 2026)
        ]
        pending_2026_amt = pending_2026['Amount'].sum()
        
        with m1:
            render_metric_card("2026 Pipeline", format_currency(pipe_2026), "HubSpot Deals")
        with m2:
            render_metric_card("2026 Pending SOs", format_currency(pending_2026_amt), "Fulfillment in 2026")
        with m3:
            total_2026_view = pipe_2026 + pending_2026_amt
            render_metric_card("Total 2026 Visibility", format_currency(total_2026_view), "Pipeline + Pending")
        
        st.markdown("---")
        
        col_pipe26, col_so26 = st.columns(2)
        
        with col_pipe26:
            section_header("🔮 2026 Pipeline by Quarter")
            if not pipe_2026_deals.empty:
                q_breakdown = []
                for q in [1, 2, 3, 4]:
                    q_val = pipe_2026_deals[pipe_2026_deals['Close Date'].dt.quarter == q]['Amount'].sum()
                    q_breakdown.append({'Quarter': f'Q{q} 2026', 'Pipeline': q_val})
                st.dataframe(pd.DataFrame(q_breakdown), use_container_width=True, hide_index=True,
                    column_config={"Pipeline": st.column_config.NumberColumn(format="$%,.0f")})
                
                st.markdown("**Pipeline Deals:**")
                st.dataframe(
                    pipe_2026_deals[['Close Date', 'Deal Name', 'Matched_Customer', 'Stage', 'Amount']].sort_values('Close Date'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Close Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
                        "Amount": st.column_config.NumberColumn(format="$%,.0f")
                    }
                )
            else:
                st.info("No 2026 pipeline deals yet.")
        
        with col_so26:
            section_header("📦 2026 Pending Sales Orders")
            if not pending_2026.empty:
                st.dataframe(
                    pending_2026[['Fulfill_Date', 'Customer', 'SO_Num', 'Item', 'Amount']].sort_values('Fulfill_Date'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Fulfill_Date": st.column_config.DateColumn("Fulfill Date", format="MMM DD, YYYY"),
                        "Amount": st.column_config.NumberColumn(format="$%,.0f")
                    }
                )
            else:
                st.info("No Sales Orders with 2026 fulfillment dates.")
        
        # 2026 by Customer & Product
        st.markdown("---")
        section_header("📊 2026 Pipeline by Customer")
        if not pipe_2026_deals.empty:
            pipe_by_cust = pipe_2026_deals.groupby('Matched_Customer')['Amount'].sum().sort_values(ascending=False).reset_index()
            pipe_by_cust.columns = ['Customer', '2026 Pipeline']
            st.dataframe(pipe_by_cust, use_container_width=True, hide_index=True,
                column_config={"2026 Pipeline": st.column_config.NumberColumn(format="$%,.0f")})

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5: FORECAST GENERATOR
    # ──────────────────────────────────────────────────────────────────────────
    with tab5:
        st.subheader("🧮 2026 Forecast Generator")
        st.markdown("Generate quarterly forecasts for each customer based on historical patterns and current pipeline.")
        
        # Reload unfiltered data for forecast generation
        df_main_full, df_deals_full = load_and_prep_data()
        
        # Apply rep filter if selected
        if sel_rep != "All Reps":
            df_main_full = df_main_full[df_main_full['Rep'] == sel_rep]
            df_deals_full = df_deals_full[df_deals_full['Rep'] == sel_rep]
        
        # Get unique customers
        all_customers = sorted(df_main_full['Customer'].unique())
        
        # Option: Generate for all or specific customer
        forecast_mode = st.radio("Forecast Mode:", ["Single Customer", "All Customers"], horizontal=True)
        
        if forecast_mode == "Single Customer":
            selected_customer = st.selectbox("Select Customer:", all_customers)
            
            if st.button("Generate Forecast", type="primary"):
                forecast = generate_client_forecast(df_main_full, df_deals_full, selected_customer)
                
                st.markdown(f"### Forecast for: {selected_customer}")
                st.caption(f"Method: {forecast['method']}")
                
                # Build comparison table
                forecast_table = pd.DataFrame({
                    'Period': ['Q1', 'Q2', 'Q3', 'Q4', 'TOTAL'],
                    '2024 Actual': [
                        forecast['historical_2024']['Q1'],
                        forecast['historical_2024']['Q2'],
                        forecast['historical_2024']['Q3'],
                        forecast['historical_2024']['Q4'],
                        forecast['historical_2024']['Total']
                    ],
                    '2025 Actual': [
                        forecast['historical_2025']['Q1'],
                        forecast['historical_2025']['Q2'],
                        forecast['historical_2025']['Q3'],
                        forecast['historical_2025']['Q4'],
                        forecast['historical_2025']['Total']
                    ],
                    '2026 Pipeline': [
                        forecast['pipeline_2026']['Q1'],
                        forecast['pipeline_2026']['Q2'],
                        forecast['pipeline_2026']['Q3'],
                        forecast['pipeline_2026']['Q4'],
                        forecast['pipeline_2026']['Total']
                    ],
                    '2026 FORECAST': [
                        forecast['forecast_2026']['Q1'],
                        forecast['forecast_2026']['Q2'],
                        forecast['forecast_2026']['Q3'],
                        forecast['forecast_2026']['Q4'],
                        forecast['forecast_2026']['Total']
                    ]
                })
                
                st.dataframe(
                    forecast_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "2024 Actual": st.column_config.NumberColumn(format="$%,.0f"),
                        "2025 Actual": st.column_config.NumberColumn(format="$%,.0f"),
                        "2026 Pipeline": st.column_config.NumberColumn(format="$%,.0f"),
                        "2026 FORECAST": st.column_config.NumberColumn(format="$%,.0f")
                    }
                )
        
        else:  # All Customers
            if st.button("Generate All Forecasts", type="primary"):
                all_forecasts = []
                
                progress = st.progress(0)
                for i, cust in enumerate(all_customers):
                    fc = generate_client_forecast(df_main_full, df_deals_full, cust)
                    
                    # Only include if there's any activity
                    if (fc['historical_2024']['Total'] + fc['historical_2025']['Total'] + 
                        fc['pipeline_2026']['Total'] + fc['forecast_2026']['Total']) > 0:
                        all_forecasts.append({
                            'Customer': cust,
                            '2024 Total': fc['historical_2024']['Total'],
                            '2025 Total': fc['historical_2025']['Total'],
                            'Q1 2026': fc['forecast_2026']['Q1'],
                            'Q2 2026': fc['forecast_2026']['Q2'],
                            'Q3 2026': fc['forecast_2026']['Q3'],
                            'Q4 2026': fc['forecast_2026']['Q4'],
                            '2026 FORECAST': fc['forecast_2026']['Total'],
                            'Method': fc['method']
                        })
                    progress.progress((i + 1) / len(all_customers))
                
                progress.empty()
                
                if all_forecasts:
                    fc_df = pd.DataFrame(all_forecasts)
                    fc_df = fc_df.sort_values('2026 FORECAST', ascending=False)
                    
                    # Summary metrics
                    total_2026_fc = fc_df['2026 FORECAST'].sum()
                    total_2025 = fc_df['2025 Total'].sum()
                    
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Total 2025", f"${total_2025:,.0f}")
                    mc2.metric("Total 2026 Forecast", f"${total_2026_fc:,.0f}")
                    if total_2025 > 0:
                        growth = ((total_2026_fc - total_2025) / total_2025) * 100
                        mc3.metric("Projected Growth", f"{growth:+.1f}%")
                    
                    st.markdown("---")
                    st.dataframe(
                        fc_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "2024 Total": st.column_config.NumberColumn(format="$%,.0f"),
                            "2025 Total": st.column_config.NumberColumn(format="$%,.0f"),
                            "Q1 2026": st.column_config.NumberColumn(format="$%,.0f"),
                            "Q2 2026": st.column_config.NumberColumn(format="$%,.0f"),
                            "Q3 2026": st.column_config.NumberColumn(format="$%,.0f"),
                            "Q4 2026": st.column_config.NumberColumn(format="$%,.0f"),
                            "2026 FORECAST": st.column_config.NumberColumn(format="$%,.0f")
                        }
                    )
                    
                    # Download button
                    csv = fc_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Forecast CSV",
                        csv,
                        "2026_forecast.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No customer data to generate forecasts.")

        # Product-level forecast section
        st.markdown("---")
        section_header("📦 2026 Forecast by Product Type")
        
        if st.button("Generate Product Forecasts"):
            # Get 2025 by product
            prod_2025 = df_main_full[
                (df_main_full['Type'] == 'Invoiced') & 
                (df_main_full['Inv_Date'].dt.year == 2025)
            ].groupby('Product Type')['Amount'].sum()
            
            # Get 2024 by product
            prod_2024 = df_main_full[
                (df_main_full['Type'] == 'Invoiced') & 
                (df_main_full['Inv_Date'].dt.year == 2024)
            ].groupby('Product Type')['Amount'].sum()
            
            prod_forecast = []
            for prod in prod_2025.index:
                val_25 = prod_2025.get(prod, 0)
                val_24 = prod_2024.get(prod, 0)
                
                if val_24 > 0:
                    growth = (val_25 - val_24) / val_24
                    fc_26 = val_25 * (1 + max(growth, 0))
                else:
                    fc_26 = val_25
                
                prod_forecast.append({
                    'Product Type': prod,
                    '2024': val_24,
                    '2025': val_25,
                    '2026 Forecast': fc_26,
                    'YoY Growth': f"{((val_25-val_24)/val_24)*100:+.1f}%" if val_24 > 0 else "N/A"
                })
            
            prod_df = pd.DataFrame(prod_forecast).sort_values('2026 Forecast', ascending=False)
            st.dataframe(
                prod_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "2024": st.column_config.NumberColumn(format="$%,.0f"),
                    "2025": st.column_config.NumberColumn(format="$%,.0f"),
                    "2026 Forecast": st.column_config.NumberColumn(format="$%,.0f")
                }
            )

if __name__ == "__main__":
    main()
