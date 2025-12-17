"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX - SALES ACTION CENTER (v4.1)
    Focus:
    1. 2025 Performance (Invoiced vs Pending)
    2. 2026 Forecasting (Customer & SKU Level)
    3. Robust Error Handling (Header Deduplication)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calyx Sales Rep View",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hardcoded timeline based on your prompt
CURRENT_YEAR = 2025
NEXT_YEAR = 2026

# Custom CSS for "Clean Corporate" look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 0.85rem; color: #6b7280; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; color: #111827; font-weight: 700; margin-top: 4px; }
    
    h1, h2, h3 { color: #111827; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

SHEET_SO_INV = "SO & invoice Data merged"
SHEET_DEALS = "Deals"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREP (ROBUST)
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

@st.cache_data(ttl=3600)
def load_and_prep_data():
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        # Connect
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

        # 1. LOAD SO/INV DATA
        try:
            ws = sh.worksheet(SHEET_SO_INV)
            rows = ws.get_all_values()
            df_main = pd.DataFrame(rows[1:], columns=deduplicate_headers(rows[0])) if len(rows) > 1 else pd.DataFrame()
        except:
            df_main = pd.DataFrame()

        # 2. LOAD DEALS
        try:
            ws_d = sh.worksheet(SHEET_DEALS)
            rows_d = ws_d.get_all_values()
            df_deals = pd.DataFrame(rows_d[2:], columns=deduplicate_headers(rows_d[1])) if len(rows_d) > 2 else pd.DataFrame()
        except:
            df_deals = pd.DataFrame()

        # --- PREP MAIN DATA ---
        if not df_main.empty:
            # Columns
            df_main['Rep'] = safe_get_col(df_main, ['Inv - Rep Master', 'SO - Rep Master']).fillna('Unassigned').astype(str).str.strip()
            df_main['Customer'] = safe_get_col(df_main, ['Inv - Correct Customer', 'SO - Customer Companyname']).fillna('Unknown').astype(str).str.strip()
            df_main['Item'] = safe_get_col(df_main, ['SO - Item', 'Item']).astype(str).str.strip()
            df_main['Product Type'] = safe_get_col(df_main, ['SO - Calyx || Product Type', 'Product Type']).astype(str).str.strip()
            df_main['SO_Num'] = safe_get_col(df_main, ['SO - Document Number', 'Document Number']).astype(str).str.strip()
            
            # Dates
            df_main['Inv_Date'] = pd.to_datetime(safe_get_col(df_main, ['Inv - Date']), errors='coerce')
            df_main['SO_Date'] = pd.to_datetime(safe_get_col(df_main, ['SO - Date Created', 'Date Created']), errors='coerce')
            
            # Use Pending Fulfillment Date for future SOs
            df_main['Fulfill_Date'] = pd.to_datetime(safe_get_col(df_main, ['SO - Pending Fulfillment Date']), errors='coerce')
            
            # Amounts
            def clean_money(s): return pd.to_numeric(s.astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            df_main['Amount'] = clean_money(safe_get_col(df_main, ['Inv - Amount', 'Amount']))
            
            # Logic: Split Invoiced vs Pending
            df_main['Type'] = np.where(df_main['Inv_Date'].notnull(), 'Invoiced', 'Pending SO')
            
            df_main['Effective_Date'] = np.where(
                df_main['Type'] == 'Invoiced', 
                df_main['Inv_Date'], 
                df_main['Fulfill_Date'].combine_first(df_main['SO_Date'])
            )
            
            df_main = df_main[df_main['Amount'] > 0]

        # --- PREP DEALS ---
        if not df_deals.empty:
            # Filter Include?
            inc = safe_get_col(df_deals, ['Include?', 'Include'])
            df_deals = df_deals[inc.astype(str).str.upper().isin(['TRUE', 'YES', '1'])].copy()
            
            # Rep & Amt
            first = safe_get_col(df_deals, ['Deal Owner First Name']).astype(str)
            last = safe_get_col(df_deals, ['Deal Owner Last Name']).astype(str)
            df_deals['Rep'] = (first + " " + last).str.strip().str.replace('None None', 'Unassigned')
            
            df_deals['Amount'] = clean_money(safe_get_col(df_deals, ['Amount']))
            df_deals['Close Date'] = pd.to_datetime(safe_get_col(df_deals, ['Close Date']), errors='coerce')
            df_deals['Stage'] = safe_get_col(df_deals, ['Deal Stage', 'Stage'])
            df_deals['Deal Name'] = safe_get_col(df_deals, ['Deal Name'])
            
            # Customer Matching
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
# FORECAST GENERATOR LOGIC (Client + SKU Level)
# ══════════════════════════════════════════════════════════════════════════════

def generate_detailed_forecast(df_hist, df_pipe, granularity='Customer', year_target=2026):
    """
    Builds a forecast table.
    Granularity can be 'Customer' or 'Item' (SKU).
    """
    forecast_rows = []
    
    # Identify entities based on granularity
    if granularity == 'Customer':
        entities = set(df_hist['Customer'].unique()) | set(df_pipe['Matched_Customer'].unique())
        key_col = 'Customer'
        pipe_col = 'Matched_Customer'
    else: # Item / Product Level
        # Note: Pipeline deals don't always have Items, so we may show 'General Pipeline' for unmatched deals
        entities = set(df_hist['Item'].unique())
        key_col = 'Item'
        # Pipeline matching to SKU is hard without direct link. 
        # We will group by Item for history, and add a separate row for 'Pipeline (Allocated)' if we wanted to get fancy.
        # For simplicity in this view, we'll focus on Historical Run Rate for SKU, 
        # and assume Deals are separate unless we build that complex allocator back.
        pipe_col = None 
        
    entities.discard('Unassigned')
    entities.discard('Unknown')
    
    for entity in sorted(entities):
        # 1. Historicals (2025)
        hist_rows = df_hist[df_hist[key_col] == entity]
        rev_2025 = hist_rows[hist_rows['Effective_Date'].dt.year == 2025]['Amount'].sum()
        
        # 2. Pipeline (2026)
        # Only applicable if we can link pipeline to the entity
        pipe_2026_val = 0
        if granularity == 'Customer' and pipe_col:
            pipe_rows = df_pipe[df_pipe[pipe_col] == entity]
            pipe_2026_val = pipe_rows[pipe_rows['Close Date'].dt.year == year_target]['Amount'].sum()
            
            # Quarterly pipe breakdown
            p_q1 = pipe_rows[(pipe_rows['Close Date'].dt.year == year_target) & (pipe_rows['Close Date'].dt.quarter == 1)]['Amount'].sum()
            p_q2 = pipe_rows[(pipe_rows['Close Date'].dt.year == year_target) & (pipe_rows['Close Date'].dt.quarter == 2)]['Amount'].sum()
            p_q3 = pipe_rows[(pipe_rows['Close Date'].dt.year == year_target) & (pipe_rows['Close Date'].dt.quarter == 3)]['Amount'].sum()
            p_q4 = pipe_rows[(pipe_rows['Close Date'].dt.year == year_target) & (pipe_rows['Close Date'].dt.quarter == 4)]['Amount'].sum()
        else:
            # SKU level pipeline is unknown, assume flat 0 or purely historical run rate
            p_q1, p_q2, p_q3, p_q4 = 0, 0, 0, 0

        # 3. Baseline Logic (Flat Run Rate / 4)
        # This answers: "If they do exactly what they did last year"
        baseline_q = rev_2025 / 4
        
        q1 = baseline_q + p_q1
        q2 = baseline_q + p_q2
        q3 = baseline_q + p_q3
        q4 = baseline_q + p_q4
        
        if (rev_2025 + q1 + q2 + q3 + q4) > 0: # Only show active items/customers
            row = {
                key_col: entity,
                "2025 Actuals": rev_2025,
                f"Q1 {year_target}": q1,
                f"Q2 {year_target}": q2,
                f"Q3 {year_target}": q3,
                f"Q4 {year_target}": q4,
                f"Total {year_target}": q1+q2+q3+q4
            }
            forecast_rows.append(row)
        
    return pd.DataFrame(forecast_rows)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    
    # LOAD
    with st.spinner("Loading Sales Data..."):
        df_main, df_deals = load_and_prep_data()

    if df_main.empty:
        st.error("No data found.")
        st.stop()

    # SIDEBAR FILTERS
    with st.sidebar:
        st.header("👤 Sales Filters")
        
        # 1. Rep
        all_reps = sorted(list(set(df_main['Rep'].unique()) | set(df_deals['Rep'].unique())))
        sel_rep = st.selectbox("Select Sales Rep", ["All"] + all_reps)
        
        # Filter Dataframes
        if sel_rep != "All":
            df_main = df_main[df_main['Rep'] == sel_rep]
            df_deals = df_deals[df_deals['Rep'] == sel_rep]
            
        # 2. Customer
        cust_list = sorted(df_main['Customer'].unique())
        sel_cust = st.selectbox("Filter Customer", ["All"] + cust_list)
        
        if sel_cust != "All":
            df_main = df_main[df_main['Customer'] == sel_cust]
            df_deals = df_deals[df_deals['Matched_Customer'] == sel_cust]
            
        st.markdown("---")
        st.caption(f"📅 Current View: {CURRENT_YEAR}")
        st.caption(f"🔭 Forecast View: {NEXT_YEAR}")

    # HEADER
    st.title(f"Sales Performance & {NEXT_YEAR} Forecast")
    if sel_rep != "All":
        st.markdown(f"**Rep:** {sel_rep}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # GLOBAL METRICS (TOP ROW)
    # ══════════════════════════════════════════════════════════════════════════
    
    # 1. 2025 Revenue (Invoiced)
    rev_2025 = df_main[
        (df_main['Type'] == 'Invoiced') & 
        (df_main['Inv_Date'].dt.year == CURRENT_YEAR)
    ]['Amount'].sum()
    
    # 2. Q4 2025 Revenue (Invoiced)
    rev_q4 = df_main[
        (df_main['Type'] == 'Invoiced') & 
        (df_main['Inv_Date'].dt.year == CURRENT_YEAR) & 
        (df_main['Inv_Date'].dt.quarter == 4)
    ]['Amount'].sum()
    
    # 3. Active Backlog (Pending SOs)
    backlog = df_main[df_main['Type'] == 'Pending SO']['Amount'].sum()
    
    # 4. Pipeline Remainder of Year (Q4 2025 deals)
    pipe_q4 = df_deals[
        (df_deals['Close Date'].dt.year == CURRENT_YEAR) & 
        (df_deals['Close Date'].dt.quarter == 4)
    ]['Amount'].sum()

    # Display Metrics
    c1, c2, c3, c4 = st.columns(4)
    def kpi(label, val, col):
        col.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">${val:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    kpi(f"{CURRENT_YEAR} Total Invoiced", rev_2025, c1)
    kpi(f"Q4 {CURRENT_YEAR} Invoiced", rev_q4, c2)
    kpi("Active Pending Orders", backlog, c3)
    kpi(f"Pipeline Closing Q4 {CURRENT_YEAR}", pipe_q4, c4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TABBED VIEW
    # ══════════════════════════════════════════════════════════════════════════
    
    tab1, tab2 = st.tabs([f"📉 {CURRENT_YEAR} Performance & Active Orders", f"🔭 {NEXT_YEAR} Forecast Builder"])

    # --------------------------------------------------------------------------
    # TAB 1: CURRENT STATUS
    # --------------------------------------------------------------------------
    with tab1:
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader(f"✅ {CURRENT_YEAR} Invoiced Revenue (By Customer)")
            # Show top customers for 2025
            hist_25 = df_main[
                (df_main['Type'] == 'Invoiced') & 
                (df_main['Inv_Date'].dt.year == CURRENT_YEAR)
            ]
            if not hist_25.empty:
                cust_perf = hist_25.groupby('Customer')['Amount'].sum().sort_values(ascending=False).reset_index()
                st.dataframe(
                    cust_perf, 
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Amount": st.column_config.NumberColumn(format="$%d")}
                )
            else:
                st.info("No invoiced revenue for 2025 yet.")

        with c_right:
            st.subheader("📦 Pending Sales Orders (Backlog)")
            # Show specific Pending Orders details
            pending = df_main[df_main['Type'] == 'Pending SO'].copy()
            if not pending.empty:
                pending_view = pending[['SO_Date', 'Customer', 'SO_Num', 'Amount']].sort_values('SO_Date', ascending=False)
                st.dataframe(
                    pending_view,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "SO_Date": st.column_config.DateColumn("Date Created", format="MMM DD, YYYY"),
                        "Amount": st.column_config.NumberColumn(format="$%d")
                    }
                )
            else:
                st.success("No pending backorders! All caught up.")

        st.markdown("---")
        st.subheader(f"🎯 Pipeline Deals Closing Remaining {CURRENT_YEAR}")
        # Deals closing in Q4 2025
        deals_now = df_deals[
            (df_deals['Close Date'].dt.year == CURRENT_YEAR) & 
            (df_deals['Close Date'].dt.quarter == 4)
        ]
        if not deals_now.empty:
            st.dataframe(
                deals_now[['Close Date', 'Deal Name', 'Stage', 'Amount']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Close Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
                    "Amount": st.column_config.NumberColumn(format="$%d")
                }
            )
        else:
            st.info(f"No pipeline deals set to close in remainder of {CURRENT_YEAR}.")

    # --------------------------------------------------------------------------
    # TAB 2: 2026 FORECAST BUILDER
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown(f"### 🔮 {NEXT_YEAR} Quarterly Forecast")
        
        # Toggle for Granularity
        granularity = st.radio("Forecast Level:", ["Customer", "Item"], horizontal=True)
        
        # 1. GENERATE FORECAST TABLE
        forecast_df = generate_detailed_forecast(df_main, df_deals, granularity=granularity, year_target=NEXT_YEAR)
        
        if not forecast_df.empty:
            # Sort by Total 2026
            forecast_df = forecast_df.sort_values(f"Total {NEXT_YEAR}", ascending=False)
            
            # Display interactive table
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "2025 Actuals": st.column_config.NumberColumn(format="$%d", help="Total invoiced in 2025"),
                    f"Q1 {NEXT_YEAR}": st.column_config.NumberColumn(format="$%d"),
                    f"Q2 {NEXT_YEAR}": st.column_config.NumberColumn(format="$%d"),
                    f"Q3 {NEXT_YEAR}": st.column_config.NumberColumn(format="$%d"),
                    f"Q4 {NEXT_YEAR}": st.column_config.NumberColumn(format="$%d"),
                    f"Total {NEXT_YEAR}": st.column_config.NumberColumn(format="$%d"),
                }
            )
            
            # 2. TOTAL 2026 METRICS
            total_26 = forecast_df[f"Total {NEXT_YEAR}"].sum()
            growth = total_26 - rev_2025
            
            m1, m2 = st.columns(2)
            m1.metric(f"Total {NEXT_YEAR} Forecast", f"${total_26:,.0f}")
            m2.metric("Projected Growth vs 2025", f"${growth:,.0f}", delta_color="normal")
            
        else:
            st.warning("Insufficient data to generate forecast.")

        st.markdown("---")
        st.subheader(f"📋 Raw Pipeline for {NEXT_YEAR}")
        # Show the raw deals feeding the forecast
        deals_next = df_deals[df_deals['Close Date'].dt.year == NEXT_YEAR]
        if not deals_next.empty:
            st.dataframe(
                deals_next[['Close Date', 'Matched_Customer', 'Deal Name', 'Stage', 'Amount']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Close Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
                    "Amount": st.column_config.NumberColumn(format="$%d")
                }
            )
        else:
            st.info(f"No deals currently in pipeline for {NEXT_YEAR}.")

if __name__ == "__main__":
    main()
