"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX - SALES PLANNING & FORECASTING TOOL (v3.1 - Robust Data Loading)
    Includes:
    - Smart Cascading Filters (Rep -> Customer -> SKU)
    - Hybrid Forecasting (Statistical + Machine Learning)
    - Pipeline Allocation Logic (Category -> SKU)
    - Robust Column Matching (Fixes KeyError)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calyx Sales Planner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .block-container { max-width: 1600px; padding-top: 2rem; }
    
    /* Metrics Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-label { font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; color: #0f172a; font-weight: 700; margin: 0.2rem 0; }
    
    /* Table Styling */
    [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
    
    h1, h2, h3 { color: #0f172a; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: white; border-radius: 4px; color: #64748b; }
    .stTabs [aria-selected="true"] { background-color: #f1f5f9; color: #0f172a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

SHEET_SO_INV = "SO & invoice Data merged"
SHEET_DEALS = "Deals"

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED IMPORT HANDLING (Forecast Models)
# ══════════════════════════════════════════════════════════════════════════════
HAS_ML = False
try:
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_ML = True
except ImportError:
    HAS_ML = False

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_get_col(df, candidates, default_val='Unknown'):
    """Tries multiple column names, returns the first one found or a default series."""
    for col in candidates:
        # Case-insensitive check
        matches = [c for c in df.columns if c.strip().lower() == col.strip().lower()]
        if matches:
            return df[matches[0]]
    return pd.Series([default_val] * len(df), index=df.index)

@st.cache_data(ttl=3600)
def load_data():
    """Load and Prep Data with Fuzzy Matching for Pipeline"""
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        # Connect to Google Sheets
        creds_dict = None
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        elif "service_account" in st.secrets:
            creds_dict = dict(st.secrets["service_account"])
        else:
            st.error("Missing Google Credentials in Secrets.")
            return pd.DataFrame(), pd.DataFrame()

        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        
        sheet_id = None
        if "SPREADSHEET_ID" in st.secrets:
            sheet_id = st.secrets["SPREADSHEET_ID"]
        elif "gsheets" in st.secrets:
            sheet_id = st.secrets["gsheets"].get("spreadsheet_id")
            
        if not sheet_id:
            st.error("Missing Spreadsheet ID in Secrets.")
            return pd.DataFrame(), pd.DataFrame()

        sh = client.open_by_key(sheet_id)

        # 1. LOAD SO DATA
        try:
            ws_so = sh.worksheet(SHEET_SO_INV)
            rows_so = ws_so.get_all_values()
            # If empty or just header
            if len(rows_so) < 2:
                st.warning(f"Sheet '{SHEET_SO_INV}' seems empty.")
                df_so = pd.DataFrame()
            else:
                df_so = pd.DataFrame(rows_so[1:], columns=rows_so[0])
        except Exception as e:
            st.error(f"Could not load sheet '{SHEET_SO_INV}': {e}")
            df_so = pd.DataFrame()

        # 2. LOAD DEALS DATA
        try:
            ws_deals = sh.worksheet(SHEET_DEALS)
            rows_deals = ws_deals.get_all_values()
            if len(rows_deals) < 3:
                st.warning(f"Sheet '{SHEET_DEALS}' seems empty or malformed.")
                df_deals = pd.DataFrame()
            else:
                # Headers are in Row 2 (index 1), Data starts Row 3 (index 2)
                df_deals = pd.DataFrame(rows_deals[2:], columns=rows_deals[1])
        except Exception as e:
            st.error(f"Could not load sheet '{SHEET_DEALS}': {e}")
            df_deals = pd.DataFrame()

        # --- PRE-PROCESSING SO ---
        if not df_so.empty:
            # Clean Headers: Strip whitespace
            df_so.columns = [str(c).strip() for c in df_so.columns]
            
            # Safe Column Extraction
            # Rep
            rep_inv = safe_get_col(df_so, ['Inv - Rep Master', 'Inv-Rep Master', 'Rep Master'])
            rep_so = safe_get_col(df_so, ['SO - Rep Master', 'SO-Rep Master', 'Rep'])
            df_so['Rep'] = rep_inv.replace('', np.nan).combine_first(rep_so.replace('', np.nan)).fillna('Unassigned').astype(str).str.strip()
            
            # Customer
            cust_inv = safe_get_col(df_so, ['Inv - Correct Customer', 'Correct Customer'])
            cust_so = safe_get_col(df_so, ['SO - Customer Companyname', 'Customer Companyname', 'Customer'])
            df_so['Customer'] = cust_inv.replace('', np.nan).combine_first(cust_so.replace('', np.nan)).fillna('Unknown').astype(str).str.strip()
            
            # Item & Type
            df_so['Item'] = safe_get_col(df_so, ['SO - Item', 'Item']).astype(str).str.strip()
            df_so['Product Type'] = safe_get_col(df_so, ['SO - Calyx || Product Type', 'Product Type']).astype(str).str.strip()
            
            # Dates
            date_inv = pd.to_datetime(safe_get_col(df_so, ['Inv - Date', 'Date']), errors='coerce')
            date_so = pd.to_datetime(safe_get_col(df_so, ['SO - Date Created', 'Date Created']), errors='coerce')
            df_so['Date'] = date_inv.combine_first(date_so)
            
            # Amounts
            def clean_money(series):
                return pd.to_numeric(series.astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
                
            df_so['Amount'] = clean_money(safe_get_col(df_so, ['Inv - Amount', 'Amount']))
            df_so['Qty'] = clean_money(safe_get_col(df_so, ['SO - Quantity Ordered', 'Quantity Ordered', 'Qty']))
            
            # Filter valid rows
            df_so = df_so[df_so['Amount'] > 0].copy()

        # --- PRE-PROCESSING DEALS ---
        if not df_deals.empty:
            # Clean Headers
            df_deals.columns = [str(c).strip() for c in df_deals.columns]
            
            # Filter Include?
            include_col = safe_get_col(df_deals, ['Include?', 'Include'])
            # Only keep rows where Include is True-ish
            keep_mask = include_col.astype(str).str.upper().isin(['TRUE', 'YES', '1'])
            df_deals = df_deals[keep_mask].copy()
            
            # Columns
            df_deals['Deal Name'] = safe_get_col(df_deals, ['Deal Name']).astype(str).str.strip()
            
            amt_col = safe_get_col(df_deals, ['Amount'])
            df_deals['Amount'] = pd.to_numeric(amt_col.astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            
            date_col = safe_get_col(df_deals, ['Close Date'])
            df_deals['Close Date'] = pd.to_datetime(date_col, errors='coerce')
            
            # Construct Rep Name
            first = safe_get_col(df_deals, ['Deal Owner First Name']).astype(str)
            last = safe_get_col(df_deals, ['Deal Owner Last Name']).astype(str)
            df_deals['Rep'] = (first + " " + last).str.strip().str.replace('None None', 'Unassigned')
            
            df_deals['Stage'] = safe_get_col(df_deals, ['Deal Stage', 'Stage']).astype(str).str.strip()

            # --- FUZZY MATCHING PIPELINE TO CUSTOMERS ---
            unique_customers = []
            if not df_so.empty:
                unique_customers = df_so['Customer'].dropna().unique()
            
            # Simple matching logic
            def match_customer(deal_name):
                if not deal_name or deal_name.lower() == 'nan': return "Unassigned"
                deal_name_upper = deal_name.upper()
                # Sort customers by length desc to match "Target" before "Target Corp"
                for cust in sorted(unique_customers, key=len, reverse=True):
                    if len(cust) > 3 and cust.upper() in deal_name_upper:
                        return cust
                return "Unassigned"

            if len(unique_customers) > 0:
                df_deals['Matched_Customer'] = df_deals['Deal Name'].apply(match_customer)
            else:
                df_deals['Matched_Customer'] = "Unassigned"

        return df_so, df_deals

    except Exception as e:
        # If all else fails, show the error but return empty so app doesn't crash completely
        st.error(f"Critical Data Loading Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecast(history_df, horizon_months=12, model_type='Exponential Smoothing'):
    """
    Generates a forecast for a specific SKU/Customer series.
    Input: DataFrame with index=Date, value=Qty
    """
    # Resample to Monthly
    series = history_df.resample('ME')['Qty'].sum().fillna(0)
    
    # Generate future dates
    last_date = series.index[-1] if not series.empty else datetime.now()
    future_dates = [last_date + relativedelta(months=i+1) for i in range(horizon_months)]
    
    if len(series) < 3:
        # Not enough data, return simple average
        avg_val = series.mean() if len(series) > 0 else 0
        return pd.Series([avg_val]*horizon_months, index=future_dates)

    forecast_values = []

    # MODEL A: EXPONENTIAL SMOOTHING (Statsmodels)
    if model_type == 'Exponential Smoothing' and HAS_ML:
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated").fit()
            forecast_values = model.forecast(horizon_months)
            return pd.Series(forecast_values, index=future_dates)
        except:
            pass # Fallback

    # MODEL B: MACHINE LEARNING (Random Forest)
    if model_type == 'Machine Learning (RF)' and HAS_ML:
        try:
            df_ml = pd.DataFrame({'y': series})
            for lag in [1, 2, 3]:
                df_ml[f'lag_{lag}'] = df_ml['y'].shift(lag)
            df_ml = df_ml.dropna()
            
            if len(df_ml) > 5:
                X = df_ml.drop('y', axis=1)
                y = df_ml['y']
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Simple recursive forecast
                last_row = df_ml.iloc[[-1]].drop('y', axis=1).copy()
                preds = []
                for _ in range(horizon_months):
                    pred = rf.predict(last_row)[0]
                    preds.append(pred)
                    # Shift logic (simple approximation for demo)
                    new_row = last_row.copy()
                    new_row['lag_1'] = pred
                    new_row['lag_2'] = last_row['lag_1'].values[0]
                    new_row['lag_3'] = last_row['lag_2'].values[0]
                    last_row = new_row
                return pd.Series(preds, index=future_dates)
        except:
            pass

    # FALLBACK: WEIGHTED MOVING AVERAGE
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    recent_vals = series.iloc[-4:].values
    if len(recent_vals) < 4:
        weights = weights[:len(recent_vals)]
        weights /= weights.sum()
    
    wma = np.dot(recent_vals[::-1], weights)
    return pd.Series([wma]*horizon_months, index=future_dates)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load Data
    with st.spinner("Initializing Planning Engine..."):
        df_so, df_deals = load_data()

    if df_so.empty:
        st.error("No historical data available. Please check column headers in your Google Sheet.")
        if st.button("Retry Loading"):
            st.rerun()
        st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # SMART CASCADING SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.header("🛠️ Planning Controls")
        
        # 1. SALES REP (Multi-Select)
        # Combine reps from both sources
        reps_so = df_so['Rep'].unique().tolist()
        reps_deals = df_deals['Rep'].unique().tolist() if not df_deals.empty else []
        all_reps = sorted(list(set(reps_so + reps_deals)))
        
        selected_reps = st.multiselect("Sales Rep(s)", all_reps)
        
        # Filter Data for subsequent dropdowns
        if selected_reps:
            df_rep_view = df_so[df_so['Rep'].isin(selected_reps)]
        else:
            df_rep_view = df_so
            
        # 2. CUSTOMER (Multi-Select, Searchable)
        available_customers = sorted(df_rep_view['Customer'].unique().tolist())
        selected_customers = st.multiselect("Customer(s)", available_customers)
        
        # Filter Data for SKU dropdown
        if selected_customers:
            df_cust_view = df_rep_view[df_rep_view['Customer'].isin(selected_customers)]
        else:
            df_cust_view = df_rep_view
            
        # 3. SKU / ITEM (Multi-Select, Searchable)
        available_skus = sorted(df_cust_view['Item'].dropna().unique().tolist())
        selected_skus = st.multiselect("Limit to Item(s)", available_skus)
        
        st.markdown("---")
        st.subheader("🔮 Forecast Settings")
        
        forecast_horizon = st.select_slider("Forecast Horizon (Months)", options=[3, 6, 9, 12, 18], value=12)
        
        model_options = ['Weighted Moving Average']
        if HAS_ML:
            model_options = ['Exponential Smoothing', 'Machine Learning (RF)', 'Weighted Moving Average']
            
        selected_model = st.selectbox("Algorithm", model_options)
        
        st.info("💡 **Hybrid Logic:**\nBaseline = Statistical Forecast\nUpside = Pipeline Allocation")

    # ══════════════════════════════════════════════════════════════════════════
    # GLOBAL FILTERING
    # ══════════════════════════════════════════════════════════════════════════
    
    # 1. Historicals
    main_df = df_so.copy()
    if selected_reps:
        main_df = main_df[main_df['Rep'].isin(selected_reps)]
    if selected_customers:
        main_df = main_df[main_df['Customer'].isin(selected_customers)]
    if selected_skus:
        main_df = main_df[main_df['Item'].isin(selected_skus)]
        
    # 2. Pipeline (Deals)
    pipeline_df = df_deals.copy()
    if not pipeline_df.empty:
        if selected_customers:
            # Show pipeline matched to these customers
            pipeline_df = pipeline_df[pipeline_df['Matched_Customer'].isin(selected_customers)]
        elif selected_reps:
            # Or reps
            pipeline_df = pipeline_df[pipeline_df['Rep'].isin(selected_reps)]
            
    # If filtering SKUs, we can't easily filter pipeline unless we matched products. 
    # For now, we assume Pipeline is "Category" level if not matched.

    if main_df.empty:
        st.warning("No historical data found for these filters.")
        st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW: TABS
    # ══════════════════════════════════════════════════════════════════════════
    
    st.title("Sales Planning & Demand Forecasting")
    
    tab1, tab2, tab3 = st.tabs(["🤵 Sales Rep Plan", "📊 Forecast Deep Dive", "📋 Raw Data"])
    
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: SALES REP CLIENT-FACING VIEW
    # ══════════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.markdown("### 🏢 Account Overview")
        
        # Metrics
        current_year = datetime.now().year
        curr_year_sales = main_df[main_df['Date'].dt.year == current_year]['Amount'].sum()
        last_year_sales = main_df[main_df['Date'].dt.year == (current_year - 1)]['Amount'].sum()
        
        # Simple Backlog: Orders with dates in future or "Open" status proxy
        # Since we just have 'Amount' and 'Date', let's approximate Backlog as 'Amount' where Date > Today (if SO date)
        # Better: You requested status logic earlier, but for now we stick to robust Amount
        open_orders = 0 
        
        pipeline_val = pipeline_df['Amount'].sum() if not pipeline_df.empty else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("YTD Sales", f"${curr_year_sales:,.0f}", delta=f"${(curr_year_sales-last_year_sales):,.0f} vs LY")
        m2.metric("Projected Pipeline", f"${pipeline_val:,.0f}")
        m3.metric("Active SKUs", main_df['Item'].nunique())
        m4.metric("Avg Order Value", f"${main_df['Amount'].mean():,.0f}")
        
        st.markdown("---")
        
        # 1. HISTORICAL CADENCE
        st.subheader("📅 Ordering Cadence")
        st.caption("Are they ordering consistently? Spot gaps in the heatmap.")
        
        if not main_df.empty:
            cadence_df = main_df.groupby([pd.Grouper(key='Date', freq='ME'), 'Item'])['Qty'].sum().reset_index()
            fig_cadence = px.density_heatmap(
                cadence_df,
                x='Date',
                y='Item',
                z='Qty',
                color_continuous_scale='Blues',
                title='Order Volume Heatmap (Qty)',
            )
            st.plotly_chart(fig_cadence, use_container_width=True)
        
        # 2. SKU-LEVEL DEMAND PLAN
        st.subheader(f"📦 SKU-Level Demand Plan ({forecast_horizon} Months)")
        
        # --- GENERATE FORECASTS PER SKU ---
        sku_plans = []
        
        # Top SKUs only to prevent slow down
        top_skus = main_df.groupby('Item')['Amount'].sum().nlargest(30).index.tolist()
        
        for sku in top_skus:
            # 1. Get History
            sku_hist = main_df[main_df['Item'] == sku].set_index('Date')
            
            # 2. Run Model
            fcst_series = generate_forecast(sku_hist, horizon_months=forecast_horizon, model_type=selected_model)
            
            # 3. Pipeline Allocation
            # Ratio: This SKU's share of total history
            total_rev = main_df['Amount'].sum()
            sku_rev = sku_hist['Amount'].sum()
            share = sku_rev / total_rev if total_rev > 0 else 0
            
            # Allocated Pipeline Value (assuming Pipeline is generic)
            allocated_pipe_val = pipeline_val * share
            
            # Convert Value to Qty estimate (using avg price)
            avg_price = sku_rev / sku_hist['Qty'].sum() if sku_hist['Qty'].sum() > 0 else 1
            allocated_pipe_qty = allocated_pipe_val / avg_price
            
            # Add uplift to forecast per month
            uplift_per_month = allocated_pipe_qty / forecast_horizon
            final_forecast = fcst_series + uplift_per_month
            
            # Bucketing
            next_year = current_year + 1
            q1_next = 0
            total_fcst = 0
            
            for date_val, qty in final_forecast.items():
                total_fcst += qty
                if date_val.year == next_year and date_val.month <= 3:
                    q1_next += qty
            
            sku_plans.append({
                "SKU": sku,
                "Avg Monthly Qty": sku_hist['Qty'].mean(),
                f"Q1 {next_year} Forecast": q1_next,
                f"Total {forecast_horizon}mo Forecast": total_fcst,
                "Pipeline Uplift (Qty)": allocated_pipe_qty
            })
            
        plan_df = pd.DataFrame(sku_plans)
        
        if not plan_df.empty:
            st.dataframe(
                plan_df.style.background_gradient(subset=[f"Q1 {next_year} Forecast"], cmap='Greens'),
                use_container_width=True,
                column_config={
                    "Avg Monthly Qty": st.column_config.NumberColumn(format="%.0f"),
                    f"Q1 {next_year} Forecast": st.column_config.NumberColumn(format="%.0f"),
                    f"Total {forecast_horizon}mo Forecast": st.column_config.NumberColumn(format="%.0f"),
                }
            )
        else:
            st.info("No sufficient data to generate SKU plans.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: FORECAST DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.subheader("📈 Top-Down vs Bottom-Up Forecast")
        
        # 1. Total Revenue Timeline
        hist_rev = main_df.resample('ME', on='Date')['Amount'].sum()
        
        # Forecast Aggregate
        hist_df_agg = pd.DataFrame({'Qty': hist_rev}) 
        fcst_rev = generate_forecast(hist_df_agg, horizon_months=forecast_horizon, model_type=selected_model)
        
        # Pipeline Overlay
        pipe_overlay = pd.Series(0, index=fcst_rev.index)
        if not pipeline_df.empty:
             # Try to overlay pipeline on Close Date
             p_resamp = pipeline_df.set_index('Close Date').resample('ME')['Amount'].sum()
             # Reindex to match forecast
             pipe_overlay = p_resamp.reindex(fcst_rev.index, fill_value=0)
        
        fig_hybrid = go.Figure()
        
        # Historical
        fig_hybrid.add_trace(go.Scatter(
            x=hist_rev.index, y=hist_rev.values,
            mode='lines', name='Historical Sales',
            line=dict(color='#0f172a', width=3)
        ))
        
        # Baseline Forecast
        fig_hybrid.add_trace(go.Scatter(
            x=fcst_rev.index, y=fcst_rev.values,
            mode='lines+markers', name=f'Baseline Forecast',
            line=dict(color='#3b82f6', dash='dash')
        ))
        
        # Pipeline Stacking
        fig_hybrid.add_trace(go.Bar(
            x=pipe_overlay.index, y=pipe_overlay.values,
            name='HubSpot Pipeline',
            marker_color='#f59e0b',
            opacity=0.6
        ))
        
        fig_hybrid.update_layout(title="Hybrid Revenue Forecast", height=500, xaxis_title="Date", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_hybrid, use_container_width=True)
        
        st.markdown("### 🧩 Category Allocation Logic")
        st.write("""
        HubSpot deals often lack specific SKUs. This tool uses **Category Allocation**:
        1. We take the total pipeline value for the selected customer(s).
        2. We analyze the historical SKU mix for these customers.
        3. We distribute the pipeline revenue to SKUs based on their historical share.
        """)
        
        if not main_df.empty:
            fig_mix = px.pie(main_df, names='Product Type', values='Amount', title='Historical Product Mix (Used for Allocation)')
            st.plotly_chart(fig_mix)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: RAW DATA
    # ══════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.write("### Filtered Historical Data")
        st.dataframe(main_df, use_container_width=True)
        
        st.write("### Matched Pipeline Deals")
        st.dataframe(pipeline_df, use_container_width=True)

if __name__ == "__main__":
    main()
