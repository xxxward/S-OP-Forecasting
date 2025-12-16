"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX - SALES REP PLANNING & FORECASTING DASHBOARD
    Single Data Source: SO & invoice Data merged
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sales Rep Planning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .block-container { padding: 2rem 3rem; max-width: 1800px; }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        z-index: -1;
        opacity: 0.8;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255, 255, 255, 0.8) !important;
    }
    h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    }
    h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    [data-testid="stDataFrame"] table { color: #ffffff !important; }
    [data-testid="stDataFrame"] thead tr th {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(20px);
    }
</style>
""", unsafe_allow_html=True)

SHEET_NAME = "SO & invoice Data merged"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load merged SO & Invoice data from Google Sheets"""
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        # Get credentials
        creds_dict = None
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        elif "service_account" in st.secrets:
            creds_dict = dict(st.secrets["service_account"])
        else:
            raise ValueError("Missing Google credentials")
        
        # Get sheet ID
        sheet_id = None
        if "SPREADSHEET_ID" in st.secrets:
            sheet_id = st.secrets["SPREADSHEET_ID"]
        elif "gsheets" in st.secrets:
            sheet_id = st.secrets["gsheets"].get("spreadsheet_id")
        
        if not sheet_id:
            raise ValueError("Missing spreadsheet ID")

        # Connect
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
        
        # Load merged data
        ws = sh.worksheet(SHEET_NAME)
        rows = ws.get_all_values()
        
        if len(rows) > 1:
            headers = rows[0]
            
            # ═══ Deduplicate Headers ═══
            deduped_headers = []
            seen = {}
            for h in headers:
                h = h.strip()
                if h in seen:
                    seen[h] += 1
                    deduped_headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    deduped_headers.append(h)
            
            df = pd.DataFrame(rows[1:], columns=deduped_headers)
            df = df.replace('', np.nan)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare merged SO & Invoice data
    """
    
    # Check if required columns exist
    required_cols = ['Inv - Rep Master', 'SO - Rep Master', 'Inv - Correct Customer', 'SO - Customer Companyname']
    df.columns = df.columns.str.strip()
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()
    
    df = df.copy()
    
    # ═══ Sales Rep Master ═══
    inv_rep = df['Inv - Rep Master']
    so_rep = df['SO - Rep Master']
    df['sales_rep_master'] = inv_rep.combine_first(so_rep)
    
    # ═══ Customer Corrected ═══
    df['customer_corrected'] = df['Inv - Correct Customer'].combine_first(df['SO - Customer Companyname'])
    
    # ═══ Date Parsing ═══
    date_cols = {
        'SO - Date Created': 'so_date_created',
        'SO - Pending Fulfillment Date': 'so_pending_fulfillment',
        'SO - Actual Ship Date': 'so_actual_ship',
        'SO - Date Billed': 'so_date_billed',
        'SO - Date Closed': 'so_date_closed',
        'Inv - Date': 'inv_date',
        'Inv - Date Closed': 'inv_date_closed',
        'Inv - Due Date': 'inv_due_date'
    }
    
    for orig_col, new_col in date_cols.items():
        if orig_col in df.columns:
            df[new_col] = pd.to_datetime(df[orig_col], errors='coerce')
    
    # ═══ Numeric Conversions ═══
    numeric_cols = {
        'SO - Amount': 'so_amount',
        'SO - Quantity Ordered': 'so_qty_ordered',
        'SO - Item Rate': 'so_item_rate',
        'Inv - Amount': 'inv_amount',
        'Inv - Quantity': 'inv_quantity',
        'Inv - Amount Remaining': 'inv_amount_remaining',
        'Inv - Amount (Transaction Total)': 'inv_amount_total'
    }
    
    for orig_col, new_col in numeric_cols.items():
        if orig_col in df.columns:
            df[new_col] = pd.to_numeric(df[orig_col], errors='coerce').fillna(0)
    
    # ═══ String Cleaning ═══
    string_cols = ['SO - Status', 'SO - Item', 'SO - Document Number', 
                   'SO - Calyx || Product Type', 'SO - HubSpot Pipeline']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # ═══ FILTER: Exclude Cancelled and $0 Orders ═══
    if 'SO - Status' in df.columns:
        df = df[df['SO - Status'] != 'Cancelled']
    
    if 'so_amount' in df.columns:
        df = df[df['so_amount'] > 0]
    
    # ═══ Aggregate Invoice Metrics per SO Line ═══
    agg_cols = {
        'inv_quantity': 'sum',
        'inv_amount': 'sum',
        'inv_amount_remaining': 'sum'
    }
    
    if 'SO - Internal ID' in df.columns and 'SO - Item' in df.columns:
        df['so_line_id'] = df['SO - Internal ID'].astype(str) + '_' + df['SO - Item'].astype(str)
        
        # Aggregate invoice data
        inv_agg = df.groupby('so_line_id').agg(agg_cols).reset_index()
        inv_agg.columns = ['so_line_id', 'actual_qty_billed', 'actual_revenue_billed', 'total_amount_remaining']
        
        # Merge back
        df = df.merge(inv_agg, on='so_line_id', how='left')
        
        # Fill NaN for lines without invoices
        df['actual_qty_billed'] = df['actual_qty_billed'].fillna(0)
        df['actual_revenue_billed'] = df['actual_revenue_billed'].fillna(0)
        df['total_amount_remaining'] = df['total_amount_remaining'].fillna(0)
        
        # ═══ DERIVED METRICS ═══
        df['qty_remaining'] = df['so_qty_ordered'] - df['actual_qty_billed']
        df['revenue_remaining'] = df['so_amount'] - df['actual_revenue_billed']
        
        # ═══ BOOLEAN FLAGS ═══
        df['is_fully_invoiced'] = (df['actual_revenue_billed'] >= df['so_amount']) & (df['actual_revenue_billed'] > 0)
        df['is_partially_invoiced'] = (df['actual_revenue_billed'] > 0) & (df['actual_revenue_billed'] < df['so_amount'])
        df['is_not_invoiced'] = df['actual_revenue_billed'] == 0
        
        # ═══ Invoice Lag ═══
        if 'inv_date' in df.columns and 'so_date_created' in df.columns:
            df['invoice_lag_days'] = (df['inv_date'] - df['so_date_created']).dt.days
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def revenue_forecast_by_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue Forecast by Sales Rep & Month"""
    df['forecast_month'] = df['so_pending_fulfillment'].dt.to_period('M').dt.to_timestamp()
    
    forecast = df.groupby(['sales_rep_master', 'forecast_month']).agg({
        'so_amount': 'sum',
        'actual_revenue_billed': 'sum',
        'revenue_remaining': 'sum'
    }).reset_index()
    
    forecast.columns = ['Sales Rep', 'Month', 'Planned Revenue', 'Actual Revenue', 'Remaining Revenue']
    
    return forecast.sort_values(['Sales Rep', 'Month'])

def pipeline_health_by_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline Health per Sales Rep"""
    health = df.groupby('sales_rep_master').agg({
        'so_amount': 'sum',
        'actual_revenue_billed': 'sum',
        'revenue_remaining': 'sum',
        'SO - Internal ID': 'count'
    }).reset_index()
    
    health.columns = ['Sales Rep', 'Total Planned Revenue', 'Actual Revenue', 'Remaining Revenue', 'Total SO Lines']
    
    # Calculate % invoiced
    health['% Invoiced'] = (health['Actual Revenue'] / health['Total Planned Revenue'] * 100).fillna(0)
    health['% Remaining'] = (health['Remaining Revenue'] / health['Total Planned Revenue'] * 100).fillna(0)
    
    # Count open SO lines (not fully invoiced)
    open_lines = df[~df['is_fully_invoiced']].groupby('sales_rep_master').size().reset_index(name='Open SO Lines')
    
    # ═══ FIX: Rename column to match 'health' dataframe before merge ═══
    open_lines.rename(columns={'sales_rep_master': 'Sales Rep'}, inplace=True)
    
    health = health.merge(open_lines, on='Sales Rep', how='left')
    health['Open SO Lines'] = health['Open SO Lines'].fillna(0).astype(int)
    
    return health

def invoice_lag_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Average invoice lag per Sales Rep"""
    invoiced = df[df['actual_revenue_billed'] > 0].copy()
    
    if invoiced.empty:
        return pd.DataFrame()
    
    lag = invoiced.groupby('sales_rep_master')['invoice_lag_days'].mean().reset_index()
    lag.columns = ['Sales Rep', 'Avg Invoice Lag (Days)']
    lag['Avg Invoice Lag (Days)'] = lag['Avg Invoice Lag (Days)'].round(1)
    
    return lag.sort_values('Avg Invoice Lag (Days)', ascending=False)

def invoice_status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count of SO lines by invoice status per Sales Rep"""
    breakdown = df.groupby('sales_rep_master').agg({
        'is_fully_invoiced': 'sum',
        'is_partially_invoiced': 'sum',
        'is_not_invoiced': 'sum'
    }).reset_index()
    
    breakdown.columns = ['Sales Rep', 'Fully Invoiced', 'Partially Invoiced', 'Not Invoiced']
    
    return breakdown

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def create_revenue_forecast_chart(forecast_df: pd.DataFrame, selected_rep: str):
    """Create revenue forecast chart for selected rep"""
    rep_data = forecast_df[forecast_df['Sales Rep'] == selected_rep]
    
    if rep_data.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=rep_data['Month'],
        y=rep_data['Planned Revenue'],
        name='Planned',
        marker_color='rgba(102, 126, 234, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=rep_data['Month'],
        y=rep_data['Actual Revenue'],
        name='Actual',
        marker_color='rgba(79, 172, 254, 0.9)'
    ))
    
    fig.add_trace(go.Bar(
        x=rep_data['Month'],
        y=rep_data['Remaining Revenue'],
        name='Remaining',
        marker_color='rgba(245, 87, 108, 0.7)'
    ))
    
    fig.update_layout(
        title=f'Revenue Forecast - {selected_rep}',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Revenue ($)'),
        height=500
    )
    
    return fig

def create_pipeline_health_chart(health_df: pd.DataFrame):
    """Create pipeline health bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=health_df['Sales Rep'],
        y=health_df['% Invoiced'],
        name='% Invoiced',
        marker_color='rgba(79, 172, 254, 0.9)',
        text=health_df['% Invoiced'].round(1),
        textposition='outside',
        texttemplate='%{text}%'
    ))
    
    fig.add_trace(go.Bar(
        x=health_df['Sales Rep'],
        y=health_df['% Remaining'],
        name='% Remaining',
        marker_color='rgba(245, 87, 108, 0.7)',
        text=health_df['% Remaining'].round(1),
        textposition='outside',
        texttemplate='%{text}%'
    ))
    
    fig.update_layout(
        title='Pipeline Health by Sales Rep',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Percentage'),
        height=500
    )
    
    return fig

def create_invoice_status_chart(status_df: pd.DataFrame):
    """Create stacked bar chart for invoice status"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=status_df['Sales Rep'],
        y=status_df['Fully Invoiced'],
        name='Fully Invoiced',
        marker_color='rgba(79, 172, 254, 0.9)'
    ))
    
    fig.add_trace(go.Bar(
        x=status_df['Sales Rep'],
        y=status_df['Partially Invoiced'],
        name='Partially Invoiced',
        marker_color='rgba(240, 147, 251, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=status_df['Sales Rep'],
        y=status_df['Not Invoiced'],
        name='Not Invoiced',
        marker_color='rgba(245, 87, 108, 0.7)'
    ))
    
    fig.update_layout(
        title='Invoice Status Breakdown',
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Count of SO Lines'),
        height=500
    )
    
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
        st.markdown("---")
        
        st.markdown("### Data Source")
        st.info(f"📊 {SHEET_NAME}")
        
        st.markdown("---")
        st.markdown("### 📡 Status")
        st.success("✓ Connected")
        st.caption(f"Updated: {datetime.now().strftime('%I:%M %p')}")
    
    # Header
    st.markdown('<h1>📊 Sales Rep Planning & Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">Merged SO & Invoice Analysis</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔮 Loading merged data..."):
        raw_df = load_data()
    
    if raw_df.empty:
        st.error("🚫 No data available")
        st.stop()
    
    # Prepare data
    df = prepare_data(raw_df)
    
    if df.empty:
        st.error("🚫 No data after filtering")
        st.stop()
    
    # Get unique sales reps
    sales_reps = sorted(df['sales_rep_master'].unique())
    
    # ═══ TOP METRICS ═══
    total_planned = df['so_amount'].sum()
    total_actual = df['actual_revenue_billed'].sum()
    total_remaining = df['revenue_remaining'].sum()
    pct_invoiced = (total_actual / total_planned * 100) if total_planned > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Planned Revenue", f"${total_planned:,.0f}")
    col2.metric("✅ Actual Revenue", f"${total_actual:,.0f}")
    col3.metric("⏳ Remaining", f"${total_remaining:,.0f}")
    col4.metric("📊 % Invoiced", f"{pct_invoiced:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══ ANALYTICS SECTIONS ═══
    
    # 1. Revenue Forecast by Rep
    st.markdown("## 📈 Revenue Forecast by Sales Rep")
    
    selected_rep = st.selectbox("👤 Select Sales Rep", sales_reps)
    
    forecast_df = revenue_forecast_by_rep(df)
    fig_forecast = create_revenue_forecast_chart(forecast_df, selected_rep)
    
    if fig_forecast:
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Show table
    rep_forecast = forecast_df[forecast_df['Sales Rep'] == selected_rep]
    if not rep_forecast.empty:
        rep_forecast['Month'] = rep_forecast['Month'].dt.strftime('%b %Y')
        rep_forecast['Planned Revenue'] = rep_forecast['Planned Revenue'].apply(lambda x: f"${x:,.0f}")
        rep_forecast['Actual Revenue'] = rep_forecast['Actual Revenue'].apply(lambda x: f"${x:,.0f}")
        rep_forecast['Remaining Revenue'] = rep_forecast['Remaining Revenue'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(rep_forecast[['Month', 'Planned Revenue', 'Actual Revenue', 'Remaining Revenue']], 
                     use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 2. Pipeline Health
    st.markdown("## 🎯 Pipeline Health by Sales Rep")
    
    health_df = pipeline_health_by_rep(df)
    fig_health = create_pipeline_health_chart(health_df)
    st.plotly_chart(fig_health, use_container_width=True)
    
    # Format table
    health_display = health_df.copy()
    health_display['Total Planned Revenue'] = health_display['Total Planned Revenue'].apply(lambda x: f"${x:,.0f}")
    health_display['Actual Revenue'] = health_display['Actual Revenue'].apply(lambda x: f"${x:,.0f}")
    health_display['Remaining Revenue'] = health_display['Remaining Revenue'].apply(lambda x: f"${x:,.0f}")
    health_display['% Invoiced'] = health_display['% Invoiced'].apply(lambda x: f"{x:.1f}%")
    health_display['% Remaining'] = health_display['% Remaining'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(health_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 3. Invoice Lag Analysis
    st.markdown("## ⏱️ Invoice Lag Analysis")
    
    lag_df = invoice_lag_analysis(df)
    
    if not lag_df.empty:
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Bar chart
            fig_lag = go.Figure()
            fig_lag.add_trace(go.Bar(
                x=lag_df['Sales Rep'],
                y=lag_df['Avg Invoice Lag (Days)'],
                marker_color='rgba(240, 147, 251, 0.7)',
                text=lag_df['Avg Invoice Lag (Days)'],
                textposition='outside'
            ))
            fig_lag.update_layout(
                title='Average Invoice Lag by Sales Rep',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Days'),
                height=400
            )
            st.plotly_chart(fig_lag, use_container_width=True)
        
        with col_b:
            st.dataframe(lag_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("No invoiced data available for lag analysis")
    
    st.markdown("---")
    
    # 4. Invoice Status Breakdown
    st.markdown("## 📋 Invoice Status Breakdown")
    
    status_df = invoice_status_breakdown(df)
    fig_status = create_invoice_status_chart(status_df)
    st.plotly_chart(fig_status, use_container_width=True)
    
    st.dataframe(status_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
