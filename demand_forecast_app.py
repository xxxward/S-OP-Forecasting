import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Demand Forecast",
    page_icon="📊",
    layout="wide"
)

# Google Sheets connection
@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets client using service account credentials"""
    try:
        # Define the scope
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Load credentials from Streamlit secrets
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scope
        )
        
        # Authorize and return client
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

def detect_header_row(data, max_rows_to_check=5):
    """
    Auto-detect which row contains the headers
    Returns the index (0-based) of the header row
    """
    if not data or len(data) == 0:
        return 0
    
    # Check first few rows to find headers
    for i in range(min(max_rows_to_check, len(data))):
        row = data[i]
        
        # Count non-empty cells
        non_empty = sum(1 for cell in row if cell and str(cell).strip())
        
        # If row has at least 3 non-empty cells, consider it a potential header
        if non_empty >= 3:
            # Check if next row exists and looks like data (not another header)
            if i + 1 < len(data):
                next_row = data[i + 1]
                next_non_empty = sum(1 for cell in next_row if cell and str(cell).strip())
                
                # If next row also has data, this is likely the header row
                if next_non_empty >= 3:
                    return i
            else:
                # Last row with data, assume it's header
                return i
    
    # Default to first row if nothing found
    return 0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheet_data(sheet_url, worksheet_name=None):
    """Load data from Google Sheet with auto-detected headers"""
    try:
        client = get_google_sheets_client()
        if client is None:
            return None
        
        # Open the spreadsheet
        spreadsheet = client.open_by_url(sheet_url)
        
        # Get the first worksheet or specified worksheet
        if worksheet_name:
            worksheet = spreadsheet.worksheet(worksheet_name)
        else:
            worksheet = spreadsheet.get_worksheet(0)
        
        # Get all values and convert to DataFrame
        data = worksheet.get_all_values()
        
        if not data:
            st.warning("No data found in the sheet")
            return None
        
        # Auto-detect header row
        header_index = detect_header_row(data)
        
        # Convert to DataFrame using detected header row
        headers = data[header_index]
        data_rows = data[header_index + 1:]
        
        if not data_rows:
            st.warning("No data rows found after header row")
            return None
        
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Remove completely empty rows
        df = df.replace('', None)
        df = df.dropna(how='all')
        
        # Show which row was detected as header (helpful for debugging)
        st.sidebar.success(f"✓ Headers detected in row {header_index + 1}")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main app
def main():
    st.title("📊 Demand Planning Forecast")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Google Sheet URL
        sheet_url = "https://docs.google.com/spreadsheets/d/15JhBZ_7aHHZA1W1qsoC2163borL6RYjk0xTDWPmWPfA/edit?usp=sharing"
        
        st.info("📋 Connected to: Demand_planning_DB_aistudio")
        
        # Worksheet selector
        worksheet_name = st.text_input("Worksheet Name (leave blank for first sheet)", "")
        
        st.caption("💡 Headers are auto-detected")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading data from Google Sheets..."):
        df = load_google_sheet_data(
            sheet_url, 
            worksheet_name if worksheet_name else None
        )
    
    if df is not None:
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Worksheets", "Connected ✓")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Data Preview", "🔮 Forecast"])
        
        with tab1:
            st.subheader("Raw Data")
            st.dataframe(df, use_container_width=True, height=500)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="demand_planning_data.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("Data Preview")
            
            # Show basic stats
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Non-Null Count': [df[col].notna().sum() for col in df.columns],
                'Data Type': [df[col].dtype for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Show first few rows
            st.write("**First 10 Rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab3:
            st.subheader("Forecast Builder")
            st.info("🚧 Forecast functionality will be built here")
            
            # Placeholder for forecast logic
            st.write("Available columns for forecasting:")
            st.write(df.columns.tolist())
            
            # Column selector for forecast
            forecast_column = st.selectbox("Select column to forecast", df.columns.tolist())
            
            if forecast_column:
                st.write(f"Preview of **{forecast_column}**:")
                st.write(df[forecast_column].head(20))
    else:
        st.warning("⚠️ Unable to load data. Please check your Google Sheets credentials.")
        st.info("""
        **Setup Instructions:**
        1. Create a Google Cloud project
        2. Enable Google Sheets API
        3. Create a service account
        4. Download the service account JSON key
        5. Add the credentials to `.streamlit/secrets.toml`
        6. Share your Google Sheet with the service account email
        """)

if __name__ == "__main__":
    main()
