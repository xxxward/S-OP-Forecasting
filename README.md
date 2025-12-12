# MJBiz ROI Dashboard

Quick Streamlit dashboard for tracking MJBiz conference ROI metrics.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run mjbiz_roi_dashboard.py
   ```

3. **Open in browser:**
   - Should automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in terminal

## Features

### Metrics Tracked:
- ✅ New Companies created (last week + this week, by rep)
- ✅ New Contacts created (last week + this week, by rep)
- ✅ New Deals created (count + $ value, last week + this week, by rep)
- ✅ Meetings logged (last week, by rep)

### Two Input Methods:

#### 1. Manual Entry (Quick)
- Use sidebar to enter totals and rep breakdowns
- Format for reps:
  - **Companies/Contacts:** `Name:LastWeek:ThisWeek`
    - Example: `Brad Sherman:5:3, Jake Lynch:2:4`
  - **Deals:** `Name:LWCount:LWValue:TWCount:TWValue`
    - Example: `Brad Sherman:3:50000:2:30000`
  - **Meetings:** `Name:Count`
    - Example: `Brad Sherman:12, Jake Lynch:8`

#### 2. CSV Upload (Works with HubSpot exports)
- Export your HubSpot custom reports as CSV
- Upload via sidebar
- Dashboard automatically detects owner columns (Deal owner, HubSpot Owner, etc.)
- **Note:** Since HubSpot exports don't include date ranges or amounts in the CSV:
  - You'll need to manually specify "Last Week vs This Week" split in sidebar
  - You'll need to manually enter total deal values in sidebar
  - **Recommended:** Export separate CSVs filtered by date range in HubSpot for accuracy

## Dashboard Sections:

1. **Key Metrics Overview** - Top-level totals with week-over-week changes
2. **Companies & Contacts** - Side-by-side bar charts with rep breakdowns
3. **Deals** - Count and value tracking with rep details
4. **Meetings** - Bar chart showing meeting activity by rep
5. **ROI Summary** - Quick summary stats for Kyle's calculation
6. **Export** - Download summary as CSV

## Tips:

- **For tomorrow's meeting:** Use Manual Entry method - fastest way to input data
- **For recurring reports:** Set up CSV exports from HubSpot and save templates
- **Sharing:** Can share the app link, or export summary CSV and attach to email

## Example Data Entry:

If you had this data from HubSpot:
- Brad Sherman: 5 companies last week, 3 this week
- Jake Lynch: 2 companies last week, 4 this week

Enter in sidebar:
```
Brad Sherman:5:3, Jake Lynch:2:4
```

Dashboard will calculate totals and show visualizations automatically.

## HubSpot Export Best Practices:

For the most accurate CSV upload workflow:

1. **Create 2 separate deal reports in HubSpot:**
   - Report 1: Filter by "Create date" = "Last week" → Export as CSV
   - Report 2: Filter by "Create date" = "This week" → Export as CSV

2. **Include these columns in your HubSpot export:**
   - Deal owner (automatically detected)
   - Deal name
   - Amount (if available)

3. **Upload workflow:**
   - Upload "Last Week" CSV
   - Note the count in sidebar
   - Upload "This Week" CSV  
   - Dashboard will auto-calculate by owner

4. **For deals without Amount column:**
   - Manually enter total pipeline value in sidebar after upload
   - Dashboard will show count by rep, but totals for value
