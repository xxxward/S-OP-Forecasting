# Demand Planning Forecast App

A Streamlit application for demand planning and forecasting using Google Sheets as a data source.

## Features

- 📊 Real-time Google Sheets integration
- 🔄 Automatic data refresh
- 📈 Data visualization with Plotly
- 🔮 Forecast builder (coming soon)
- 📥 CSV export functionality

## Setup Instructions

### 1. Clone this repository

```bash
git clone <your-repo-url>
cd demand_forecast
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Google Sheets API

#### Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Sheets API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

#### Create a Service Account

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Fill in the service account details and click "Create"
4. Skip granting additional access (click "Continue" then "Done")
5. Click on the newly created service account
6. Go to the "Keys" tab
7. Click "Add Key" > "Create New Key"
8. Choose "JSON" and click "Create"
9. Save the downloaded JSON file securely

#### Share your Google Sheet

1. Open your Google Sheet: [Demand_planning_DB_aistudio](https://docs.google.com/spreadsheets/d/15JhBZ_7aHHZA1W1qsoC2163borL6RYjk0xTDWPmWPfA/edit?usp=sharing)
2. Click "Share" button
3. Add the service account email (found in your JSON file: `client_email`)
4. Give it "Viewer" or "Editor" permissions

### 5. Configure Streamlit Secrets

Create a `.streamlit` folder in your project root:

```bash
mkdir .streamlit
```

Create a `secrets.toml` file inside `.streamlit/`:

```bash
touch .streamlit/secrets.toml
```

Add your Google service account credentials to `.streamlit/secrets.toml`:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Private-Key-Here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

**Important:** The `.streamlit/` folder is in `.gitignore` to prevent accidentally committing your credentials.

### 6. Run the app

```bash
streamlit run demand_forecast_app.py
```

The app will open in your browser at `http://localhost:8501`

## GitHub Codespaces Setup

If you're using GitHub Codespaces, the devcontainer will automatically:
- Install all dependencies
- Start the Streamlit server
- Forward port 8501 for preview

You'll still need to add your `.streamlit/secrets.toml` file in Codespaces.

## Project Structure

```
demand_forecast/
├── .devcontainer/
│   └── devcontainer.json       # Codespaces/devcontainer configuration
├── .streamlit/
│   └── secrets.toml            # Google Sheets credentials (not in git)
├── demand_forecast_app.py      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Usage

1. **View Raw Data:** See all data from your Google Sheet
2. **Data Preview:** Explore column information and data types
3. **Forecast:** Build and visualize demand forecasts (coming soon)

## Development

### Adding new features

The main app is in `demand_forecast_app.py`. Key areas to extend:

- **Forecast logic:** Add forecasting algorithms in the "Forecast" tab
- **Data processing:** Add data transformation functions
- **Visualizations:** Create custom Plotly charts

### Refreshing data

Click the "🔄 Refresh Data" button in the sidebar to clear the cache and reload from Google Sheets.

## Troubleshooting

### "Error connecting to Google Sheets"

- Verify your service account credentials in `.streamlit/secrets.toml`
- Ensure the Google Sheet is shared with your service account email
- Check that the Google Sheets API is enabled in your GCP project

### "No data found in the sheet"

- Verify the sheet URL is correct
- Check that the worksheet name (if specified) exists
- Ensure the sheet has data (at least headers)

## Contributing

Feel free to submit issues or pull requests!

## License

MIT License
