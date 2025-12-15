# Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

### Step 1: Set up Google Sheets API (First Time Only)

1. **Go to Google Cloud Console:** https://console.cloud.google.com/
2. **Create or select a project**
3. **Enable Google Sheets API:**
   - Click on "APIs & Services" → "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

4. **Create Service Account:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "Service Account"
   - Name it (e.g., "demand-forecast-app")
   - Click "Create and Continue"
   - Skip optional steps → "Done"

5. **Create & Download Key:**
   - Click on your new service account
   - Go to "Keys" tab
   - Click "Add Key" → "Create New Key"
   - Choose JSON → "Create"
   - **Save this file securely!**

6. **Share Your Google Sheet:**
   - Open: https://docs.google.com/spreadsheets/d/15JhBZ_7aHHZA1W1qsoC2163borL6RYjk0xTDWPmWPfA/edit
   - Click "Share"
   - Add the email from your JSON file (looks like: `xxxxx@xxxxx.iam.gserviceaccount.com`)
   - Give "Viewer" or "Editor" access

### Step 2: Set Up Your Local Environment

```bash
# Clone your repository
git clone <your-repo-url>
cd demand_forecast

# Install dependencies
pip install -r requirements.txt

# Create .streamlit folder
mkdir .streamlit
```

### Step 3: Add Your Credentials

1. Copy the template:
   ```bash
   cp secrets.toml.template .streamlit/secrets.toml
   ```

2. Open `.streamlit/secrets.toml` and paste your credentials from the JSON file you downloaded:
   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "your-project-123456"
   private_key_id = "abc123..."
   private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_KEY_HERE\n-----END PRIVATE KEY-----\n"
   client_email = "demand-forecast@your-project.iam.gserviceaccount.com"
   client_id = "123456789"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
   universe_domain = "googleapis.com"
   ```

   **Note:** Just copy and paste each field from your downloaded JSON file.

### Step 4: Run the App

```bash
streamlit run demand_forecast_app.py
```

Your app will open at: http://localhost:8501

## ✅ Verification Checklist

- [ ] Google Sheets API enabled in GCP
- [ ] Service account created
- [ ] JSON key downloaded
- [ ] Google Sheet shared with service account email
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.streamlit/secrets.toml` created with credentials
- [ ] App running (`streamlit run demand_forecast_app.py`)

## 🐛 Troubleshooting

**"Error connecting to Google Sheets"**
- Make sure you shared the Google Sheet with your service account email
- Verify the credentials in `.streamlit/secrets.toml` are correct
- Check that the private key includes the full key with BEGIN/END tags

**"Module not found"**
- Run: `pip install -r requirements.txt`

**"Permission denied"**
- The service account needs access to your Google Sheet
- Go to the sheet → Share → Add service account email

## 🎯 Next Steps

Once you can see your data in the app:

1. **Explore the Data Preview tab** - Understand your data structure
2. **Check the Forecast tab** - This is where we'll build forecasting logic
3. **Start adding forecast models** - Use the `utils.py` helper functions

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Sheets API](https://developers.google.com/sheets/api)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)

---

Need help? Check the full README.md or create an issue in the repository.
