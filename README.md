# Pie Chart Analyzer - Quick Start Guide

## Requirements

- **Python 3.8+** (recommended: 3.10+)
- **Google Cloud Project** with:
  - Gmail API enabled
  - Google Sheets API enabled
  - OAuth 2.0 credentials (download `credentials.json`)
  - See "GOOGLE-SETUP-GUIDE.md" for detailed instructions on how to register a Google Cloud project
- **Google Sheet** an empty Google Sheet for results
- **config.json** a configuration file

## Initial Setup

1. **Obtain the Script Package**
   - Download the provided zip package cotainig the script

2. **Extract the Package**
   ```bash
   unzip pie-chart-cv-*.zip
   cd pie_chart_cv
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Google Credentials**
   - Place your `credentials.json` (OAuth 2.0 client) in the project root.

5. **Configure the Application**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env`:
     - Set your `USER_ID` (Gmail address to fetch emails for)
     - **Set `API_TOKEN`** (generate using `openssl rand -hex 32`)
     - Set `GOOGLE_SHEET_ID` (ID of your Google Sheet)
     - Set `GEMINI_API_KEY` (if using Gemini features)
     - Adjust other settings as needed

## Running the Script

1. **Start the API Server**
   ```bash
   python main.py
   ```
   - The server will run on `http://localhost:8080` by default.

2. **Authorize Gmail and Sheets Access**
   - Visit: [http://localhost:8080/authorize](http://localhost:8080/authorize)
   - Complete the Google OAuth flow in your browser.

3. **Fetch Emails**
   - Fetch emails with the required subject and attachments:
     ```bash
     curl -H "Authorization: Bearer your_api_token" \
          "http://localhost:8080/fetch-emails"
     ```
   - This saves emails to the local database and attachments to the local folder.

4. **Process Emails**
   - Extract MBTI and pie chart data from fetched emails:
     ```bash
     curl -H "Authorization: Bearer your_api_token" \
          "http://localhost:8080/process-emails"
     ```
   - Results are saved to the database.

5. **Save Results to Google Sheets**
   - Overwrite the Google Sheet with all results:
     ```bash
     curl -H "Authorization: Bearer your_api_token" \
          "http://localhost:8080/mbti-results"
     ```
    - Results are saved to Google Sheet.

## Periodic Sync (Recommended for Ongoing Use)

To automate fetching, processing, and syncing new results:

1. **Call the Sync Endpoint**
   ```bash
   curl -H "Authorization: Bearer your_api_token" \
        "http://localhost:8080/sync-emails"
   ```
   - This will:
     - Fetch new emails since the last processed date
     - Process them for MBTI and pie chart data
     - Append new results to your Google Sheet

2. **Automate with a Scheduler**
   - Use `cron`, Windows Task Scheduler, or a similar tool to call the sync endpoint periodically (e.g., every hour or day).

## Useful Endpoints

- `/authorize` - Start OAuth flow
- `/fetch-emails` - Fetch emails and attachments
- `/process-emails` - Process unprocessed emails
- `/mbti-results` - Get/save results (JSON, CSV, or Google Sheets)
- `/sync-emails` - Fetch, process, and sync in one step
- `/status` - Check OAuth status
- `/` - Check health

---

**For more details, see the full documentation and distribution guide.**