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

1. **Clone the repository**
   - Clone the repository and enter the newly created folder

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Google Credentials**
   - Place your `credentials.json` (OAuth 2.0 client) in the project root.

4. **Configure the Application**
   - Edit `config.json`:
     - Set your `user_id` (Gmail address to fetch emails for)
     - Set `google_sheet_id` (ID of your Google Sheet)
     - Set `gemini_api_key` (if using Gemini features)
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
     ```
     GET http://localhost:8080/fetch-emails
     ```
   - This saves emails to the local database and attachments to the local folder.

4. **Process Emails**
   - Extract MBTI and pie chart data from fetched emails:
     ```
     GET http://localhost:8080/process-emails
     ```
   - Results are saved to the database.

5. **Save Results to Google Sheets**
   - Overwrite the Google Sheet with all results:
     ```
     GET http://localhost:8080/mbti-results
     ```
    - Results are saved to Google Sheet.

## Periodic Sync (Recommended for Ongoing Use)

To automate fetching, processing, and syncing new results:

1. **Call the Sync Endpoint**
   ```
   GET http://localhost:8080/sync-emails
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
