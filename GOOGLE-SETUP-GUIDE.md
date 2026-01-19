# Google API Setup Guide

This guide will help you set up Google Cloud credentials to use with Pie Chart CV.

## What You'll Create

- Google Cloud Project
- OAuth 2.0 credentials
- Enable Gmail API and Google Sheets API

## Setup Steps

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Click **"Select a project"** at the top → **"NEW PROJECT"**
4. Enter project name (e.g., "Pie Chart CV")
5. Click **"CREATE"**

### Step 2: Enable Required APIs

1. In the left sidebar, go to **"APIs & Services"** → **"Library"**
2. Search for **"Gmail API"**
   - Click on it → Click **"ENABLE"**
3. Go back to Library and search for **"Google Sheets API"**
   - Click on it → Click **"ENABLE"**

### Step 3: Configure OAuth Consent Screen

1. Go to **"APIs & Services"** → **"OAuth consent screen"**
2. Select **"External"** (unless you have a Google Workspace account)
3. Click **"CREATE"**
4. Fill in required information:
   - **App name:** Pie Chart CV
   - **User support email:** Your email
   - **Developer contact:** Your email
5. Click **"SAVE AND CONTINUE"**
6. On "Scopes" page, click **"ADD OR REMOVE SCOPES"**
7. Add these scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/spreadsheets`
8. Click **"UPDATE"** → **"SAVE AND CONTINUE"**
9. On "Test users" page:
   - Click **"ADD USERS"**
   - Enter the Gmail addresses that will use the application
   - Click **"ADD"** → **"SAVE AND CONTINUE"**
10. Review and click **"BACK TO DASHBOARD"**

### Step 4: Create OAuth 2.0 Credentials

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"CREATE CREDENTIALS"** → **"OAuth client ID"**
3. Select application type: **"Web application"**
   - **Important:** Use "Web application" not "Desktop app" for proper OAuth flow
4. Enter name: "Pie Chart CV"
5. Under **"Authorized redirect URIs"**, click **"ADD URI"**
6. Add the callback URI: `http://localhost:8080/oauth2callback`
   - This is where Google will redirect after authentication
   - Port 8080 is the default application port
   - If you change the port or "oauth_redirect_uri" in `config.json`, update this URI accordingly
7. Click **"CREATE"**
8. A dialog appears with your client ID and secret
9. Click **"DOWNLOAD JSON"**
10. Save the file

**Important Notes:**
- The callback URI must match exactly the "oauth_redirect_uri" setting in your config.json
- For multiple instances on different ports, each needs its own OAuth client with matching redirect URI

### Step 5: Prepare credentials.json

1. Rename the downloaded file to `credentials.json`
2. Place it in your Pie Chart CV folder

## Important Notes

### Test Mode Limitations

Your app will be in "Testing" mode, which means:
- Only test users you added can use it
- Authorization expires after 7 days (you'll need to reauthorize)

**To remove the 7-day limitation:**
1. Go to OAuth consent screen
2. Click **"PUBLISH APP"**
3. Follow the verification process (if required)

### Multiple Users

Each user needs:
- To be added as a test user (Step 3.9) OR the app must be published
- Their own Google account authorization
- Their own `credentials.json` file (can be shared if all using same port)
  - **If running on different ports:** Each needs separate OAuth client with matching redirect URI

**Multiple Instances on Different Ports:**

If running multiple instances (e.g., instance1 on port 8080, instance2 on port 8081):

**Option 1: Separate OAuth Clients (Recommended)**
- Create separate OAuth client for each port
- Instance 1: redirect URI = `http://localhost:8080/oauth2callback`
- Instance 2: redirect URI = `http://localhost:8081/oauth2callback`
- Each instance gets its own `credentials.json`

**Option 2: Single OAuth Client with Multiple Redirect URIs**
- Add multiple redirect URIs to one OAuth client:
  - `http://localhost:8080/oauth2callback`
  - `http://localhost:8081/oauth2callback`
  - `http://localhost:8082/oauth2callback`
- Same `credentials.json` can be used for all instances

### Security

- Keep `credentials.json` private
- Don't share your client secret
- The application only accesses Gmail and Sheets
- You can revoke access anytime at [Google Account Security](https://myaccount.google.com/permissions)

## Troubleshooting

**"Access blocked: This app's request is invalid"**
- Make sure OAuth consent screen is configured
- Check that required scopes are added
- Verify the email is added as a test user

**"The application has been blocked"**
- Your app is in testing mode
- Add the user's email to test users in OAuth consent screen

**"Redirect URI mismatch" error**
- The redirect URI in Google Cloud Console must match exactly the "oauth_redirect_uri" setting
- Make sure there are no typos or extra spaces
- URI is case-sensitive

**Authorization expires after 7 days**
- This is normal for apps in testing mode
- Publish the app to remove this limitation
- Or simply reauthorize after 7 days
