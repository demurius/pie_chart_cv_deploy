from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import cv2
import numpy as np
import requests
from typing import List, Dict, Optional
import traceback
import os
from datetime import datetime, timezone
import base64
import json
import csv
import re
import warnings

# Suppress runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from googleapiclient.discovery import build
from google.auth.transport import requests as google_requests
from email.utils import parsedate_to_datetime
import asyncio
import socket
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

from database import (
    save_user_credentials,
    get_user_credentials,
    delete_user_credentials,
    save_email_with_attachments,
    get_emails_by_user,
    get_email_by_message_id,
    get_email_details_by_message_id,
    get_unprocessed_emails,
    get_most_recent_email_date,
    save_mbti_result,
    get_mbti_results,
    Email,
    Attachment,
)
from gemini_processor import process_single_image_with_gemini, calculate_tritype
from pie_chart_processor import analyze_pie_chart_with_contours, analyze_pie_chart_with_colors
from pathlib import Path

# Load configuration
try:
    from config_loader import get_config, init_config
    # Initialize configuration on startup
    app_config = init_config()
    is_valid, errors = app_config.validate()
    if not is_valid:
        print("[Config] WARNING: Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
except Exception as e:
    print(f"[Config] WARNING: Failed to load config_loader: {e}")
    print("[Config] Continuing with environment variables only")
    app_config = None

# Constants
MIN_REQUIRED_IMAGES = 2


# Force Python to use IPv4 only to avoid the IPv6 timeout
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = getaddrinfo_ipv4


def refresh_credentials(user_id: str, credentials: Credentials) -> Credentials:
    """
    Refresh expired credentials and save to database.

    Args:
        user_id: User identifier
        credentials: Current credentials object

    Returns:
        Refreshed credentials
    """
    try:
        print(f"[OAuth] Refreshing token for user_id: {user_id}")

        # Manual token refresh to avoid hanging
        refresh_data = {
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "refresh_token": credentials.refresh_token,
            "grant_type": "refresh_token",
        }

        response = requests.post(credentials.token_uri, data=refresh_data, timeout=10)
        response.raise_for_status()
        token_data = response.json()

        # Update credentials with new token
        credentials.token = token_data["access_token"]
        credentials.expiry = None  # Will be set on next use

        # Save refreshed token to database
        updated_creds = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }
        save_user_credentials(user_id, updated_creds)

        print(f"[OAuth] Token refreshed successfully for user_id: {user_id}")
        return credentials

    except Exception as e:
        print(f"[OAuth] Failed to refresh token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Token refresh failed. Please re-authorize at /authorize?user_id={user_id}",
        )


# Allow OAuth over HTTP for local development (REMOVE IN PRODUCTION)
# Check if we should set this from environment
if os.getenv("OAUTHLIB_INSECURE_TRANSPORT") is None:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

app = FastAPI(
    title="Pie Chart Analyzer API",
    description="API for analyzing pie charts from images",
    version="1.0.0",
)

# Gmail OAuth Configuration
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
    "openid",
    "email",
    "profile",
]
CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")

# Get redirect URI from config or environment
if app_config:
    REDIRECT_URI = app_config.get("oauth_redirect_uri", os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback"))
else:
    REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback")

# Temporary state storage for OAuth flow (only stores state during authorization)
# User credentials are stored in SQLite database with automatic token refresh
# Tokens are automatically refreshed when expired, so users stay authenticated indefinitely
temp_state_store = {}


def get_user_id_or_fallback(provided_user_id: Optional[str]) -> str:
    """
    Get user_id from parameter or fallback to config value.
    
    Args:
        provided_user_id: User ID provided as query parameter
        
    Returns:
        User ID to use for the request
        
    Raises:
        HTTPException: If user_id is not provided and not configured
    """
    # Check if provided_user_id is valid (not None and not empty/whitespace)
    if provided_user_id is not None and provided_user_id.strip():
        return provided_user_id.strip()
    
    # Error message for missing user_id
    error_msg = "user_id parameter is required or must be configured in config.json"
    
    # Fall back to config
    if app_config is None or app_config.user_id is None:
        raise HTTPException(status_code=400, detail=error_msg)
    
    config_user_id = app_config.user_id.strip()
    if not config_user_id:
        raise HTTPException(status_code=400, detail=error_msg)
    
    return config_user_id


class ImageRequest(BaseModel):
    id: str
    url: Optional[HttpUrl] = None
    base64_image: Optional[str] = None
    name: str
    debug: bool = False


class SegmentData(BaseModel):
    name: str
    # color: List[int]
    percentage: float


class EmailData(BaseModel):
    id: str
    subject: str
    sender: str
    snippet: str
    date: str
    body: Optional[str] = None


class EmailListResponse(BaseModel):
    emails: List[EmailData]
    total: int

class AnalysisResponse(BaseModel):
    success: bool
    id: str = None
    name: str = None
    date: str = None
    segment_1: Optional[float] = None
    segment_2: Optional[float] = None
    segment_3: Optional[float] = None
    segment_4: Optional[float] = None
    segment_5: Optional[float] = None
    segment_6: Optional[float] = None
    segment_7: Optional[float] = None
    segment_8: Optional[float] = None
    segment_9: Optional[float] = None


@app.get("/authorize")
async def authorize(user_id: Optional[str] = None):
    """
    Start OAuth flow for a specific user.

    Parameters:
    - user_id: Unique identifier for the user (optional, uses config.user_id if not provided)
    """
    user_id = get_user_id_or_fallback(user_id)
    
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )

    # Generate the authorization URL
    auth_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )

    # Store state with user_id for callback
    temp_state_store[state] = {"user_id": user_id}
    print(f"[OAuth] Starting authorization for user_id: {user_id}")

    return RedirectResponse(auth_url)


@app.get("/oauth2callback")
async def oauth2callback(request: Request):
    # 1. Extract state and code from the URL
    state = request.query_params.get("state")
    code = request.query_params.get("code")

    print(
        f"[OAuth] Callback received - state: {state[:20] if state else None}, code: {bool(code)}"
    )

    if not state or state not in temp_state_store:
        raise HTTPException(status_code=400, detail="State mismatch/Possible CSRF")

    user_id = temp_state_store[state]["user_id"]
    print(f"[OAuth] Processing callback for user_id: {user_id}")

    # 2. Read client secrets to get client_id and client_secret
    try:
        with open(CLIENT_SECRETS_FILE, "r") as f:
            client_config = json.load(f)
            client_id = client_config["web"]["client_id"]
            client_secret = client_config["web"]["client_secret"]
    except Exception as e:
        print(f"[OAuth] Failed to read credentials: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read OAuth credentials")

    # 3. Manually exchange the authorization code for tokens
    # This bypasses the Flow.fetch_token() which seems to hang
    try:
        print("[OAuth] Starting manual token exchange...")

        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }

        # Make the request with explicit timeout
        async def exchange_token():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: requests.post(
                    token_url, data=token_data, timeout=10  # 10 second HTTP timeout
                ),
            )

        response = await asyncio.wait_for(exchange_token(), timeout=15.0)

        if response.status_code != 200:
            print(f"[OAuth] Token exchange failed with status {response.status_code}")
            print(f"[OAuth] Response: {response.text}")
            raise HTTPException(
                status_code=500, detail=f"Token exchange failed: {response.text}"
            )

        token_response = response.json()
        print("[OAuth] Token exchange successful")

        # Build credentials object
        creds = Credentials(
            token=token_response["access_token"],
            refresh_token=token_response.get("refresh_token"),
            token_uri=token_url,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )

    except asyncio.TimeoutError:
        print("[OAuth] Token exchange timed out after 15 seconds")
        raise HTTPException(
            status_code=500,
            detail="Token exchange timed out. Please check your internet connection.",
        )
    except requests.exceptions.Timeout:
        print("[OAuth] HTTP request timed out")
        raise HTTPException(
            status_code=500, detail="Network timeout connecting to Google"
        )
    except Exception as e:
        print(f"[OAuth] Token exchange failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {str(e)}")

    # 4. Store the tokens for this user
    user_tokens = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

    # Save to database
    save_user_credentials(user_id, user_tokens)

    # Clean up the state entry from temp storage

    del temp_state_store[state]

    print(f"[OAuth] Stored credentials for user_id: {user_id}")

    return {
        "status": "success",
        "user_id": user_id,
        "message": "Authentication successful. You can now close this window.",
    }


@app.get("/fetch-emails")
async def fetch_emails(
    user_id: Optional[str] = None,
    start_page: int = 1,
    start_date: Optional[str] = "auto"
):
    """
    Fetch ALL emails with specific subject and at least 2 image attachments, save to database.
    
    Query Parameters:
    - user_id: User identifier (optional, uses config.user_id if not provided)
    - start_page: Page number to start fetching from (default: 1, 1-based indexing)
    - start_date: Start date in YYYY/MM/DD format, or "auto" to use most recent email date (default: "auto")
    
    Note: This endpoint will fetch ALL matching emails using pagination starting from the specified page.
    If start_date is "auto", it will automatically use the most recent email date in database.
    Settings like subject_filter and save_attachments are read from config.json.
    
    Returns:
    - List of saved emails with attachment count
    """
    # Get settings from config
    if app_config is None:
        raise HTTPException(
            status_code=500,
            detail="Configuration not loaded. Please ensure config.json exists."
        )
    
    user_id = get_user_id_or_fallback(user_id)
    
    subject_filter = app_config.subject_filter
    save_attachments = app_config.save_attachments
    start_page = max(1, start_page)
    
    # If start_date is "auto", get the most recent email date
    if start_date and start_date.lower() == "auto":
        most_recent_date = get_most_recent_email_date(user_id)
        if most_recent_date:
            # Convert datetime to Unix timestamp for precise filtering
            timestamp = int(most_recent_date.timestamp())
            start_date = str(timestamp)
            print(f"[Gmail] Auto mode enabled: using most recent email timestamp as start_date: {start_date} (from {most_recent_date})")
        else:
            print(f"[Gmail] Auto mode enabled but no existing emails found for user {user_id}. Will fetch all emails.")
            start_date = None
    
    # Create attachments directory - use configurable path for Railway volume persistence
    attachments_base = os.getenv("ATTACHMENTS_PATH", "./attachments")
    attachments_dir = Path(attachments_base) / user_id
    attachments_dir.mkdir(parents=True, exist_ok=True)

    # Get credentials from database
    creds_data = get_user_credentials(user_id)

    if not creds_data:
        raise HTTPException(
            status_code=401,
            detail=f"User '{user_id}' not authenticated. Please complete OAuth flow first at /authorize?user_id={user_id}",
        )

    try:
        # Reconstruct credentials
        print(
            f"[Gmail] Fetching emails with subject '{subject_filter}' for user_id: {user_id}"
        )
        credentials = Credentials(
            token=creds_data["token"],
            refresh_token=creds_data.get("refresh_token"),
            token_uri=creds_data["token_uri"],
            client_id=creds_data["client_id"],
            client_secret=creds_data["client_secret"],
            scopes=creds_data["scopes"],
        )

        # Always try to refresh token if refresh_token exists (tokens typically expire)
        if credentials.refresh_token:
            try:
                credentials = refresh_credentials(user_id, credentials)
            except Exception as e:
                print(
                    f"[OAuth] Token refresh failed, will try with existing token: {str(e)}"
                )

        # Use direct HTTP requests instead of building service (better macOS compatibility)
        access_token = credentials.token
        headers = {"Authorization": f"Bearer {access_token}"}

        print(f"[Gmail] Using token: {access_token[:20]}...")

        # Create session with proper SSL verification
        session = requests.Session()
        # session.verify = True is the default, ensuring secure HTTPS connections

        # Search for emails with subject filter
        # Use quotes for exact phrase matching to ensure entire subject is searched
        if ' ' in subject_filter:
            # If subject contains spaces, wrap in quotes for exact phrase search
            query = f'subject:"{subject_filter}"'
        else:
            # Single word can use basic subject: search
            query = f"subject:{subject_filter}"
        
        # Add date filter if start_date is provided
        if start_date:
            if start_date.isdigit():
                query += f" after:{start_date}"
                print(f"[Gmail] Added timestamp filter: after:{start_date}")                
            else:
                # Gmail date format: after:YYYY/MM/DD
                try:
                    # Parse date format YYYY/MM/DD
                    date_obj = datetime.strptime(start_date, "%Y/%m/%d")
                    gmail_date = date_obj.strftime("%Y/%m/%d")
                    query += f" after:{gmail_date}"
                    print(f"[Gmail] Added date filter: after:{gmail_date} (from {start_date})")
                except ValueError as e:
                    print(f"[Gmail] Invalid start_date format: {start_date}. Expected YYYY/MM/DD. Ignoring date filter. Error: {e}")

        # Direct HTTP request to Gmail API with pagination to fetch ALL emails
        print(f"[Gmail] Searching with query: {query}")
        search_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
        
        all_messages = []
        next_page_token = None
        page_count = 0
        
        # If starting from a page > 1, we need to skip to that page first
        if start_page > 1:
            print(f"[Gmail] Skipping to start page {start_page}...")
            # We'll need to fetch pages 1 through (start_page - 1) just to get the page tokens
            # but we won't process the messages from those pages
            temp_page_count = 0
            while temp_page_count < start_page - 1:
                temp_page_count += 1
                print(f"[Gmail] Skipping page {temp_page_count} to reach start page...")
                
                params = {"q": query, "maxResults": 500}
                if next_page_token:
                    params["pageToken"] = next_page_token

                skip_response = session.get(
                    search_url, headers=headers, params=params, timeout=30
                )
                
                if skip_response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Gmail API returned 401 Unauthorized. Token may be invalid or have wrong scopes. Please re-authorize at /authorize?user_id={user_id}",
                    )
                
                skip_response.raise_for_status()
                skip_results = skip_response.json()
                
                next_page_token = skip_results.get("nextPageToken")
                if not next_page_token and temp_page_count < start_page - 1:
                    print(f"[Gmail] No more pages available. Cannot reach start page {start_page}.")
                    break
            
            page_count = start_page - 1
            print(f"[Gmail] Ready to start fetching from page {start_page}...")
        
        while True:
            page_count += 1
            print(f"[Gmail] Fetching page {page_count}...")                    
            
            # Prepare parameters for this page
            params = {"q": query, "maxResults": 500}  # Use maximum allowed per request
            if next_page_token:
                params["pageToken"] = next_page_token

            search_response = session.get(
                search_url, headers=headers, params=params, timeout=30
            )
            print(f"[Gmail] Page {page_count} request completed with status: {search_response.status_code}")

            if search_response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail=f"Gmail API returned 401 Unauthorized. Token may be invalid or have wrong scopes. Please re-authorize at /authorize?user_id={user_id}",
                )

            search_response.raise_for_status()
            page_results = search_response.json()
            page_messages = page_results.get("messages", [])
            
            if page_messages:
                all_messages.extend(page_messages)
                print(f"[Gmail] Page {page_count}: Found {len(page_messages)} messages (Total so far: {len(all_messages)})")
            
            # Check if there are more pages
            next_page_token = page_results.get("nextPageToken")
            if not next_page_token:
                break
            
            # Safety limit to prevent infinite loops (adjust as needed)
            if page_count >= 100:  # Max 50,000 emails (100 pages * 500 per page)
                print(f"[Gmail] Reached maximum page limit ({page_count}), stopping pagination")
                break

        print(f"[Gmail] Pagination complete. Found {len(all_messages)} total messages matching subject")
        messages = all_messages
        print(f"[Gmail] Found {len(messages)} messages matching subject")

        saved_emails = []

        for idx, message in enumerate(messages):

            # Refresh token every 1000 emails to prevent expiration during long operations
            if (idx + 1) % 1000 == 0 and credentials.refresh_token:
                try:
                    credentials = refresh_credentials(user_id, credentials)
                    access_token = credentials.token
                    headers = {"Authorization": f"Bearer {access_token}"}
                    session.headers.update(headers)
                    print(f"[Gmail] Token refreshed after processing {idx + 1} emails")
                except Exception as e:
                    print(f"[Gmail] Token refresh failed after {idx + 1} emails: {str(e)}")
                    
            print(f"[Gmail] Processing message {idx + 1}/{len(messages)}...")

            # Fetch full message with direct HTTP request
            message_url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{message["id"]}'
            message_params = {"format": "full"}

            msg_response = session.get(
                message_url, headers=headers, params=message_params, timeout=10
            )
            msg_response.raise_for_status()
            msg = msg_response.json()

            # Extract thread ID
            thread_id = msg.get("threadId")

            # Extract headers
            msg_headers = msg["payload"]["headers"]
            subject = next(
                (h["value"] for h in msg_headers if h["name"].lower() == "subject"),
                "No Subject",
            )
            sender_raw = next(
                (h["value"] for h in msg_headers if h["name"].lower() == "from"),
                "Unknown",
            )
            recipient = next(
                (h["value"] for h in msg_headers if h["name"].lower() == "to"),
                "Unknown",
            )
            date_str = next(
                (h["value"] for h in msg_headers if h["name"].lower() == "date"), ""
            )

            # Extract sender name and email
            if "<" in sender_raw and ">" in sender_raw:
                sender_name = sender_raw.split("<")[0].strip().strip('"')
                sender_email = sender_raw.split("<")[-1].strip(">").strip()
            else:
                sender_name = sender_raw.strip()
                sender_email = sender_raw.strip()

            # Skip if sender email is the user_id (sent by user themselves)
            if sender_email.lower() == user_id.lower():
                print(
                    f"[Gmail] Email {msg['id']} sent by user ({sender_email}), skipping"
                )
                continue

            # Parse date
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(date_str) if date_str else None
                if date:
                    date = date.astimezone(timezone.utc)
            except:
                date = None

            # Find image attachments
            image_attachments = []

            def find_attachments(parts):
                for part in parts:
                    if "parts" in part:
                        find_attachments(part["parts"])
                    else:
                        filename = part.get("filename", "")
                        mime_type = part.get("mimeType", "")

                        # Check if it's an image
                        if filename and mime_type.startswith("image/"):
                            body = part.get("body", {})
                            attachment_id = body.get("attachmentId")
                            size = body.get("size", 0)

                            image_attachments.append(
                                {
                                    "filename": filename,
                                    "content_type": mime_type,
                                    "size_bytes": size,
                                    "attachment_id": attachment_id,
                                }
                            )

            if "parts" in msg["payload"]:
                find_attachments(msg["payload"]["parts"])

            # Check if email has at least 2 image attachments
            if len(image_attachments) < 2:
                print(
                    f"[Gmail] Email {msg['id']} has only {len(image_attachments)} image(s), skipping"
                )
                continue

            # Check if email already exists in database
            existing_email_id = get_email_by_message_id(msg["id"])
            if existing_email_id:
                print(
                    f"[Gmail] Email {msg['id']} already exists in database (ID: {existing_email_id}), skipping"
                )
                continue

            print(f"[Gmail] Email has {len(image_attachments)} images, saving...")

            # Download attachments if requested
            attachment_data = []
            for att in image_attachments:
                file_path = None

                if save_attachments and att["attachment_id"]:
                    try:
                        # Download attachment with direct HTTP request
                        attachment_url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg["id"]}/attachments/{att["attachment_id"]}'

                        att_response = session.get(
                            attachment_url, headers=headers, timeout=15
                        )
                        att_response.raise_for_status()
                        attachment = att_response.json()

                        file_data = base64.urlsafe_b64decode(attachment["data"])

                        # Save to file
                        safe_filename = "".join(
                            c
                            for c in att["filename"]
                            if c.isalnum() or c in (" ", ".", "_", "-")
                        ).rstrip()
                        
                        # Prevent overwriting by adding counter if file exists
                        base_file_path = attachments_dir / f"{msg['id']}_{safe_filename}"
                        file_path = base_file_path
                        counter = 1
                        
                        while file_path.exists():
                            # Split filename and extension
                            name_parts = safe_filename.rsplit('.', 1)
                            if len(name_parts) == 2:
                                name, ext = name_parts
                                new_filename = f"{msg['id']}_{name}_{counter}.{ext}"
                            else:
                                new_filename = f"{msg['id']}_{safe_filename}_{counter}"
                            
                            file_path = attachments_dir / new_filename
                            counter += 1
                            
                            # Safety limit to prevent infinite loops
                            if counter > 1000:
                                break

                        with open(file_path, "wb") as f:
                            f.write(file_data)

                        # Store only filename without path
                        file_path = file_path.name
                        print(f"[Gmail] Saved attachment: {file_path}")

                    except Exception as e:
                        print(f"[Gmail] Failed to download attachment: {str(e)}")

                attachment_data.append(
                    {
                        "filename": att["filename"],
                        "content_type": att["content_type"],
                        "size_bytes": att["size_bytes"],
                        "attachment_id": att["attachment_id"],
                        "file_path": file_path,
                    }
                )

            # Get email body
            body = ""
            if "parts" in msg["payload"]:
                for part in msg["payload"]["parts"]:
                    if part["mimeType"] == "text/plain" and "data" in part.get(
                        "body", {}
                    ):
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode(
                            "utf-8", errors="ignore"
                        )
                        break
            elif "body" in msg["payload"] and "data" in msg["payload"]["body"]:
                body = base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode(
                    "utf-8", errors="ignore"
                )

            # Save to database
            email_data = {
                "message_id": msg["id"],
                "thread_id": thread_id,
                "subject": subject,
                "sender": sender_name,
                "sender_email": sender_email,
                "date": date,
            }

            email_id = save_email_with_attachments(user_id, email_data, attachment_data)

            if email_id:
                saved_emails.append(
                    {
                        "email_id": email_id,
                        "message_id": msg["id"],
                        "subject": subject,
                        "image_count": len(image_attachments),
                        "saved_attachments": save_attachments,
                    }
                )

        print(f"[Gmail] Saved {len(saved_emails)} emails to database")

        return {
            "status": "success",
            "total_found": len(messages),
            "saved_count": len(saved_emails),
        }

    except asyncio.TimeoutError:
        print("[Gmail] Request timed out")
        raise HTTPException(
            status_code=500, detail="Gmail API request timed out. Please try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"[Gmail] Error fetching emails: {error_msg}")
        print(traceback.format_exc())

        # Check if it's a scope/permission error
        if (
            "insufficient" in error_msg.lower()
            or "permission" in error_msg.lower()
            or "scope" in error_msg.lower()
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Please re-authorize at /authorize?user_id={user_id} to grant required scopes.",
            )

        raise HTTPException(
            status_code=500, detail=f"Failed to fetch emails: {error_msg}"
        )


def download_image(url: str) -> np.ndarray:
    """Download image from URL and convert to OpenCV format."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process image: {str(e)}"
        )


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image format."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to decode base64 image: {str(e)}"
        )


def save_to_csv(image_name, result_status, percentages, csv_file="results.csv"):
    while len(percentages) < 9:
        percentages.append(None)

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        fieldnames = ["Image Name", "Date", "Result"] + [f"{i+1}" for i in range(9)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row_data = {
            "Image Name": image_name,
            "Date": current_date,
            "Result": result_status,
        }

        for i in range(9):
            row_data[f"{i+1}"] = (
                f"{percentages[i]:.2f}" if percentages[i] is not None else ""
            )

        writer.writerow(row_data)

    print(f"\nResults saved to '{csv_file}'")

    
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/config")
async def check_config():
    """
    Check configuration status and validation.
    
    Returns configuration status without exposing sensitive values.
    """
    if app_config is None:
        return {
            "status": "not_loaded",
            "message": "Configuration module not available",
            "using_env_vars": True
        }
    
    is_valid, errors = app_config.validate()
    
    return {
        "status": "valid" if is_valid else "invalid",
        "user_id_configured": bool(app_config.user_id),
        "gemini_api_key_configured": bool(app_config.gemini_api_key),
        "google_sheet_id_configured": bool(app_config.google_sheet_id),
        "subject_filter": app_config.subject_filter,
        "max_workers": app_config.max_workers,
        "skip_mbti": app_config.skip_mbti,
        "debug_mode": app_config.debug_mode,
        "auto_sync_to_sheet": app_config.auto_sync_to_sheet,
        "errors": errors if not is_valid else []
    }


@app.get("/status")
async def check_status(user_id: Optional[str] = None):
    """
    Check OAuth status for a user.

    Parameters:
    - user_id: User identifier to check (optional, uses config.user_id if not provided)

    Returns:
    - Authorization status
    """
    user_id = get_user_id_or_fallback(user_id)
    
    creds_data = get_user_credentials(user_id)

    if creds_data:
        return {
            "authorized": True,
            "user_id": user_id,
            "has_refresh_token": bool(creds_data.get("refresh_token")),
        }
    else:
        return {
            "authorized": False,
            "user_id": user_id,
            "message": f"User not authorized. Please visit /authorize?user_id={user_id}",
        }


@app.delete("/logout")
async def logout_user(user_id: Optional[str] = None):
    """
    Remove stored credentials for a user.

    Parameters:
    - user_id: User identifier (optional, uses config.user_id if not provided)

    Returns:
    - Success status
    """
    user_id = get_user_id_or_fallback(user_id)
    
    success = delete_user_credentials(user_id)

    if success:
        return {
            "status": "success",
            "message": f"Credentials removed for user '{user_id}'",
        }
    else:
        raise HTTPException(
            status_code=404, detail=f"No credentials found for user '{user_id}'"
        )


@app.get("/test-image")
async def get_test_image():
    """
    Serve a test pie chart image for download.

    Returns a static pie chart image that can be used to test the /measure endpoint.
    """
    test_images = ["image1.png"]

    for image_name in test_images:
        if os.path.exists(image_name):
            return FileResponse(
                image_name, media_type="image/png", filename="test_piechart.png"
            )

    raise HTTPException(
        status_code=404,
        detail="No test image available. Please place a pie chart image in the server directory.",
    )


def process_single_email(email: Email, gemini_api_key: str, skip_mbti: bool = False, debug_mode: bool = False) -> Dict:
    """
    Process a single email: extract personality data with Gemini and pie chart with OpenCV.
    
    Args:
        email: Email object from database with attachments
        gemini_api_key: Gemini API key
        skip_mbti: Skip Gemini extraction (default: False)
        debug_mode: Enable debug mode (default: False)
        
    Returns:
        Dictionary with processing results
    """
    try:
        print(f"[Process] Processing email {email['message_id']}")
        
        # Get image attachments
        image_attachments = [att for att in email['attachments'] if att['content_type'] and att['content_type'].startswith('image/')]
        
        if len(image_attachments) < MIN_REQUIRED_IMAGES:
            print(f"[Process] Email has only {len(image_attachments)} image(s), need at least {MIN_REQUIRED_IMAGES}")
            return {
                'email_id': email['id'],
                'message_id': email['message_id'],
                'success': 'error',
                'error_message': f'Only {len(image_attachments)} image attachment(s) found, need at least {MIN_REQUIRED_IMAGES}'
            }
        
        print(f"[Process] Found {len(image_attachments)} image attachments")
        
        # Load all images as base64
        images_data = []
        for idx, att in enumerate(image_attachments):
            if att['file_path'] and os.path.exists(att['file_path']):
                with open(att['file_path'], 'rb') as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    images_data.append({
                        'index': idx,
                        'base64': image_base64,
                        'filename': att['filename'],
                        'attachment': att
                    })
            else:
                print(f"[Process] Warning: Attachment file not found: {att['file_path']}")
        
        if len(images_data) < MIN_REQUIRED_IMAGES:
            return {
                'email_id': email['id'],
                'message_id': email['message_id'],
                'success': 'error',
                'error_message': 'Image files not found on disk'
            }
        
        # Step 1: Try Gemini on each image until we find personality data
        personality_data = None
        personality_image_index = None
        
        if not skip_mbti:
            first_zero_result = None
            first_zero_index = None
            for img_data in images_data:
                print(f"[Process] Trying Gemini on image {img_data['index']}: {img_data['filename']}")
                gemini_result = process_single_image_with_gemini(
                    img_data['base64'],
                    gemini_api_key,
                    img_data['index']
                )
                # Check for critical errors that should stop processing
                error_type = gemini_result.get('error')
                if error_type in ['rate_limit_exceeded', 'invalid_api_key', 'permission_denied', 'service_unavailable']:
                    error_messages = {
                        'rate_limit_exceeded': 'Gemini API rate limit exceeded',
                        'invalid_api_key': 'Invalid Gemini API key',
                        'permission_denied': 'Gemini API key lacks required permissions',
                        'service_unavailable': 'Gemini service temporarily unavailable'
                    }
                    print(f"[Process] Gemini critical error: {error_type}")
                    return {
                        'email_id': email['id'],
                        'message_id': email['message_id'],
                        'success': 'gemini_failed',
                        'error_message': gemini_result.get('error_message', error_messages.get(error_type, 'Gemini API error'))
                    }
                # Safety blocked - not critical, continue to next image
                if error_type == 'safety_blocked':
                    print(f"[Process] Image {img_data['index']} blocked by safety filters, trying next image")
                    continue
                if gemini_result['success']:
                    pd = gemini_result['personalityData']
                    # Check if all percentage values are zero
                    has_zero = any(
                        int(pd.get(k, 0)) == 0 for k in ['energy', 'mind', 'nature', 'tactics', 'identity']
                    )
                    if has_zero:
                        if first_zero_result is None:
                            first_zero_result = pd
                            first_zero_index = img_data['index']
                        print(f"[Process] Gemini result for image {img_data['index']} has zeros, trying next image...")
                        continue
                    personality_data = pd
                    personality_image_index = img_data['index']
                    print(f"[Process] Found non-zero personality data in image {personality_image_index}")
                    break
            else:
                # If loop completes with no break (no non-zero found)
                if first_zero_result is not None:
                    personality_data = first_zero_result
                    personality_image_index = first_zero_index
                    print(f"[Process] No non-zero Gemini result found, using first result with zeros from image {personality_image_index}")
                else:
                    print("[Process] Gemini failed to find personality data, will continue with pie chart extraction")
                    # Use default values for personality data
                    personality_data = {
                        'energy': 0, 'energy_type': '-',
                        'mind': 0, 'mind_type': '-',
                        'nature': 0, 'nature_type': '-',
                        'tactics': 0, 'tactics_type': '-',
                        'identity': 0, 'identity_type': '-'
                    }
                    personality_image_index = -1
        else:
            print("[Process] Skipping Gemini extraction (skip_mbti=True)")
            personality_data = {
                'energy': 0, 'energy_type': '-',
                'mind': 0, 'mind_type': '-',
                'nature': 0, 'nature_type': '-',
                'tactics': 0, 'tactics_type': '-',
                'identity': 0, 'identity_type': '-'
            }
            personality_image_index = -1
        
        # Step 2: Try OpenCV on remaining images until we find a pie chart
        pie_chart_result = None
        pie_chart_image_index = None
        
        for img_data in images_data:
            # Skip the image that had personality data
            if img_data['index'] == personality_image_index:
                continue
            
            print(f"[Process] Trying OpenCV on image {img_data['index']}: {img_data['filename']}")
            
            try:
                # Decode base64 to OpenCV image
                image_bytes = base64.b64decode(img_data['base64'])
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"[Process] Failed to decode image {img_data['index']}")
                    continue
                
                # Analyze pie chart using method with colors
                result = analyze_pie_chart_with_colors(
                    img,
                    image_id=email['message_id'],
                    image_name=img_data['filename'],
                    debug=debug_mode
                )
                
                if result.get('success') is False:
                    print(f"[Process] Error analyzing image {img_data['index']} using colors method, trying contours method...")
                    result = analyze_pie_chart_with_contours(
                        img,
                        image_id=email['message_id'],
                        image_name=img_data['filename'],
                        debug=debug_mode
                    )                
                
                if result.get('success'):
                    pie_chart_result = result
                    pie_chart_image_index = img_data['index']
                    print(f"[Process] Found pie chart in image {pie_chart_image_index}")
                    break
                else:
                    print(f"[Process] No pie chart found in image {img_data['index']}")
            
            except Exception as e:
                print(f"[Process] Error analyzing image {img_data['index']}: {str(e)}")
                continue
        
        if not pie_chart_result:
            print("[Process] OpenCV failed to find pie chart in remaining images")
            return {
                'email_id': email['id'],
                'message_id': email['message_id'],
                'success': 'api_failed',
                'error_message': 'No pie chart found',
                # Save personality data even if pie chart failed
                'mbti_type': f"{personality_data['energy_type']}{personality_data['mind_type']}{personality_data['nature_type']}{personality_data['tactics_type']}-{personality_data['identity_type']}",
                'energy': int(personality_data['energy']),
                'energy_type': personality_data['energy_type'],
                'mind': int(personality_data['mind']),
                'mind_type': personality_data['mind_type'],
                'nature': int(personality_data['nature']),
                'nature_type': personality_data['nature_type'],
                'tactics': int(personality_data['tactics']),
                'tactics_type': personality_data['tactics_type'],
                'identity': int(personality_data['identity']),
                'identity_type': personality_data['identity_type'],
                'personality_image_index': personality_image_index
            }
        
        # Step 3: Calculate tritype from pie chart results
        tritype_with_8, tritype_without_8 = calculate_tritype(pie_chart_result)
        
        # Determine success status based on what was extracted
        # If personality_image_index is -1 and not skip_mbti, it means personality extraction failed
        has_personality = personality_image_index != -1 or skip_mbti
        
        if not has_personality:
            success_status = 'gemini_failed'
            error_message = 'Personality data extraction failed, pie chart data only'
        else:
            success_status = 'success'
            error_message = None
        
        # Build complete result
        result = {
            'email_id': email['id'],
            'message_id': email['message_id'],
            'success': success_status,
            'error_message': error_message,
            # MBTI data
            'mbti_type': f"{personality_data['energy_type']}{personality_data['mind_type']}{personality_data['nature_type']}{personality_data['tactics_type']}-{personality_data['identity_type']}",
            'energy': int(personality_data['energy']),
            'energy_type': personality_data['energy_type'],
            'mind': int(personality_data['mind']),
            'mind_type': personality_data['mind_type'],
            'nature': int(personality_data['nature']),
            'nature_type': personality_data['nature_type'],
            'tactics': int(personality_data['tactics']),
            'tactics_type': personality_data['tactics_type'],
            'identity': int(personality_data['identity']),
            'identity_type': personality_data['identity_type'],
            'personality_image_index': personality_image_index,
            # Enneagram data
            'segment_1': pie_chart_result.get('segment_1', 0.0),
            'segment_2': pie_chart_result.get('segment_2', 0.0),
            'segment_3': pie_chart_result.get('segment_3', 0.0),
            'segment_4': pie_chart_result.get('segment_4', 0.0),
            'segment_5': pie_chart_result.get('segment_5', 0.0),
            'segment_6': pie_chart_result.get('segment_6', 0.0),
            'segment_7': pie_chart_result.get('segment_7', 0.0),
            'segment_8': pie_chart_result.get('segment_8', 0.0),
            'segment_9': pie_chart_result.get('segment_9', 0.0),
            # Tritype
            'tritype': tritype_with_8,
            'tritype_no_8': tritype_without_8
        }
        
        print(f"[Process] Successfully processed email {email['message_id']}")
        return result
    
    except Exception as e:
        print(f"[Process] Error processing email {email['message_id']}: {str(e)}")
        traceback.print_exc()
        return {
            'email_id': email['id'],
            'message_id': email['message_id'],
            'success': 'error',
            'error_message': str(e)
        }



def format_result_for_sheet(result_data: Dict, email_data: Dict) -> Dict[str, str]:
    """
    Format MBTI result and email data into a row dictionary for Google Sheets.
    
    Args:
        result_data: Dictionary with MBTI result data (can be from MBTIResult object or process result dict)
        email_data: Dictionary with email data (from Email object or get_email_details)
        
    Returns:
        Dictionary mapping column names to formatted values
    """
    # Handle both dict and object attribute access
    def get_value(obj, key, default=''):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    
    # Build Gmail URL
    message_id = get_value(result_data, 'message_id') or get_value(email_data, 'message_id')
    gmail_url = f"https://mail.google.com/mail/u/0/#inbox/{message_id}" if message_id else ""
    
    # Format % Cert field
    cert_parts = []
    cert_valid = True
    for aspect, aspect_type in [('energy', 'energy_type'), ('mind', 'mind_type'), 
                                  ('nature', 'nature_type'), ('tactics', 'tactics_type'), 
                                  ('identity', 'identity_type')]:
        value = get_value(result_data, aspect, 0)
        type_val = get_value(result_data, aspect_type, aspect[0].upper())
        if type_val is not None and value is not None:
            cert_parts.append(f"{type_val} - {value}%")
        else:
            cert_parts.append(f"{aspect[0].upper()} - ?")
            cert_valid = False
    if not cert_valid:
        cert_field = " / ".join(["- - 0%"] * 5)
    else:
        cert_field = " / ".join(cert_parts)
    
    # Format Enneagram field - each segment on a new line
    ennegram_parts = []
    for i in range(1, 10):
        segment_val = get_value(result_data, f'segment_{i}')
        if segment_val is not None:
            ennegram_parts.append(f"{i}: {segment_val:.2f}")
        else:
            ennegram_parts.append(f"{i}: 0.00")
    ennegram_field = "\n".join(ennegram_parts) if ennegram_parts else ""
    
    # Extract role from subject
    role = ''
    subject = get_value(email_data, 'subject', '')
    if subject and isinstance(subject, str):
        role_match = re.search(r'Your recent application for (.+)', subject)
        if role_match:
            role = role_match.group(1).strip()
    
    # Format date
    email_date = get_value(email_data, 'date')
    date_str = email_date.strftime('%Y-%m-%d') if (email_date and hasattr(email_date, 'strftime')) else ''
    
    mbti_type = get_value(result_data, 'mbti_type', '')
    if mbti_type is None:
        mbti_type = '------'
    
    # Build row data
    return {
        'ID': message_id or '',
        'Email': get_value(email_data, 'sender_email', ''),
        'Name': get_value(email_data, 'sender', ''),
        'Subject': subject,
        'Role': role,
        'Date': date_str,
        'Link to Email': f'=HYPERLINK("{gmail_url}", "Link")' if gmail_url else '',
        '% Cert': cert_field,
        'MBTI': mbti_type,
        'Ennegram': ennegram_field,
        'Tritype': get_value(result_data, 'tritype', ''),
        'Tritype (no 8)': get_value(result_data, 'tritype_no_8', '')
    }


def get_sheet_column_map(service, spreadsheet_id: str):
    try:
        # Request the spreadsheet metadata to see where the data actually starts
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1:Z1',
            # This ensures we get a predictable structure
            majorDimension='ROWS' 
        ).execute()
        
        # The 'range' returned by the API tells us where the data actually began
        # If A1 was empty, 'range' might be 'Sheet1!B1:C1'
        returned_range = result.get('range', '')
        # Simple parser to find the starting column letter
        import re
        match = re.search(r'!([A-Z]+)', returned_range)
        start_col_letter = match.group(1) if match else 'A'
        
        # Convert start letter (e.g., 'B') to 0-based index (e.g., 1)
        def col_to_idx(col):
            idx = 0
            for char in col:
                idx = idx * 26 + (ord(char.upper()) - ord('A') + 1)
            return idx - 1
            
        start_index = col_to_idx(start_col_letter)
        
        values = result.get('values', [[]])[0]
        
        column_map = {}
        for i, header in enumerate(values):
            if header:
                # Add the start_index to offset the mapping correctly
                column_map[str(header).strip()] = i + start_index
        
        return column_map
    except Exception as e:
        print(f"[Sheets] Error: {str(e)}")
        return {}


def ensure_sheet_has_header(service, spreadsheet_id: str):
    """
    Check if sheet is empty and add header row if needed.
    Header will be formatted with gray background and bold text.
    
    Args:
        service: Google Sheets API service
        spreadsheet_id: ID of the spreadsheet
        
    Returns:
        True if header exists or was created, False on error
    """
    try:
        # Check if sheet has any data
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1:Z1'
        ).execute()
        
        values = result.get('values', [])
        
        # If first row is empty, add header
        if not values or not values[0]:
            print("[Sheets] Sheet is empty, adding header row")
            
            # Define header columns
            headers = ['ID', 'Email', 'Name', 'Subject', 'Role', 'Date', 'Link to Email', '% Cert', 'MBTI', 'Ennegram', 'Tritype', 'Tritype (no 8)']
            
            # Add header row
            body = {'values': [headers]}
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range='Sheet1!A1',
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            # Format header: gray background and bold text
            # Get sheet ID (default is 0 for first sheet)
            sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            sheet_id = sheet_metadata['sheets'][0]['properties']['sheetId']
            
            requests = [
                {
                    'repeatCell': {
                        'range': {
                            'sheetId': sheet_id,
                            'startRowIndex': 0,
                            'endRowIndex': 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(headers)
                        },
                        'cell': {
                            'userEnteredFormat': {
                                'backgroundColor': {
                                    'red': 0.85,
                                    'green': 0.85,
                                    'blue': 0.85
                                },
                                'textFormat': {
                                    'bold': True
                                }
                            }
                        },
                        'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                    }
                }
            ]
            
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': requests}
            ).execute()
            
            print("[Sheets] Header row added with formatting")
            return True
        else:
            print("[Sheets] Header row already exists")
            return True
            
    except Exception as e:
        print(f"[Sheets] Error ensuring header: {str(e)}")
        traceback.print_exc()
        return False


def append_row_to_sheet(service, spreadsheet_id: str, row_data: Dict[str, any], existing_emails=None):
    try:
        # Ensure header exists
        ensure_sheet_has_header(service, spreadsheet_id)
        column_map = get_sheet_column_map(service, spreadsheet_id)
        if not column_map:
            print("[Sheets] Warning: Could not get column mapping")
            return False

        email_col_index = column_map.get("Email")
        if email_col_index is not None and existing_emails is not None:            
            email_value = str(row_data.get("Email", "")).strip().lower()
            if email_value and email_value in existing_emails:
                print(f"[Sheets] Row with email '{email_value}' already exists in Email column. Skipping append.")
                return False
            # Add the new email to the set so subsequent calls in the same batch are aware
            existing_emails.add(email_value)

        max_col_index = max(column_map.values())
        row_array = [''] * (max_col_index + 1)
        for column_name, value in row_data.items():
            if column_name in column_map:
                index = column_map[column_name]
                row_array[index] = value
            else:
                print(f"[Sheets] Warning: Column '{column_name}' not found in header")
        body = {'values': [row_array]}
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1',
            valueInputOption='USER_ENTERED',
            insertDataOption='OVERWRITE',
            body=body
        ).execute()
        return True
    except Exception as e:
        print(f"[Sheets] Error: {str(e)}")
        return False


def overwrite_sheet_data(service, spreadsheet_id: str, rows_data: List[Dict[str, any]]):
    """
    Overwrite all data in the Google Sheet except the header row.
    
    Args:
        service: Google Sheets API service
        spreadsheet_id: ID of the spreadsheet
        rows_data: List of dictionaries mapping column names to values
        
    Returns:
        Number of rows written, or -1 if failed
    """
    try:
        # Ensure header exists
        ensure_sheet_has_header(service, spreadsheet_id)
        
        # Get the column mapping from header
        column_map = get_sheet_column_map(service, spreadsheet_id)
        
        if not column_map:
            print("[Sheets] Warning: Could not get column mapping, sheet may be empty")
            return -1
        
        # Get the number of columns in the sheet
        max_col = max(column_map.values()) + 1 if column_map else 0
        
        # Build all data rows
        data_rows = []
        for row_data in rows_data:
            # Create row array filled with empty strings
            row_array = [''] * max_col
            
            # Fill in the values based on column mapping
            for column_name, value in row_data.items():
                if column_name in column_map:
                    index = column_map[column_name]
                    row_array[index] = value
            
            data_rows.append(row_array)
        
        if not data_rows:
            print("[Sheets] No data rows to write")
            return 0
        
        # First, clear all data except header (from row 2 onwards)
        # Get the current number of rows
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range='A:A'
        ).execute()
        current_rows = len(result.get('values', []))
        
        if current_rows > 1:
            # Clear everything from row 2 onwards
            clear_range = f'A2:Z{current_rows}'
            service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=clear_range
            ).execute()
            print(f"[Sheets] Cleared data from row 2 to {current_rows}")
        
        # Write new data starting from row 2
        body = {
            'values': data_rows
        }
        
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='A2',  # Start from row 2 (after header)
            valueInputOption='USER_ENTERED',  # Allows formulas like HYPERLINK
            body=body
        ).execute()
        
        updated_rows = result.get('updatedRows', 0)
        print(f"[Sheets] Overwrote sheet data: {updated_rows} row(s) written")
        return updated_rows
        
    except Exception as e:
        print(f"[Sheets] Error overwriting sheet data: {str(e)}")
        traceback.print_exc()
        return -1


@app.get("/sync-emails")
async def sync_emails(
    user_id: Optional[str] = None,
    start_date: Optional[str] = "auto",
    spreadsheet_id: Optional[str] = None
):
    """
    Sync new emails from Gmail (using auto date filter) and process them for MBTI data in one operation.
    
    This endpoint combines:
    1. Fetching emails with start_date="auto" (only emails newer than the most recent in database)
    2. Processing the newly fetched emails for MBTI and pie chart analysis
    3. Syncing results to Google Sheet (if spreadsheet_id is provided or configured)
    
    Query Parameters:
    - user_id: User identifier (optional, uses config.user_id if not provided)
    - start_date: Start date in YYYY/MM/DD format, or "auto" to use most recent email date (default: "auto")
    - spreadsheet_id: Optional Google Sheets spreadsheet ID to sync results to (overrides config)
    
    Settings like subject_filter, save_attachments, gemini_api_key, max_workers, and skip_mbti
    are read from config.json.
    
    Returns:
    - Combined results from both fetching and processing operations
    """
    # Get settings from config
    if app_config is None:
        raise HTTPException(
            status_code=500,
            detail="Configuration not loaded. Please ensure config.json exists."
        )
    
    user_id = get_user_id_or_fallback(user_id)
       
    # Use provided spreadsheet_id or fall back to config
    if spreadsheet_id is None and app_config.auto_sync_to_sheet:
        spreadsheet_id = app_config.google_sheet_id
    try:
        print(f"[SyncProcess] Starting sync and process for user_id: {user_id}")
        
        # Step 1: Fetch new emails with auto date filter
        fetch_result = await fetch_emails(
            user_id=user_id,
            start_page=1,
            start_date=start_date
        )
        
        if fetch_result["status"] != "success":
            return {
                "status": "error",
                "message": "Failed to fetch emails",
                "fetch_result": fetch_result
            }
        
        print(f"[SyncProcess] Fetch completed: {fetch_result['saved_count']} new emails")
        
        # Step 2: Process the newly fetched emails (only for this user)
        synced_to_sheet = 0
        if fetch_result["saved_count"] > 0:
            print(f"[SyncProcess] Step 2: Processing {fetch_result['saved_count']} emails...")
            process_result = await process_emails(
                user_id=user_id,
                batch_size=fetch_result["saved_count"]
            )
            
            # Step 3: Sync results to Google Sheet if spreadsheet_id is provided
            if spreadsheet_id and process_result.get('successful', 0) > 0:
                print(f"[SyncProcess] Step 3: Syncing {process_result.get('successful', 0)} results to Google Sheet...")
                
                try:
                    # Get user credentials for Google Sheets API
                    creds_data = get_user_credentials(user_id)
                    if not creds_data:
                        print(f"[SyncProcess] Warning: No credentials found for user {user_id}, skipping sheet sync")
                    else:
                        # Build credentials
                        credentials = Credentials(
                            token=creds_data["token"],
                            refresh_token=creds_data.get("refresh_token"),
                            token_uri=creds_data["token_uri"],
                            client_id=creds_data["client_id"],
                            client_secret=creds_data["client_secret"],
                            scopes=creds_data["scopes"],
                        )
                        
                        # Build Google Sheets service
                        sheets_service = build('sheets', 'v4', credentials=credentials)
                        
                        # Get the newly processed results (only successful ones)
                        results_to_sync = [r for r in process_result.get('results', []) if r.get('success') == 'success']
                        
                        # Build list of (result, email, date) tuples for sorting
                        results_with_emails = []
                        for result in results_to_sync:
                            email = get_email_details_by_message_id(result['message_id'])
                            if email:
                                email_date = email.get('date')
                                results_with_emails.append((result, email, email_date))
                            else:
                                print(f"[SyncProcess] Warning: Could not find email for message_id {result['message_id']}")
                        
                        # Sort by email date (ascending - oldest first)
                        results_with_emails.sort(key=lambda x: x[2] if x[2] else datetime.min.replace(tzinfo=timezone.utc))
                        
                        # Filter out duplicate emails by sender_email, keeping only the first (oldest) in chronological order
                        unique_results = []
                        seen_emails = set()
                        for result, email, email_date in results_with_emails:
                            sender_email = (email.get('sender_email') if isinstance(email, dict) else getattr(email, 'sender_email', None))
                            email_key = sender_email.strip().lower() if sender_email else ''
                            if email_key and email_key not in seen_emails:
                                unique_results.append((result, email, email_date))
                                seen_emails.add(email_key)
                        results_with_emails = unique_results
                        
                        existing_emails = None
                        try:
                            # Ensure header exists
                            ensure_sheet_has_header(sheets_service, spreadsheet_id)
                            column_map = get_sheet_column_map(sheets_service, spreadsheet_id)
                            if column_map:                            
                                email_col_index = column_map.get("Email")
                                if email_col_index is not None:
                                    col_letter = ""
                                    idx = email_col_index
                                    while idx >= 0:
                                        col_letter = chr(65 + (idx % 26)) + col_letter
                                        idx = idx // 26 - 1
                                    result = sheets_service.spreadsheets().values().get(
                                        spreadsheetId=spreadsheet_id,
                                        range=f"Sheet1!{col_letter}2:{col_letter}",
                                    ).execute()
                                    existing_emails = set(row[0].strip().lower() for row in result.get("values", []) if row and row[0])
                        except Exception as e:
                            print(f"[Sheets] Error: {str(e)}")
    
                        
                        # Insert rows in sorted order
                        for result, email, email_date in results_with_emails:
                            # Format row data using helper function
                            row_data = format_result_for_sheet(result, email)
                            
                            # Append row to sheet
                            success = append_row_to_sheet(sheets_service, spreadsheet_id, row_data, existing_emails)
                            if success:
                                synced_to_sheet += 1
                        
                        print(f"[SyncProcess] Successfully synced {synced_to_sheet} rows to Google Sheet")
                        
                except Exception as sheet_error:
                    print(f"[SyncProcess] Error syncing to Google Sheet: {str(sheet_error)}")
                    traceback.print_exc()
                    # Don't fail the entire operation if sheet sync fails
            
            return {
                "status": "success",
                "message": f"Synced {fetch_result['saved_count']} new emails and processed {process_result.get('total_processed', 0)}",
                "fetch_result": {
                    "total_found": fetch_result["total_found"],
                    "saved_count": fetch_result["saved_count"],
                },
                "process_result": {
                    "total_processed": process_result.get("total_processed", 0),
                    "successful": process_result.get("successful", 0),
                    "failed": process_result.get("failed", 0)
                },
                "sheet_sync": {
                    "synced_count": synced_to_sheet
                } if spreadsheet_id else None
            }
        else:
            return {
                "status": "success",
                "message": "No new emails to process",
                "fetch_result": {
                    "total_found": fetch_result["total_found"],
                    "saved_count": fetch_result["saved_count"]
                },
                "process_result": {
                    "total_processed": 0,
                    "successful": 0,
                    "failed": 0
                }
            }
    
    except Exception as e:
        print(f"[SyncProcess] Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error in sync and process operation: {str(e)}"
        )


@app.get("/process-emails")
async def process_emails(
    user_id: Optional[str] = None,
    batch_size: Optional[int] = 10000
):
    """
    Process unprocessed emails to extract MBTI and pie chart data.
    
    Query Parameters:
    - user_id: User ID to filter emails (optional, uses config.user_id if not provided)
    - batch_size: Number of emails to process (default: 10000)
    
    Settings like max_workers, gemini_api_key, and skip_mbti are read from config.json.
    
    Returns:
    - Processing summary with success/failure counts
    """
    # Get settings from config
    if app_config is None:
        raise HTTPException(
            status_code=500,
            detail="Configuration not loaded. Please ensure config.json exists."
        )
    
    user_id = get_user_id_or_fallback(user_id)
    
    max_workers = app_config.max_workers
    gemini_api_key = app_config.gemini_api_key
    skip_mbti = app_config.skip_mbti
    debug_mode = app_config.debug_mode
    
    try:
        print(f"[ProcessEmails] Starting processing for user_id: {user_id}")
        print(f"[ProcessEmails] Batch size: {batch_size}, Workers: {max_workers}")
        
        # Get unprocessed emails
        unprocessed_emails = get_unprocessed_emails(
            user_id=user_id,
            limit=batch_size
        )
        
        if not unprocessed_emails:
            return {
                'status': 'success',
                'message': 'No unprocessed emails found',
                'total_processed': 0,
                "successful": 0,
                "failed": 0
            }
        
        print(f"[ProcessEmails] Found {len(unprocessed_emails)} unprocessed emails")
        
        # Process emails in parallel using ThreadPoolExecutor
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_email = {
                executor.submit(process_single_email, email, gemini_api_key, skip_mbti, debug_mode): email
                for email in unprocessed_emails
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_email):
                email = future_to_email[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save result to database
                    save_mbti_result(result)
                    
                except Exception as e:
                    print(f"[ProcessEmails] Error processing email {email['message_id']}: {str(e)}")
                    error_result = {
                        'email_id': email['id'],
                        'message_id': email['message_id'],
                        'success': 'error',
                        'error_message': str(e)
                    }
                    results.append(error_result)
                    save_mbti_result(error_result)
        
        # Count successes and failures
        successful = sum(1 for r in results if r.get('success') == 'success')
        failed = len(results) - successful
        
        print(f"[ProcessEmails] Completed: {successful} successful, {failed} failed")
        
        return {
            'status': 'success',
            'message': f'Processed {len(results)} emails',
            'total_processed': len(results),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    except Exception as e:
        print(f"[ProcessEmails] Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing emails: {str(e)}"
        )


@app.get("/mbti-results")
async def get_results(
    user_id: Optional[str] = None,
    format: Optional[str] = "json",
    save_csv: Optional[bool] = False,
    spreadsheet_id: Optional[str] = None
):
    """
    Get MBTI processing results.
    
    Query Parameters:
    - user_id: User ID to filter results (optional, uses config.user_id if not provided)
    - format: Response format, either "json" or "csv" (default: "json")
    - save_csv: Whether to save results as CSV on server (default: False)
    - spreadsheet_id: Optional Google Sheets spreadsheet ID to save results to (overwrites all data except header)
    
    Returns:
    - List of MBTI results in requested format
    """
    user_id = get_user_id_or_fallback(user_id)
    
    if spreadsheet_id is None:
        spreadsheet_id = app_config.google_sheet_id
        
    try:
        mbti_reports_base = os.getenv("MBTI_REPORTS_PATH", "./mbti_reports")
        mbti_reports_dir = Path(mbti_reports_base) / user_id
        if save_csv:
            mbti_reports_dir.mkdir(parents=True, exist_ok=True)
        
        results = get_mbti_results(user_id=user_id)
        
        # Handle the structure where results is a list of (MBTIResult, Email) tuples
        mbti_results = []
        email_results = []
        for i, item in enumerate(results):
            # Handle SQLAlchemy Row/tuple with exactly 2 elements
            if hasattr(item, '__len__') and len(item) == 2:
                try:
                    mbti_result, email = item[0], item[1]
                    mbti_results.append(mbti_result)
                    email_results.append(email)
                    continue
                except Exception as e:
                    print(f"[Debug] Failed to unpack item {i}: {e}")
                    continue        
        
        # Save to local CSV file if requested
        if save_csv:
            try:
                csv_filename = mbti_reports_dir / f"mbti_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header with requested columns
                    writer.writerow(['ID', 'Email', 'Name', 'Subject', 'Role', 'Date', 'Link to Email', '% Cert', 'MBTI', 'Ennegram', 'Tritype', 'Tritype (no 8)'])
                    
                    # Write data rows
                    for i, mbti_result in enumerate(mbti_results):
                        email = email_results[i] if i < len(email_results) else None
                        
                        # Build permalink using message ID (or thread ID if available)
                        link_to_email = ""
                        if email:
                            permalink = getattr(email, 'permalink', None)
                            if permalink:
                                link_to_email = permalink
                            elif email.message_id:
                                link_to_email = f"https://mail.google.com/mail/u/0/#inbox/{email.message_id}"
                        
                        # Format % Cert field
                        cert_field = ""
                        if all(getattr(mbti_result, attr, None) is not None for attr in ['energy', 'mind', 'nature', 'tactics', 'identity']):
                            cert_field = f"{mbti_result.energy_type or 'E'} - {mbti_result.energy or 0}% / {mbti_result.mind_type or 'N'} - {mbti_result.mind or 0}% / {mbti_result.nature_type or 'T'} - {mbti_result.nature or 0}% / {mbti_result.tactics_type or 'J'} - {mbti_result.tactics or 0}% / {mbti_result.identity_type or 'A'} - {mbti_result.identity or 0}%"
                        
                        # Format Enneagram field
                        ennegram_field = ""
                        segments = [mbti_result.segment_1, mbti_result.segment_2, mbti_result.segment_3, mbti_result.segment_4, mbti_result.segment_5, mbti_result.segment_6, mbti_result.segment_7, mbti_result.segment_8, mbti_result.segment_9]
                        if any(seg is not None for seg in segments):
                            ennegram_parts = []
                            for j, seg in enumerate(segments, 1):
                                if seg is not None:
                                    ennegram_parts.append(f"{j}: {seg:.2f}")
                            ennegram_field = " / ".join(ennegram_parts)
                        
                        # Extract role from subject
                        role = ''
                        if email and email.subject:
                            role_match = re.search(r'Your recent application for (.+)', email.subject)
                            if role_match:
                                role = role_match.group(1).strip()
                                
                        writer.writerow([
                            email.message_id if email else (getattr(mbti_result, 'message_id', '') or ''),  # ID - use message ID with safety check
                            email.sender_email if email else '',  # Email
                            email.sender if email else '',  # Name (sender's name)
                            email.subject if email else '',  # Subject
                            role,
                            email.date.strftime('%Y-%m-%d') if email and email.date else '',  # Date
                            link_to_email,  # Link to Email - using built permalink
                            cert_field,  # % Cert
                            mbti_result.mbti_type or '',  # MBTI
                            ennegram_field,  # Ennegram
                            mbti_result.tritype or '',  # Tritype
                            mbti_result.tritype_no_8 or ''  # Tritype (no 8)
                        ])
                
                print(f"[CSV Export] Saved {len(mbti_results)} records to {csv_filename}")
                
            except Exception as csv_error:
                print(f"[CSV Export] Error saving CSV file: {str(csv_error)}")
        
        # Save to Google Sheets if spreadsheet_id is provided
        if spreadsheet_id and len(mbti_results) > 0:
            print(f"[Sheets] Saving {len(mbti_results)} results to Google Sheet (overwrite mode)...")
            
            try:
                # Get user credentials for Google Sheets API
                creds_data = get_user_credentials(user_id)
                if not creds_data:
                    print(f"[Sheets] Warning: No credentials found for user {user_id}, skipping sheet save")
                else:
                    # Build credentials
                    credentials = Credentials(
                        token=creds_data["token"],
                        refresh_token=creds_data.get("refresh_token"),
                        token_uri=creds_data["token_uri"],
                        client_id=creds_data["client_id"],
                        client_secret=creds_data["client_secret"],
                        scopes=creds_data["scopes"],
                    )
                    
                    # Build Google Sheets service
                    sheets_service = build('sheets', 'v4', credentials=credentials)
                    
                    # Build all rows data using helper function
                    all_rows_data = []
                    for i, mbti_result in enumerate(mbti_results):
                        email = email_results[i] if i < len(email_results) else None
                        if email:
                            row_data = format_result_for_sheet(mbti_result, email)
                            all_rows_data.append(row_data)
                    
                    # Overwrite sheet data (keeping header)
                    rows_written = overwrite_sheet_data(sheets_service, spreadsheet_id, all_rows_data)
                    
                    if rows_written > 0:
                        print(f"[Sheets] Successfully overwrote {rows_written} rows in Google Sheet")
                    else:
                        print(f"[Sheets] Failed to overwrite data in Google Sheet")
                    
            except Exception as sheet_error:
                print(f"[Sheets] Error saving to Google Sheet: {str(sheet_error)}")
                traceback.print_exc()
                # Don't fail the entire operation if sheet save fails
        
        if format.lower() == "csv":
            # Return flat CSV-like format matching saved CSV structure
            csv_data = []
            
            # Add header row (same as saved CSV)
            header = ['ID', 'Email', 'Name', 'Subject', 'Role', 'Date', 'Link to Email', '% Cert', 'MBTI', 'Ennegram', 'Tritype', 'Tritype (no 8)']
            csv_data.append(header)
            
            # Add data rows
            for i, mbti_result in enumerate(mbti_results):
                email = email_results[i] if i < len(email_results) else None
                
                # Build permalink for email
                link_to_email = ""
                if email:
                    permalink = getattr(email, 'permalink', None)
                    if permalink:
                        link_to_email = permalink
                    elif email.message_id:
                        link_to_email = f"https://mail.google.com/mail/u/0/#inbox/{email.message_id}"
                
                # Format % Cert field
                cert_field = ""
                if all(getattr(mbti_result, attr, None) is not None for attr in ['energy', 'mind', 'nature', 'tactics', 'identity']):
                    cert_field = f"{mbti_result.energy_type or 'E'} - {mbti_result.energy or 0}% / {mbti_result.mind_type or 'N'} - {mbti_result.mind or 0}% / {mbti_result.nature_type or 'T'} - {mbti_result.nature or 0}% / {mbti_result.tactics_type or 'J'} - {mbti_result.tactics or 0}% / {mbti_result.identity_type or 'A'} - {mbti_result.identity or 0}%"
                
                # Format Enneagram field
                ennegram_field = ""
                segments = [mbti_result.segment_1, mbti_result.segment_2, mbti_result.segment_3, mbti_result.segment_4, mbti_result.segment_5, mbti_result.segment_6, mbti_result.segment_7, mbti_result.segment_8, mbti_result.segment_9]
                if any(seg is not None for seg in segments):
                    ennegram_parts = []
                    for j, seg in enumerate(segments, 1):
                        if seg is not None:
                            ennegram_parts.append(f"{j}: {seg:.2f}")
                    ennegram_field = " / ".join(ennegram_parts)
                
                # Extract role from subject
                role = ''
                if email and email.subject:
                    role_match = re.search(r'Your recent application for (.+)', email.subject)
                    if role_match:
                        role = role_match.group(1).strip()
                
                row = [
                    email.message_id if email else (getattr(mbti_result, 'message_id', '') or ''),  # ID
                    email.sender_email if email else '',  # Email
                    email.sender if email else '',  # Name
                    email.subject if email else '',  # Subject
                    role,  # Role
                    email.date.strftime('%Y-%m-%d') if email and email.date else '',  # Date
                    link_to_email,  # Link to Email
                    cert_field,  # % Cert
                    mbti_result.mbti_type or '',  # MBTI
                    ennegram_field,  # Ennegram
                    mbti_result.tritype or '',  # Tritype
                    mbti_result.tritype_no_8 or ''  # Tritype (no 8)
                ]
                csv_data.append(row)
            
            response = {
                'status': 'success',
                'count': len(mbti_results),
                'format': 'csv',
                'data': csv_data
            }
                           
            return response
        
        else:
            # Return JSON format matching saved CSV structure
            results_list = []
            for i, mbti_result in enumerate(mbti_results):
                email = email_results[i] if i < len(email_results) else None
                
                # Build permalink for email
                link_to_email = ""
                if email:
                    permalink = getattr(email, 'permalink', None)
                    if permalink:
                        link_to_email = permalink
                    elif email.message_id:
                        link_to_email = f"https://mail.google.com/mail/u/0/#inbox/{email.message_id}"
                
                # Format % Cert field (same as saved CSV)
                cert_field = ""
                if all(getattr(mbti_result, attr, None) is not None for attr in ['energy', 'mind', 'nature', 'tactics', 'identity']):
                    cert_field = f"{mbti_result.energy_type or 'E'} - {mbti_result.energy or 0}% / {mbti_result.mind_type or 'N'} - {mbti_result.mind or 0}% / {mbti_result.nature_type or 'T'} - {mbti_result.nature or 0}% / {mbti_result.tactics_type or 'J'} - {mbti_result.tactics or 0}% / {mbti_result.identity_type or 'A'} - {mbti_result.identity or 0}%"
                
                # Format Enneagram field (same as saved CSV)
                ennegram_field = ""
                segments = [mbti_result.segment_1, mbti_result.segment_2, mbti_result.segment_3, mbti_result.segment_4, mbti_result.segment_5, mbti_result.segment_6, mbti_result.segment_7, mbti_result.segment_8, mbti_result.segment_9]
                if any(seg is not None for seg in segments):
                    ennegram_parts = []
                    for j, seg in enumerate(segments, 1):
                        if seg is not None:
                            ennegram_parts.append(f"{j}: {seg:.2f}")
                    ennegram_field = " / ".join(ennegram_parts)
                
                # Extract role from subject (same as saved CSV)
                role = ''
                if email and email.subject:
                    role_match = re.search(r'Your recent application for (.+)', email.subject)
                    if role_match:
                        role = role_match.group(1).strip()
                
                # Structure matching saved CSV columns
                result_dict = {
                    'ID': email.message_id if email else (getattr(mbti_result, 'message_id', '') or ''),
                    'Email': email.sender_email if email else '',
                    'Name': email.sender if email else '',
                    'Subject': email.subject if email else '',
                    'Role': role,
                    'Date': email.date.strftime('%Y-%m-%d') if email and email.date else '',
                    'Link to Email': link_to_email,
                    '% Cert': cert_field,
                    'MBTI': mbti_result.mbti_type or '',
                    'Ennegram': ennegram_field,
                    'Tritype': mbti_result.tritype or '',
                    'Tritype (no 8)': mbti_result.tritype_no_8 or ''
                }
                
                results_list.append(result_dict)
            
            response = {
                'status': 'success',
                'count': len(results_list),
                'format': 'json',
                'results': results_list
            }
            
            # Add CSV file info if saved
            if save_csv:
                response['csv_file'] = f"mbti_results_{user_id or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
            return response
    
    except Exception as e:
        print(f"[GetResults] Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error getting results: {str(e)}"
        )



if __name__ == "__main__":
    import uvicorn
    port = 8080
    if app_config and hasattr(app_config, "port"):
        try:
            port = int(app_config.port)
        except Exception:
            print(f"[Config] Invalid port in config, using default 8080")
    uvicorn.run(app, host="0.0.0.0", port=port)
    
