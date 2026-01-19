from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from datetime import datetime, timezone
import json
import os

# Database setup - use Railway volume for persistence
DATABASE_PATH = os.getenv("DATABASE_PATH", "./emails.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserCredential(Base):
    __tablename__ = "user_credentials"
    
    user_id = Column(String, primary_key=True, index=True)
    token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    token_uri = Column(String, nullable=False)
    client_id = Column(String, nullable=False)
    client_secret = Column(String, nullable=False)
    scopes = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class Email(Base):
    __tablename__ = "emails"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    message_id = Column(String, unique=True, index=True, nullable=False)
    thread_id = Column(String, index=True)
    subject = Column(Text)
    sender = Column(String, index=True)
    sender_email = Column(String, index=True)
    date = Column(DateTime, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship to attachments
    attachments = relationship("Attachment", back_populates="email", cascade="all, delete-orphan")

class Attachment(Base):
    __tablename__ = "attachments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email_id = Column(Integer, ForeignKey('emails.id'), nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    size_bytes = Column(Integer)
    file_path = Column(Text)  # Path to stored file
    attachment_id = Column(String)  # Gmail attachment ID
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship to email
    email = relationship("Email", back_populates="attachments")

class MBTIResult(Base):
    __tablename__ = "mbti_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email_id = Column(Integer, ForeignKey('emails.id'), nullable=False, unique=True)
    message_id = Column(String, index=True, nullable=False)
    
    # Processing status
    success = Column(String, nullable=False)  # 'success', 'gemini_failed', 'api_failed', 'error'
    error_message = Column(Text)
    
    # MBTI data (from Gemini)
    mbti_type = Column(String)  # e.g., "INTJ-A"
    energy = Column(Integer)  # percentage
    energy_type = Column(String)  # E or I
    mind = Column(Integer)
    mind_type = Column(String)  # N or S
    nature = Column(Integer)
    nature_type = Column(String)  # T or F
    tactics = Column(Integer)
    tactics_type = Column(String)  # J or P
    identity = Column(Integer)
    identity_type = Column(String)  # A or T
    personality_image_index = Column(Integer)  # 0 or 1 (which image had personality data)
    
    # Enneagram data (from pie chart API)
    segment_1 = Column(Float)
    segment_2 = Column(Float)
    segment_3 = Column(Float)
    segment_4 = Column(Float)
    segment_5 = Column(Float)
    segment_6 = Column(Float)
    segment_7 = Column(Float)
    segment_8 = Column(Float)
    segment_9 = Column(Float)
    
    # Tritype calculations
    tritype = Column(Integer)  # Three-digit number with segment 8
    tritype_no_8 = Column(Integer)  # Three-digit number without segment 8
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

# Create tables
Base.metadata.create_all(bind=engine)

def save_user_credentials(user_id: str, credentials_dict: dict):
    """Save or update user credentials in database."""
    db = SessionLocal()
    try:
        # Check if user already exists
        user_cred = db.query(UserCredential).filter(UserCredential.user_id == user_id).first()
        
        if user_cred:
            # Update existing
            user_cred.token = credentials_dict['token']
            user_cred.refresh_token = credentials_dict.get('refresh_token')
            user_cred.token_uri = credentials_dict['token_uri']
            user_cred.client_id = credentials_dict['client_id']
            user_cred.client_secret = credentials_dict['client_secret']
            user_cred.scopes = json.dumps(credentials_dict['scopes'])
            user_cred.updated_at = datetime.now(timezone.utc)
        else:
            # Create new
            user_cred = UserCredential(
                user_id=user_id,
                token=credentials_dict['token'],
                refresh_token=credentials_dict.get('refresh_token'),
                token_uri=credentials_dict['token_uri'],
                client_id=credentials_dict['client_id'],
                client_secret=credentials_dict['client_secret'],
                scopes=json.dumps(credentials_dict['scopes'])
            )
            db.add(user_cred)
        
        db.commit()
        print(f"[Database] Saved credentials for user_id: {user_id}")
        return True
    except Exception as e:
        db.rollback()
        print(f"[Database] Error saving credentials: {str(e)}")
        return False
    finally:
        db.close()

def get_user_credentials(user_id: str):
    """Get user credentials from database."""
    db = SessionLocal()
    try:
        user_cred = db.query(UserCredential).filter(UserCredential.user_id == user_id).first()
        
        if not user_cred:
            return None
        
        return {
            'token': user_cred.token,
            'refresh_token': user_cred.refresh_token,
            'token_uri': user_cred.token_uri,
            'client_id': user_cred.client_id,
            'client_secret': user_cred.client_secret,
            'scopes': json.loads(user_cred.scopes)
        }
    except Exception as e:
        print(f"[Database] Error getting credentials: {str(e)}")
        return None
    finally:
        db.close()

def delete_user_credentials(user_id: str):
    """Delete user credentials from database."""
    db = SessionLocal()
    try:
        user_cred = db.query(UserCredential).filter(UserCredential.user_id == user_id).first()
        if user_cred:
            db.delete(user_cred)
            db.commit()
            print(f"[Database] Deleted credentials for user_id: {user_id}")
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"[Database] Error deleting credentials: {str(e)}")
        return False
    finally:
        db.close()


def get_email_by_message_id(message_id: str):
    """Check if email exists by message_id."""
    db = SessionLocal()
    try:
        email = db.query(Email).filter(Email.message_id == message_id).first()
        return email.id if email else None
    except Exception as e:
        print(f"[Database] Error checking email: {str(e)}")
        return None
    finally:
        db.close()


def get_email_details_by_message_id(message_id: str):
    """Get full email details by message_id."""
    db = SessionLocal()
    try:
        email = db.query(Email).filter(Email.message_id == message_id).first()
        if not email:
            return None
        
        # Convert to dictionary
        return {
            'id': email.id,
            'user_id': email.user_id,
            'message_id': email.message_id,
            'thread_id': email.thread_id,
            'subject': email.subject,
            'sender': email.sender,
            'sender_email': email.sender_email,
            'date': email.date,
            'created_at': email.created_at
        }
    except Exception as e:
        print(f"[Database] Error getting email details: {str(e)}")
        return None
    finally:
        db.close()


def save_email_with_attachments(user_id: str, email_data: dict, attachments_data: list):
    """
    Save email and its attachments to database.
    
    Args:
        user_id: User identifier
        email_data: Dictionary with email details
        attachments_data: List of attachment dictionaries
    
    Returns:
        Email ID if successful, None otherwise
    """
    db = SessionLocal()
    try:
        # Check if email already exists
        existing = db.query(Email).filter(Email.message_id == email_data['message_id']).first()
        if existing:
            print(f"[Database] Email {email_data['message_id']} already exists")
            return existing.id
        
        # Create email record
        email = Email(
            user_id=user_id,
            message_id=email_data['message_id'],
            thread_id=email_data.get('thread_id'),
            subject=email_data.get('subject'),
            sender=email_data.get('sender'),
            sender_email=email_data.get('sender_email'),
            date=email_data.get('date')
        )
        db.add(email)
        db.flush()  # Get the email ID
        
        # Create attachment records
        for att_data in attachments_data:
            attachment = Attachment(
                email_id=email.id,
                filename=att_data['filename'],
                content_type=att_data.get('content_type'),
                size_bytes=att_data.get('size_bytes'),
                file_path=att_data.get('file_path'),
                attachment_id=att_data.get('attachment_id')
            )
            db.add(attachment)
        
        db.commit()
        print(f"[Database] Saved email {email_data['message_id']} with {len(attachments_data)} attachments")
        return email.id
        
    except Exception as e:
        db.rollback()
        print(f"[Database] Error saving email: {str(e)}")
        return None
    finally:
        db.close()


def get_emails_by_user(user_id: str, limit: int = 100):
    """Get emails for a specific user."""
    db = SessionLocal()
    try:
        emails = db.query(Email).filter(Email.user_id == user_id).order_by(Email.date.desc()).limit(limit).all()
        return emails
    except Exception as e:
        print(f"[Database] Error getting emails: {str(e)}")
        return []
    finally:
        db.close()


def get_most_recent_email_date(user_id: str):
    """Get the date of the most recent email for a user."""
    db = SessionLocal()
    try:
        email = db.query(Email).filter(Email.user_id == user_id).order_by(Email.date.desc()).first()
        return email.date if email else None
    except Exception as e:
        print(f"[Database] Error getting most recent email date: {str(e)}")
        return None
    finally:
        db.close()


def get_unprocessed_emails(user_id: str = None, limit: int = None):
    """Get emails that haven't been processed for MBTI yet."""
    db = SessionLocal()
    try:
        # Get emails with at least 2 attachments that don't have MBTI results
        # Use joinedload to eagerly load attachments to avoid session issues
        query = db.query(Email).options(joinedload(Email.attachments)).outerjoin(MBTIResult).filter(MBTIResult.id == None)
        
        if user_id:
            query = query.filter(Email.user_id == user_id)
        
        query = query.order_by(Email.date.desc())
        
        if limit:
            query = query.limit(limit)
        
        emails = query.all()
        
        # Filter emails to only include those with image attachments
        # and convert to dictionaries to avoid session issues
        result = []
        for email in emails:
            # Access attachments while session is still active
            image_attachments = [
                att for att in email.attachments 
                if att.content_type and att.content_type.startswith('image/') and att.file_path
            ]
            
            if len(image_attachments) >= 2:  # At least 2 image attachments
                email_dict = {
                    'id': email.id,
                    'message_id': email.message_id,
                    'subject': email.subject,
                    'sender': email.sender,
                    'sender_email': email.sender_email,
                    'date': email.date,
                    'user_id': email.user_id,
                    'created_at': email.created_at,
                    'attachments': [
                        {
                            'id': att.id,
                            'filename': att.filename,
                            'content_type': att.content_type,
                            'size_bytes': att.size_bytes,
                            'file_path': os.path.join(os.getenv("ATTACHMENTS_PATH", "./attachments"), email.user_id, att.file_path),
                            'attachment_id': att.attachment_id
                        }
                        for att in image_attachments
                    ]
                }
                result.append(email_dict)
        
        return result
    except Exception as e:
        print(f"[Database] Error getting unprocessed emails: {str(e)}")
        return []
    finally:
        db.close()


def save_mbti_result(result_data: dict):
    """Save MBTI processing result to database."""
    db = SessionLocal()
    try:
        # Check if result already exists for this email
        existing = db.query(MBTIResult).filter(MBTIResult.email_id == result_data['email_id']).first()
        
        if existing:
            # Update existing record
            for key, value in result_data.items():
                if hasattr(existing, key) and key != 'id':
                    setattr(existing, key, value)
            existing.updated_at = datetime.now(timezone.utc)
            print(f"[Database] Updated MBTI result for email_id: {result_data['email_id']}")
        else:
            # Create new record
            mbti_result = MBTIResult(**result_data)
            db.add(mbti_result)
            print(f"[Database] Created new MBTI result for email_id: {result_data['email_id']}")
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[Database] Error saving MBTI result: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def get_mbti_results(user_id: str = None):
    """Get MBTI results with email data, optionally filtered by user. Returns only one result per sender email (earliest occurrence)."""
    db = SessionLocal()
    try:
        # Join MBTIResult with Email to get email details
        query = db.query(MBTIResult, Email).join(Email, MBTIResult.email_id == Email.id)
              
        if user_id:
            query = query.filter(Email.user_id == user_id)
        
        # Order by date to get consistent results (earliest first)
        all_results = query.order_by(Email.date.asc()).all()
        
        # Filter to keep only one result per sender email (the first one chronologically)
        seen_emails = set()
        unique_results = []
        
        for mbti_result, email in all_results:
            sender_email = email.sender_email.lower() if email.sender_email else None
            if sender_email and sender_email not in seen_emails:
                seen_emails.add(sender_email)
                unique_results.append((mbti_result, email))
            elif not sender_email:
                # If no sender_email, include it (shouldn't happen but handle gracefully)
                unique_results.append((mbti_result, email))
        
        # Detach objects from session so they can be used after session is closed
        for mbti_result, email in unique_results:
            if mbti_result:
                db.expunge(mbti_result)
            if email:
                db.expunge(email)
        
        return unique_results
    except Exception as e:
        return []
    finally:
        db.close()
