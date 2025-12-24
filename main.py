from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import httpx
import json

load_dotenv()

app = FastAPI()

# CORS configuration for Chrome extensions and web access
# Note: allow_origins=["*"] cannot be used with allow_credentials=True
# For Chrome extensions, we allow all origins but set credentials to False
# Content scripts run in page context, so requests come from mail.google.com origin
# Using ["*"] allows all origins including mail.google.com and Chrome extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (including mail.google.com and Chrome extensions)
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ---------- Azure OpenAI Client ----------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, DEPLOYMENT_NAME]):
    raise ValueError("All Azure OpenAI environment variables are required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# ---------- MongoDB Connection ----------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is required")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["replywise"]
users_collection = db["users"]

# Create unique index on email
users_collection.create_index("email", unique=True)

# ---------- Admin Configuration ----------
ADMIN_ID = os.getenv("ADMIN_ID", "admin.quickreply@gmail.com").lower()
ADMIN_PWD = os.getenv("ADMIN_PWD") or os.getenv("ADMI_PWD")  # Support both spellings
if not ADMIN_PWD:
    raise ValueError("ADMIN_PWD environment variable is required")

# ---------- Authentication Setup ----------
# SECRET_KEY is used to sign JWT tokens - should be a long random string
# For production, generate a secure random key (e.g., using: python -c "import secrets; print(secrets.token_urlsafe(32))")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-use-secrets-token-urlsafe-32")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# Initialize password context with explicit bcrypt backend
# This helps avoid version compatibility issues
# Workaround for bcrypt version detection issues on Azure
try:
    # Try to initialize with explicit settings
    pwd_context = CryptContext(
        schemes=["bcrypt"],
        deprecated="auto",
        bcrypt__rounds=12,
        bcrypt__ident="2b"  # Use bcrypt 2b format for better compatibility
    )
except Exception as e:
    print(f"Warning: bcrypt initialization with explicit settings failed: {e}")
    # Fallback: try without explicit bcrypt config
    try:
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    except Exception as e2:
        print(f"Error: Could not initialize bcrypt: {e2}")
        raise

security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    Handles bcrypt version compatibility issues gracefully.
    """
    try:
        # Truncate password if it's longer than 72 bytes (bcrypt limit)
        if isinstance(plain_password, str):
            plain_password_bytes = plain_password.encode('utf-8')
            if len(plain_password_bytes) > 72:
                plain_password = plain_password_bytes[:72].decode('utf-8', errors='ignore')
        
        return pwd_context.verify(plain_password, hashed_password)
    except (ValueError, AttributeError, TypeError) as e:
        # Log the error for debugging but don't expose details
        print(f"Password verification error: {type(e).__name__}: {str(e)}")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error in password verification: {type(e).__name__}: {str(e)}")
        return False

def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    Handles password length limits and version compatibility.
    """
    try:
        # Truncate password if it's longer than 72 bytes (bcrypt limit)
        if isinstance(password, str):
            password_bytes = password.encode('utf-8')
            if len(password_bytes) > 72:
                password = password_bytes[:72].decode('utf-8', errors='ignore')
        
        return pwd_context.hash(password)
    except (ValueError, AttributeError, TypeError) as e:
        # Log the error for debugging
        print(f"Password hashing error: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error hashing password"
        )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

# ---------- Models ----------
class EmailInput(BaseModel):
    subject: str
    sender: str
    body: str
    tone: str | None = "Professional & Polite"
    length: str | None = "Balanced reply"


class InboxLiteInput(BaseModel):
    sender: str
    subject: str

class RegenerateInput(BaseModel):
    subject: str
    original_body: str
    previous_draft: str
    options: list[str]
    instruction: str | None = ""
    tone: str | None = "Professional & Polite"
    length: str | None = "Balanced reply"

class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    email: str

class GoogleAuthRequest(BaseModel):
    id_token: str

# ---------- Authentication Endpoints ----------
@app.post("/auth/signup", response_model=TokenResponse)
async def signup(request: SignupRequest):
    # Check if user already exists
    existing_user = users_collection.find_one({"email": request.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if this is the admin account
    is_admin = request.email.lower() == ADMIN_ID
    
    # Hash password and create user
    hashed_password = get_password_hash(request.password)
    user = {
        "email": request.email,
        "hashed_password": hashed_password,
        "plan": "admin" if is_admin else "free",  # Admin or free plan
        "replies_used_this_month": 0,
        "regenerations_used_this_month": 0,
        "monthly_reset_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "learn_writing_style": False,
        "writing_samples": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    try:
        users_collection.insert_one(user)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": request.email
    }

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    try:
        # Check if this is the admin account
        is_admin = request.email.lower() == ADMIN_ID
        
        # Find user (optimized - single query)
        user = users_collection.find_one({"email": request.email})
        
        # If admin account doesn't exist, create it with password from env
        if is_admin and not user:
            hashed_password = get_password_hash(ADMIN_PWD)
            user = {
                "email": request.email,
                "hashed_password": hashed_password,
                "plan": "admin",
                "replies_used_this_month": 0,
                "regenerations_used_this_month": 0,
                "monthly_reset_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "learn_writing_style": False,
                "writing_samples": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            users_collection.insert_one(user)
        elif not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password (for admin, check against stored password or env password)
        password_valid = False
        if is_admin:
            # For admin, verify against stored password or env password
            if verify_password(request.password, user["hashed_password"]):
                password_valid = True
            elif request.password == ADMIN_PWD:
                # If env password matches, update stored password
                password_valid = True
                hashed_password = get_password_hash(ADMIN_PWD)
                users_collection.update_one(
                    {"email": request.email},
                    {"$set": {"hashed_password": hashed_password}}
                )
        else:
            # For regular users, verify stored password
            password_valid = verify_password(request.password, user["hashed_password"])
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create access token (do this before DB updates for faster response)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": request.email}, expires_delta=access_token_expires
        )
        
        # Update user info in single operation (optimized - combine updates)
        update_data = {"updated_at": datetime.utcnow()}
        if is_admin:
            update_data["plan"] = "admin"
        
        users_collection.update_one(
            {"email": request.email},
            {"$set": update_data}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "email": request.email
        }
    except HTTPException:
        # Re-raise HTTP exceptions (like 401)
        raise
    except Exception as e:
        # Log unexpected errors
        print(f"Login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )

# Add explicit OPTIONS handler for CORS preflight requests
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    from fastapi import Response
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.get("/auth/verify")
async def verify_token(current_user: dict = Depends(get_current_user)):
    return {
        "email": current_user["email"],
        "authenticated": True
    }

@app.post("/auth/google", response_model=TokenResponse)
async def google_auth(request: GoogleAuthRequest):
    try:
        # For Chrome extensions, we get the access token and fetch user info
        # The token from Chrome identity API is an OAuth2 access token
        async with httpx.AsyncClient() as client:
            user_info_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {request.id_token}"}
            )
            
            if user_info_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Failed to verify Google token"
                )
            
            user_data = user_info_response.json()
            email = user_data.get("email")
            
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email not found in Google account"
                )
        
        # Check if user exists
        user = users_collection.find_one({"email": email})
        
        if not user:
            # Create new user with Google auth
            user = {
                "email": email,
                "auth_provider": "google",
                "google_id": user_data.get("id"),
                "name": user_data.get("name"),
                "picture": user_data.get("picture"),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            users_collection.insert_one(user)
        else:
            # Update last login and Google info
            users_collection.update_one(
                {"email": email},
                {"$set": {
                    "updated_at": datetime.utcnow(),
                    "google_id": user_data.get("id"),
                    "name": user_data.get("name"),
                    "picture": user_data.get("picture"),
                    "auth_provider": "google"
                }}
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "email": email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google authentication failed: {str(e)}"
        )

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- User Management Endpoints ----------
@app.get("/user/info")
async def get_user_info(current_user: dict = Depends(get_current_user)):
    user = users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if monthly limit needs reset
    reset_date = datetime.fromisoformat(user.get("monthly_reset_date", (datetime.utcnow() + timedelta(days=30)).isoformat()))
    if datetime.utcnow() > reset_date:
        users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {
                "replies_used_this_month": 0,
                "regenerations_used_this_month": 0,
                "monthly_reset_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
            }}
        )
        user["replies_used_this_month"] = 0
        user["regenerations_used_this_month"] = 0
    
    return {
        "email": user["email"],
        "plan": user.get("plan", "free"),
        "replies_used_this_month": user.get("replies_used_this_month", 0),
        "regenerations_used_this_month": user.get("regenerations_used_this_month", 0),
        "monthly_reset_date": user.get("monthly_reset_date"),
        "learn_writing_style": user.get("learn_writing_style", False),
        "writing_samples_count": len(user.get("writing_samples", []))
    }

class LearnStyleRequest(BaseModel):
    enabled: bool

@app.put("/user/learn-style")
async def update_learn_style(request: LearnStyleRequest, current_user: dict = Depends(get_current_user)):
    user = users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_plan = user.get("plan", "free")
    if user_plan != "pro" and user_plan != "admin":
        raise HTTPException(status_code=403, detail="This feature is only available for Pro users")
    
    users_collection.update_one(
        {"email": current_user["email"]},
        {"$set": {"learn_writing_style": request.enabled}}
    )
    
    return {"success": True, "learn_writing_style": request.enabled}

@app.post("/user/reset-style")
async def reset_writing_style(current_user: dict = Depends(get_current_user)):
    user = users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_plan = user.get("plan", "free")
    if user_plan != "pro" and user_plan != "admin":
        raise HTTPException(status_code=403, detail="This feature is only available for Pro users")
    
    users_collection.update_one(
        {"email": current_user["email"]},
        {"$set": {"writing_samples": []}}
    )
    
    return {"success": True}

# ---------- Core Intelligence ----------
@app.post("/analyze-email")
def analyze_email(email: EmailInput, current_user: dict = Depends(get_current_user)):
    # Check reply limit for free users
    user = users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if monthly limit needs reset
    reset_date = datetime.fromisoformat(user.get("monthly_reset_date", (datetime.utcnow() + timedelta(days=30)).isoformat()))
    if datetime.utcnow() > reset_date:
        users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {
                "replies_used_this_month": 0,
                "monthly_reset_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
            }}
        )
        user["replies_used_this_month"] = 0
    
    # Check limit for free users
    if user.get("plan", "free") == "free":
        if user.get("replies_used_this_month", 0) >= 25:
            raise HTTPException(
                status_code=403,
                detail="Monthly reply limit reached. Upgrade to Pro for unlimited replies."
            )
    
    # Build tone instructions
    tone_instructions = {
        "Professional & Polite": "Use a professional, courteous, and respectful tone. Be formal but warm.",
        "Warm & Friendly": "Use a warm, friendly, and approachable tone. Be conversational and personable.",
        "Clear & Direct": "Use a clear, direct, and straightforward tone. Be concise and to the point.",
        "Empathetic": "Use an empathetic, understanding, and compassionate tone. Show care and concern.",
        "Apology": "Use a sincere, apologetic tone. Express genuine regret and offer solutions.",
        "Assertive": "Use a confident, assertive, and decisive tone. Be firm but respectful.",
        "Sales-friendly": "Use a persuasive, enthusiastic, and engaging tone. Highlight benefits and value.",
        "Support / HR": "Use a helpful, supportive, and professional tone. Be clear and solution-oriented."
    }
    
    tone_instruction = tone_instructions.get(email.tone or "Professional & Polite", tone_instructions["Professional & Polite"])
    
    # Build length instructions
    length_instructions = {
        "Quick reply (1–2 lines)": "Keep the reply very brief - 1 to 2 lines maximum. Be concise and direct.",
        "Balanced reply": "Write a balanced reply - 2 to 4 sentences addressing the key points.",
        "Detailed explanation": "Write a detailed reply - 4 to 6 sentences with thorough explanation and context."
    }
    
    length_instruction = length_instructions.get(email.length or "Balanced reply", length_instructions["Balanced reply"])
    
    prompt = f"""Classify and draft a professional email reply.

Classify: Work/Personal/Promotional/Spam
Importance: High/Medium/Low

TONE: {tone_instruction}
LENGTH: {length_instruction}

Write a well-structured, professional email reply. Format it properly as an email, not a casual message.

REQUIRED STRUCTURE:
1. Greeting (Hi/Hello/Dear [Name] or appropriate greeting based on context)
2. Opening line acknowledging their email
3. Main body ({length_instruction})
4. Closing line if needed
5. ALWAYS end with "Regards," on a new line

FORMATTING RULES:
- Use proper email structure with line breaks
- Start with appropriate greeting
- Tone: {tone_instruction}
- No AI-sounding phrases
- Always include "Regards," at the end
- Make it look like a proper email, not a WhatsApp message

Email:
Subject: {email.subject}
From: {email.sender}
Body: {email.body[:1500]}

JSON only:
{{
  "category": "",
  "importance": "",
  "draft": ""
}}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an expert at writing professional, well-structured email replies. Always format emails properly with greeting, body, and 'Regards' closing. Never write casual messages like WhatsApp."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=350  # Increased for proper structure
    )
    
    # Increment reply count
    users_collection.update_one(
        {"email": current_user["email"]},
        {"$inc": {"replies_used_this_month": 1}}
    )

    return {
        "result": response.choices[0].message.content
    }

@app.post("/regenerate-reply")
def regenerate_reply(data: RegenerateInput, current_user: dict = Depends(get_current_user)):
    # Load user and enforce regenerate limits
    user = users_collection.find_one({"email": current_user["email"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Reset monthly counters if needed
    reset_date = datetime.fromisoformat(user.get("monthly_reset_date", (datetime.utcnow() + timedelta(days=30)).isoformat()))
    if datetime.utcnow() > reset_date:
        users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {
                "replies_used_this_month": 0,
                "regenerations_used_this_month": 0,
                "monthly_reset_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
            }}
        )
        user["replies_used_this_month"] = 0
        user["regenerations_used_this_month"] = 0

    # Note: Per-reply regeneration limits are handled on the frontend
    # Backend only tracks monthly limits for analytics, but doesn't block per-reply regenerations
    user_plan = user.get("plan", "free")
    # Build tone instructions
    tone_instructions = {
        "Professional & Polite": "Use a professional, courteous, and respectful tone. Be formal but warm.",
        "Warm & Friendly": "Use a warm, friendly, and approachable tone. Be conversational and personable.",
        "Clear & Direct": "Use a clear, direct, and straightforward tone. Be concise and to the point.",
        "Empathetic": "Use an empathetic, understanding, and compassionate tone. Show care and concern.",
        "Apology": "Use a sincere, apologetic tone. Express genuine regret and offer solutions.",
        "Assertive": "Use a confident, assertive, and decisive tone. Be firm but respectful.",
        "Sales-friendly": "Use a persuasive, enthusiastic, and engaging tone. Highlight benefits and value.",
        "Support / HR": "Use a helpful, supportive, and professional tone. Be clear and solution-oriented."
    }
    
    tone_instruction = tone_instructions.get(data.tone or "Professional & Polite", tone_instructions["Professional & Polite"])
    
    # Build length instructions
    length_instructions = {
        "Quick reply (1–2 lines)": "Keep the reply very brief - 1 to 2 lines maximum. Be concise and direct.",
        "Balanced reply": "Write a balanced reply - 2 to 4 sentences addressing the key points.",
        "Detailed explanation": "Write a detailed reply - 4 to 6 sentences with thorough explanation and context."
    }
    
    length_instruction = length_instructions.get(data.length or "Balanced reply", length_instructions["Balanced reply"])
    
    options_text = ""
    if data.options:
        options_text = "Make it: " + ", ".join(data.options)

    instruction_text = ""
    if data.instruction:
        instruction_text = f"\nAlso: {data.instruction}"

    prompt = f"""Improve this email reply draft while maintaining proper email structure.

Subject: {data.subject}
Original email: {data.original_body[:800]}
Current draft: {data.previous_draft}
{options_text}{instruction_text}

TONE: {tone_instruction}
LENGTH: {length_instruction}

REQUIREMENTS:
- Maintain proper email structure (greeting, body, closing)
- ALWAYS end with "Regards," on a new line
- Tone: {tone_instruction}
- Length: {length_instruction}
- Address the modifications requested
- Make it well-structured, not casual
- Use proper line breaks and formatting

Output ONLY the improved reply text with proper email formatting."""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You improve email replies while maintaining professional email structure. Always include 'Regards' at the end. Format as a proper email, not a casual message."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=350  # Increased for proper structure
    )

    # Increment regenerate count if free user
    if user_plan == "free":
        users_collection.update_one(
            {"email": current_user["email"]},
            {"$inc": {"regenerations_used_this_month": 1}}
    )

    return {
        "draft": response.choices[0].message.content.strip()
    }


class DraftInput(BaseModel):
    subject: str
    body: str

@app.post("/inbox-classify-lite")
def inbox_classify_lite(email: InboxLiteInput):
    prompt = f"""Classify this email into one of three categories: Important, Personal, or Ads.

RULES:
1. **Important**: 
   - Work emails from colleagues, clients, or business contacts
   - Emails with action items, deadlines, or requests
   - Professional communications (even from companies like Google, Microsoft, etc. if they're about account security, important updates)
   - Emails from real people asking for something or requiring a response
   - Examples: "Meeting tomorrow", "Please review", "Your account was accessed", "Invoice attached"

2. **Personal**: 
   - Emails from friends, family, or personal contacts
   - Casual conversations and personal updates
   - Social invitations or personal messages
   - Examples: "Hey, how are you?", "Birthday party", "Catch up soon"

3. **Ads**: 
   - Marketing emails, promotions, sales
   - Newsletters and automated updates
   - Spam or promotional content
   - No action required, just informational
   - Examples: "50% off sale", "Weekly newsletter", "New product launch", "Special offer"

EMAIL TO CLASSIFY:
From: {email.sender}
Subject: {email.subject}

Analyze the sender and subject carefully. If the sender is a company but the subject suggests it's an important account-related email (security, billing, important updates), classify as "Important". If it's clearly promotional or marketing, classify as "Ads".

Respond with ONLY valid JSON, no other text:
{{
  "category": "Important",
  "importance": "High"
}}

Use "Important", "Personal", or "Ads" for category. Use "High" or "Low" for importance."""

    try:
        # Try with response_format first (if supported), otherwise fallback
        try:
            # Try with response_format first (if supported), otherwise fallback
            try:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert email classifier. Analyze sender and subject to determine if an email is Important (work/action needed), Personal (friends/family), or Ads (promotional/marketing). Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more consistent classification
                    max_tokens=50,  # Reduced tokens for faster response
                    response_format={"type": "json_object"}  # Force JSON response if supported
                )
            except Exception as format_error:
                # Fallback if response_format is not supported
                print(f"response_format not supported, using fallback: {format_error}")
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert email classifier. Analyze sender and subject to determine if an email is Important (work/action needed), Personal (friends/family), or Ads (promotional/marketing). Always respond with valid JSON only, no markdown, no code blocks, just the JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50  # Reduced tokens for faster response
                )

        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        # Parse to validate JSON
        import json
        parsed = json.loads(result_text)
        
        # Validate category
        category = parsed.get("category", "Ads")
        if category not in ["Important", "Personal", "Ads"]:
            # Try to normalize
            category_lower = category.lower()
            if "important" in category_lower or "work" in category_lower:
                category = "Important"
            elif "personal" in category_lower or "friend" in category_lower or "family" in category_lower:
                category = "Personal"
            else:
                category = "Ads"
        
        # Validate importance
        importance = parsed.get("importance", "Low")
        if importance not in ["High", "Low"]:
            importance = "Low"
        
        return {
            "result": json.dumps({
                "category": category,
                "importance": importance
            })
        }
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Classification error: {e}")
        print(f"Traceback: {error_details}")
        # Return error response instead of silently falling back
        return {
            "result": json.dumps({
                "category": "Ads",
                "importance": "Low",
                "error": str(e)
            })
        }


'''
@app.post("/save-draft")
def save_draft(data: DraftInput):
    service = get_gmail_service()

    message = MIMEText(data.body)
    message["subject"] = f"Re: {data.subject}"

    raw_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode("utf-8")

    draft = {
        "message": {
            "raw": raw_message
        }
    }

    service.users().drafts().create(
        userId="me",
        body=draft
    ).execute()

    return {"status": "draft saved"}
'''

# ---------- Run directly ----------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


'''
import base64
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def get_gmail_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    service = build("gmail", "v1", credentials=creds)
    return service
'''