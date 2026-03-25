import os
import json
import asyncio
import httpx
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Header
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import passlib.context
import jwt
from datetime import datetime, timedelta, timezone

# RAG imports
import chromadb
from chromadb.utils import embedding_functions
import pypdf
import io

# Load environment variables
load_dotenv()

# --- INITIALIZATION ---
app = FastAPI(title="Virtual Mentor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE LOGIC (WITH CONCURRENCY FIX) ---
current_dir = Path(__file__).parent
DB_FILE = current_dir / "database.json"
# 2. Concurrency fix: Add an asyncio.Lock to prevent race conditions during DB read/write
db_lock = asyncio.Lock()

async def read_db():
    async with db_lock:
        if not os.path.exists(DB_FILE):
            initial = {"users": {}}
            with open(DB_FILE, "w") as f:
                json.dump(initial, f, indent=4)
            return initial
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Addressed bare except block
            return {"users": {}}

async def write_db(data):
    async with db_lock:
        with open(DB_FILE, "w") as f:
            json.dump(data, f, indent=4)

# --- SECURITY (PASSWORD HASHING & JWT) ---
# 1. Security fix: Hashed passwords & proper JWT tokens instead of returning raw usernames
pwd_context = passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto")

# 1. Security fix: Removing hardcoded fallback JWT secret
SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.urandom(32).hex())
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": int(expire.timestamp())})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 5. Code duplication fix: Use FastAPI dependency injection for fetching user via token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        # Catching specific JWT errors
        raise credentials_exception
    
    db = await read_db()
    user = db.get("users", {}).get(username)
    if user is None:
        raise credentials_exception
    return {"username": username, **user}

# --- MODELS ---
class UserRegister(BaseModel):
    username: str
    password: str
    major: str
    goals: List[str]

class UserLogin(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str

# --- AI CONFIGURATION & CHROMA DB ---
# read the Gemini API key from an environment variable.  The original code
# mistakenly passed a literal key string to `os.getenv` which guaranteed the
# warning would fire and left the application unable to reach the AI service.
#
# Users can create a `.env` file in the project root containing:
#
#     GEMINI_API_KEY=your_real_key_here
#
# or set the variable in their shell before launching the server.
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # not fatal; the app still runs but will return fallback text for AI calls
    print("WARNING: GEMINI_API_KEY environment variable not set. AI features will operate in offline/fallback mode.")

MODEL_ID = "gemini-2.5-flash-preview-09-2025"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=str(current_dir / "chroma_db"))

# Create embedding function linked to Gemini
# using the recommended text embedding model
if API_KEY:
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY,
        model_name="models/text-embedding-004"
    )
else:
    gemini_ef = embedding_functions.DefaultEmbeddingFunction()

# Create or get the collection where our text chunks will live
collection = chroma_client.get_or_create_collection(
    name="mentor_knowledge", 
    embedding_function=gemini_ef
)

# Text chunking helper
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        # If we are not at the very end, try to find a natural break point (newline or space)
        if end < text_length:
            natural_break = text.rfind('\n', start, end)
            if natural_break == -1:
                natural_break = text.rfind('. ', start, end)
            if natural_break != -1 and natural_break > start + chunk_size // 2:
                end = natural_break + 1
                
        chunks.append(text[start:end])
        start = end - overlap
        
        # Prevent infinite loops if natural break logic fails
        if start <= 0 or start >= end:
             start = end
             
    return chunks

# --- HELPER: RESILIENT RETRY LOGIC ---
async def generate_with_retry(prompt_text, system_instruction="", is_plan=False):
    if not API_KEY:
        return "AI capabilities are currently disabled due to missing API key."

    retries = 5
    delays = [2, 5, 10, 15, 20] 
    
    payload = {
        "contents": [{
            "parts": [{"text": f"SYSTEM ROLE: {system_instruction}\n\nSTUDENT INPUT: {prompt_text}"}]
        }]
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        for i in range(retries):
            try:
                response = await client.post(
                    ENDPOINT, 
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                
                if response.status_code == 429:
                    print(f"⚠️ Rate limit (429). Retrying in {delays[i]}s...")
                    await asyncio.sleep(delays[i])
                    continue
                
                print(f"❌ API Error {response.status_code}: {response.text}")
                break

            except Exception as e:
                print(f"❌ Request Exception: {str(e)}")
                await asyncio.sleep(delays[i])

    if is_plan:
        return "### 📅 Study Roadmap (Offline Mode)\n- AI limits reached. Please review your core syllabus and goals."
    return "I am currently processing many requests. Please try again in 60 seconds!"

# --- API ROUTES ---

# mount a static directory for any additional client assets (css, js, images)
# this keeps the root route free for the SPA, but makes `/static/*` available
static_dir = current_dir  # using project root for now; refine later if needed
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# serve the SPA frontend when visiting the root path
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # FastAPI will handle this file directly from disk.  If you later move the
    # frontend into a `static` or `templates` subdirectory you can adjust the
    # path accordingly.
    return FileResponse(current_dir / "index.html")

# optional health check route so automated tests can still verify API status
@app.get("/api/status")
def api_status():
    return {"status": "Virtual Mentor API is Online", "docs": "/docs"}

@app.post("/auth/register")
async def register(user: UserRegister):
    db = await read_db()
    
    # 3. Logic bug fix: Prevent overwriting existing user registrations
    if user.username in db["users"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username already registered"
        )
        
    db["users"][user.username] = {
        # Security fix: Hashing the password before storing
        "password": get_password_hash(user.password),
        "major": user.major,
        "goals": user.goals
    }
    # Removed user_files logic
    await write_db(db)
    
    return {"message": "Registration successful"}

@app.post("/auth/login")
async def login(user_data: UserLogin):
    db = await read_db()
    user = db["users"].get(user_data.username)
    
    # Compare with hashed password
    if not user or not verify_password(user_data.password, str(user.get("password", ""))):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Security fix: Send a JWT instead of the raw username
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    # 5. Code duplication fix: Handled via the Depends(get_current_user)
    return {
        "username": current_user["username"], 
        "major": current_user.get("major"), 
        "goals": current_user.get("goals")
    }

@app.post("/mentor/upload-context")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    try:
        # Read file dynamically based on extension
        text_content = ""
        
        # Read exactly what the user sent
        content = await file.read()
        
        if file.filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
        else:
            text_content = content.decode('utf-8', errors='ignore')
            
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file.")
            
        # 1. First, delete any old vector chunks belonging to this user
        collection.delete(where={"username": username})
        
        # 2. Chunk the new text
        chunks = chunk_text(text_content, chunk_size=1000, overlap=200)
        
        # 3. Insert new chunks into ChromaDB
        # Generate unique IDs for each chunk
        ids = [f"{username}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"username": username, "source": file.filename} for _ in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return {"message": f"Successfully processed {file.filename}. Sliced into {len(chunks)} contextual snippets and stored in vector DB."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/mentor/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    
    # RAG Retrieval Step
    # Query Chroma DB for the top 3 chunks closest to the user's question, filtering by username
    results = collection.query(
        query_texts=[request.message],
        n_results=3,
        where={"username": username}
    )
    
    file_context = ""
    docs = results.get('documents')
    if docs and isinstance(docs, list) and len(docs) > 0 and docs[0]:
        retrieved_chunks = docs[0]
        file_context = "\n\n---\n\n".join(retrieved_chunks)

    system_msg = (
        f"You are 'Virtual Mentor', an expert academic advisor. "
        f"Student: {username}. Major: {current_user.get('major', 'General')}. "
        f"Goals: {current_user.get('goals', [])}. "
        "INSTRUCTIONS: Use the provided document context below to answer accurately. "
        "The context consists of specific retrieved snippets from the user's uploaded syllabus or notes."
    )
    
    if file_context:
        system_msg += f"\n\n--- RELEVANT UPLOADED DOCUMENT SNIPPETS ---\n{file_context}\n--- END OF SNIPPETS ---"

    answer = await generate_with_retry(request.message, system_msg)
    return {"mentor_response": answer}

@app.get("/mentor/plan")
async def get_plan(current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    goals_str = " ".join(current_user.get('goals', []))
    
    prompt = f"Create a weekly study roadmap for {username} majoring in {current_user.get('major', 'General')}. Focus on: {goals_str}."

    # RAG Retrieval Step - use their goals to seek out relevant syllabus context
    results = collection.query(
        query_texts=[f"Syllabus roadmap pacing schedule topics for {goals_str}"],
        n_results=3,
        where={"username": username}
    )
    
    docs = results.get('documents')
    if docs and isinstance(docs, list) and len(docs) > 0 and docs[0]:
        retrieved_chunks = docs[0]
        file_context = "\n\n---\n\n".join(retrieved_chunks)
        prompt += f"\n\nIncorporate details from these relevant class syllabus snippets:\n{file_context}"
        
    plan = await generate_with_retry(prompt, "You are a professional career coach.", is_plan=True)
    return {"weekly_plan": plan}