# Virtual Mentor 🤖

AI-powered academic mentor with personalized guidance.

## Features
- Chat-based AI mentor
- JWT Authentication
- File upload (PDF support)
- RAG using ChromaDB

## Tech Stack
- FastAPI
- Gemini API
- ChromaDB
- HTML + Tailwind CSS

## Setup
1. Install dependencies:
   pip install -r requirements.txt

2. Create .env file:
   GEMINI_API_KEY=your_key
   JWT_SECRET_KEY=your_secret

3. Run:
   uvicorn main:app --reload