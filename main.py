# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

app = FastAPI(title="AI Text Suite API")
logger = logging.getLogger(__name__)

# CORS - UPDATE with your frontend URL after deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-txt-suite-frontend.vercel.app",  # Your exact URL
        "https://ai-txt-suite-frontend-git-main-joseph-otienos-projects-ec384454.vercel.app",  # Preview URLlace with actual URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found")
    raise ValueError("OPENAI_API_KEY not found")
    
client = OpenAI(api_key=OPENAI_API_KEY)

# ==================== MODELS ====================

class RedactRequest(BaseModel):
    text: str
    tone: str
    dialect: str

class SummarizeRequest(BaseModel):
    text: str
    length: str
    bullet_points: Optional[bool] = False

class BlogRequest(BaseModel):
    topic: str
    word_count: int = 400
    tone: str = "Professional"
    audience: str = "General"
    advanced_options: Optional[dict] = None

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "AI Text Suite API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_key_configured": bool(OPENAI_API_KEY)}

@app.post("/api/redact")
async def redact_text(request: RedactRequest):
    logger.info(f"Redact request - Tone: {request.tone}, Dialect: {request.dialect}")
    
    try:
        prompt = f"""
Rewrite the following text in a {request.tone} tone using {request.dialect} English dialect.
Keep the original meaning but adjust the style appropriately.

Text: {request.text}

Rewritten text:
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        
        return {
            "success": True,
            "original": request.text,
            "redacted": result,
            "tone": request.tone,
            "dialect": request.dialect,
            "original_word_count": len(request.text.split()),
            "redacted_word_count": len(result.split())
        }
        
    except Exception as e:
        logger.error(f"Redact error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_text(request: SummarizeRequest):
    logger.info(f"Summarize request - Length: {request.length}")
    
    try:
        length_map = {
            "Short": "in 1-2 concise sentences",
            "Medium": "in 3-4 sentences covering key points",
            "Detailed": "comprehensively with all important details"
        }
        length_instruction = length_map.get(request.length, "in a concise manner")
        
        bullet_instruction = "Format the summary as bullet points." if request.bullet_points else ""
        
        prompt = f"""
Summarize the following text {length_instruction}. {bullet_instruction}

Text: {request.text}

Summary:
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        
        original_count = len(request.text.split())
        summary_count = len(result.split())
        compression = (summary_count / original_count) * 100 if original_count > 0 else 0
        
        return {
            "success": True,
            "summary": result,
            "original_word_count": original_count,
            "summary_word_count": summary_count,
            "compression_ratio": f"{compression:.1f}%",
            "bullet_points": request.bullet_points
        }
        
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blog")
async def generate_blog(request: BlogRequest):
    logger.info(f"Blog request - Topic: {request.topic}")
    
    if not request.topic or request.topic.strip() == "":
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    try:
        structure = ""
        if request.advanced_options:
            if request.advanced_options.get("intro"):
                structure += "- Include a compelling introduction\n"
            if request.advanced_options.get("list"):
                structure += "- Include a numbered or bulleted list\n"
            if request.advanced_options.get("conclusion"):
                structure += "- End with a strong conclusion\n"
            if request.advanced_options.get("cta"):
                structure += "- Include a call-to-action at the end\n"
        
        prompt = f"""
Write a {request.word_count}-word blog post about "{request.topic}".
Tone: {request.tone}
Target Audience: {request.audience}

Structure requirements:
{structure if structure else "No specific structure requirements."}

Please write an engaging, well-structured blog post with:
1. A catchy title
2. Introduction
3. Main content with clear sections
4. Conclusion

Blog post:
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        word_count = len(result.split())
        reading_time = max(1, round(word_count / 200))
        
        return {
            "success": True,
            "blog_post": result,
            "word_count": word_count,
            "reading_time": reading_time,
            "topic": request.topic,
            "tone": request.tone,
            "audience": request.audience
        }
        
    except Exception as e:
        logger.error(f"Blog error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== THIS IS CRITICAL FOR RENDER ====================
if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 to accept all connections
    uvicorn.run(app, host="0.0.0.0", port=port)
