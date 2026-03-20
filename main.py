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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Text Suite API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app",  # Your Vercel URL
        "https://your-frontend-git-main.vercel.app",  # Preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not found")
    raise ValueError("OPENAI_API_KEY not found")
else:
    logger.info(f"✅ OpenAI API key loaded (first 5 chars: {OPENAI_API_KEY[:5]}...)")

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

# ==================== HELPER FUNCTION ====================

def get_llm(temperature=0.7, max_tokens=1000):
    """Create LLM instance safely"""
    try:
        # Try ChatOpenAI first (recommended)
        return OpenAI(
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=max_tokens,
            model="gpt-4o-mini"  # or "gpt-4" if you have access
        )
    except Exception as e:
        logger.error(f"Error creating OpenAI: {e}")
        # Fallback to older method if needed
        from langchain.llms import OpenAI
        return OpenAI(
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=max_tokens
        )

# ==================== TEMPLATES ====================

REDACT_TEMPLATE = """
Below is a draft text that may be poorly worded.
Your goal is to:
- Properly redact the draft text
- Convert the draft text to a specified tone
- Convert the draft text to a specified dialect

Here are examples of different Tones:
- Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
- Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.

Here are examples of words in different dialects:
- American: French Fries, cotton candy, apartment, garbage, cookie, green thumb, parking lot, pants, windshield
- British: chips, candyfloss, flag, rubbish, biscuit, green fingers, car park, trousers, windscreen

Please start the redaction with a warm introduction if needed.

Below is the draft text, tone, and dialect:
DRAFT: {draft}
TONE: {tone}
DIALECT: {dialect}

YOUR {dialect} RESPONSE:
"""

SUMMARIZE_TEMPLATE = """
Please summarize the following text with a {length} level of detail.
{focus_instruction}

Text to summarize:
{text}

Summary:
"""

BLOG_TEMPLATE = """
IMPORTANT: You MUST write about the exact topic provided below.

Topic: {topic}
Word count: {word_count} words
Tone: {tone}
Target Audience: {audience}
{structure_instructions}

CRITICAL: Write ONLY about "{topic}". Do NOT write about AI, technology, or any other topic.

Now write a {word_count}-word blog post about "{topic}" with:
1. A catchy title about {topic}
2. Introduction about {topic}
3. Main content with clear sections about {topic}
4. Conclusion about {topic}

Write the blog post:
"""

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "AI Text Suite API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "api_key_configured": bool(OPENAI_API_KEY)
    }

@app.post("/api/redact")
async def redact_text(request: RedactRequest):
    """Rewrite text with specified tone and dialect"""
    logger.info(f"📝 Redact request - Text: {request.text[:100]}... Tone: {request.tone}, Dialect: {request.dialect}")
    
    try:
        llm = get_llm(temperature=0.7, max_tokens=1000)
        
        prompt = PromptTemplate(
            input_variables=["tone", "dialect", "draft"],
            template=REDACT_TEMPLATE,
        )
        
        formatted_prompt = prompt.format(
            tone=request.tone,
            dialect=request.dialect,
            draft=request.text
        )
        
        logger.info(f"📤 Sending prompt to OpenAI")
        result = llm.invoke(formatted_prompt)
        
        # Handle different response types
        if hasattr(result, 'content'):
            result_text = result.content
        else:
            result_text = str(result)
        
        logger.info(f"✅ Redact successful")
        
        return {
            "success": True,
            "original": request.text,
            "redacted": result_text.strip(),
            "tone": request.tone,
            "dialect": request.dialect,
            "original_word_count": len(request.text.split()),
            "redacted_word_count": len(result_text.split())
        }
        
    except Exception as e:
        logger.error(f"❌ Redact error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_text(request: SummarizeRequest):
    """Summarize text with length control"""
    logger.info(f"📝 Summarize request - Text length: {len(request.text)} chars")
    
    try:
        llm = get_llm(temperature=0.3, max_tokens=500)
        
        # Build focus instruction
        focus_instruction = ""
        if request.length == "Short":
            focus_instruction = "Provide a very concise summary in 1-2 sentences."
        elif request.length == "Medium":
            focus_instruction = "Provide a balanced summary with key points in 3-4 sentences."
        else:
            focus_instruction = "Provide a comprehensive summary with all important details."
        
        if request.bullet_points:
            focus_instruction += " Format the summary as bullet points."
        
        prompt = PromptTemplate(
            input_variables=["length", "focus_instruction", "text"],
            template=SUMMARIZE_TEMPLATE,
        )
        
        formatted_prompt = prompt.format(
            length=request.length,
            focus_instruction=focus_instruction,
            text=request.text
        )
        
        # For long text, use map_reduce chain
        if len(request.text.split()) > 1000:
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            texts = text_splitter.split_text(request.text)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            result = chain.run(docs)
            result_text = result
        else:
            result = llm.invoke(formatted_prompt)
            if hasattr(result, 'content'):
                result_text = result.content
            else:
                result_text = str(result)
        
        original_count = len(request.text.split())
        summary_count = len(result_text.split())
        compression = (summary_count / original_count) * 100 if original_count > 0 else 0
        
        return {
            "success": True,
            "summary": result_text.strip(),
            "original_word_count": original_count,
            "summary_word_count": summary_count,
            "compression_ratio": f"{compression:.1f}%",
            "bullet_points": request.bullet_points
        }
        
    except Exception as e:
        logger.error(f"❌ Summarize error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/blog")
async def generate_blog(request: BlogRequest):
    """Generate blog post with customization"""
    logger.info(f"📝 Blog request - Topic: {request.topic}, Word count: {request.word_count}")
    
    # Verify topic is not empty
    if not request.topic or request.topic.strip() == "":
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    try:
        llm = get_llm(temperature=0.7, max_tokens=2048)
        
        # Build structure instructions
        structure_instructions = ""
        if request.advanced_options:
            if request.advanced_options.get("intro"):
                structure_instructions += "- Include a compelling introduction\n"
            if request.advanced_options.get("list"):
                structure_instructions += "- Include a numbered or bulleted list\n"
            if request.advanced_options.get("conclusion"):
                structure_instructions += "- End with a strong conclusion\n"
            if request.advanced_options.get("cta"):
                structure_instructions += "- Include a call-to-action at the end\n"
        
        if not structure_instructions:
            structure_instructions = "No special structure requirements."
        
        prompt = PromptTemplate(
            input_variables=["topic", "word_count", "tone", "audience", "structure_instructions"],
            template=BLOG_TEMPLATE,
        )
        
        formatted_prompt = prompt.format(
            topic=request.topic,
            word_count=request.word_count,
            tone=request.tone,
            audience=request.audience,
            structure_instructions=structure_instructions
        )
        
        logger.info(f"📤 Sending prompt for topic: {request.topic}")
        
        result = llm.invoke(formatted_prompt)
        
        # Handle different response types
        if hasattr(result, 'content'):
            blog_text = result.content
        else:
            blog_text = str(result)
        
        word_count = len(blog_text.split())
        reading_time = max(1, round(word_count / 200))
        
        logger.info(f"✅ Blog generated - {word_count} words")
        
        return {
            "success": True,
            "blog_post": blog_text.strip(),
            "word_count": word_count,
            "reading_time": reading_time,
            "topic": request.topic,
            "tone": request.tone,
            "audience": request.audience
        }
        
    except Exception as e:
        logger.error(f"❌ Blog error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test")
async def test_endpoint():
    """Test endpoint to verify OpenAI connection"""
    try:
        llm = get_llm(temperature=0.5, max_tokens=50)
        result = llm.invoke("Say 'Hello World' in a professional tone.")
        
        if hasattr(result, 'content'):
            response_text = result.content
        else:
            response_text = str(result)
            
        return {"success": True, "result": response_text}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)