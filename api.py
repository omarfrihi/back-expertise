from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import fact_check

# =====================================================
# 1. Create FastAPI app
# =====================================================
app = FastAPI(
    title="Fake News ML API",
    version="1.0.0"
)

# =====================================================
# 2. CORS configuration
# =====================================================
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # DO NOT use "*" with credentials
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 3. Request / Response schemas
# =====================================================
class FactCheckRequest(BaseModel):
    text: str

class FactCheckResponse(BaseModel):
    verdict: str            # ML verdict: TRUE / FALSE / UNCERTAIN
    confidence: float       # ML confidence
    llm_verdict: str        # LLM verdict
    justification: str      # LLM explanation

# =====================================================
# 4. API endpoints
# =====================================================
@app.post("/fact-check", response_model=FactCheckResponse)
def check_news(request: FactCheckRequest):
    """
    Fact-check a claim using:
    - ML model for verdict + confidence
    - LLM for justification
    """
    return fact_check(request.text)

@app.get("/health")
def health():
    return {"status": "ok"}
