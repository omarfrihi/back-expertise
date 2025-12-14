from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import fact_check

app = FastAPI(title="Fake News ML API")

# Add CORS middleware
origins = [
    "http://localhost:8080",  # frontend URL
    "http://127.0.0.1:8080",
    "*",  # allows all origins (optional, not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # allowed origins
    allow_credentials=True,
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)

class FactCheckRequest(BaseModel):
    text: str

class FactCheckResponse(BaseModel):
    verdict: str
    confidence: float
    chatgpt_found: bool
    justification: str

@app.post("/fact-check", response_model=FactCheckResponse)
def check_news(request: FactCheckRequest):
    return fact_check(request.text)

@app.get("/health")
def health():
    return {"status": "ok"}
