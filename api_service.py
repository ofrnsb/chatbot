from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from basic_rag import RAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Danfoss RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=os.getenv("ALLOWED_METHODS", "POST").split(","),
    allow_headers=os.getenv("ALLOWED_HEADERS", "*").split(","),
)

API_KEY = os.getenv("API_KEY", "secret-key") 

class ChatRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    source: str = Field(..., example="ECL Comfort Manual")
    page: str = Field(..., example="23")  

class ChatResponse(BaseModel):
    answer: str = Field(..., example="DEVIflex 6T is a flexible cable...")
    sources: list[SourceDocument]

rag_system = RAGSystem()

def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    _: bool = Depends(verify_api_key)
):
    try:
        result = rag_system.rag_chain.invoke({
            "input": request.question,
            "language": rag_system._detect_language(request.question)
        })
        sources = []
        for doc in result["context"]:
            page = doc.metadata.get("page", "N/A")
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": str(page) if page != "N/A" else page
            })
        return {
            "answer": result["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", "8000"))
    )