from fastapi import FastAPI, UploadFile, File, Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from app.face_engine import extract_embedding
from app.faiss_manager import get_faiss_manager
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

API_KEY = os.getenv("FACE_API_KEY", "secret_face_api_key")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Similarity threshold for face matching (60%)
SIMILARITY_THRESHOLD = 0.60

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    embedding: List[float]

class RebuildResponse(BaseModel):
    success: bool
    message: str

# Startup event removed - FAISS index will load on first use
# This prevents Railway deployment from hanging on startup
# The index will be built/loaded when first needed

@app.get("/")
def home():
    return {"message": "FastAPI Face Recognition Service Running"}

@app.post("/extract-embedding")
async def generate_embedding(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """Extract face embedding from uploaded image."""
    image_bytes = await file.read()
    embedding, error = extract_embedding(image_bytes)

    if error:
        return {"success": False, "message": error}

    return {"success": True, "embedding": embedding}

@app.post("/search-face")
async def search_face(request: SearchRequest, api_key: str = Depends(get_api_key)):
    """
    Search for most similar face using FAISS.
    
    Args:
        request: SearchRequest with embedding (512-dim array)
    
    Returns:
        Best match with student_id, distance, and similarity score
    """
    try:
        manager = get_faiss_manager()
        result = manager.search(request.embedding, k=1)
        
        if result is None:
            # Try to rebuild index
            print("FAISS search failed, attempting rebuild...")
            success, msg = manager.build_index_from_db()
            if success:
                # Retry search
                result = manager.search(request.embedding, k=1)
        
        if result is None:
            return {
                "success": False,
                "message": "No match found or index error"
            }
        
        # Check similarity threshold before accepting match
        if result["similarity"] < SIMILARITY_THRESHOLD:
            print(f"Match rejected: similarity {result['similarity']:.3f} below threshold {SIMILARITY_THRESHOLD}")
            return {
                "success": False,
                "message": "No matching student found",
                "similarity": result["similarity"]
            }
        
        student = manager.get_student_by_id(result["student_id"])
        print(f"Match accepted: similarity {result['similarity']:.3f}, student ID {result['student_id']}")
        return {
            "success": True,
            "student": student,
            "similarity": result["similarity"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Search error: {str(e)}"
        }

@app.post("/rebuild-index")
async def rebuild_index(api_key: str = Depends(get_api_key)):
    """
    Rebuild FAISS index from PostgreSQL database.
    
    This endpoint fetches all students with embeddings from the database
    and rebuilds the FAISS index.
    """
    try:
        manager = get_faiss_manager()
        success, message = manager.build_index_from_db()
        
        return {
            "success": success,
            "message": message
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Rebuild error: {str(e)}"
        }

@app.get("/faiss-status")
async def faiss_status(api_key: str = Depends(get_api_key)):
    """
    Get FAISS index status and health information.
    
    Returns:
        Index statistics and metadata
    """
    try:
        manager = get_faiss_manager()
        status = manager.get_status()
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Status error: {str(e)}"
        }