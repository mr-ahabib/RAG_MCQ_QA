import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.params import Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import asyncio
from fastapi import Query
from . import rag
from .models import RAGState, rag_state
from .schemas import PDFUploadResponse, QuestionRequest, QuestionResponse
from .utils import save_uploaded_file, generate_file_id

#FastAPI app
app = FastAPI(title="PDF RAG System", version="1.0.0")

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#In-memory storage for file states (in production, use a database)
file_states: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup_event():
    #Initialize models on startup
    rag.initialize_models()

@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...), 
    mode: str = Query("qa", enum=["qa", "mcq"])   
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id = generate_file_id()
    file_path = await save_uploaded_file(file, file_id)
    
    try:
        pages = rag.extract_pages(file_path)
        chunks = rag.chunk_pages(pages)
        index, _ = rag.build_index(chunks)
        
        # Generate based on selected mode
        generated = rag.generate_content(pages, chunks, mode=mode)
        
        file_states[file_id] = {
            "index": index,
            "chunks": chunks,
            "pages": pages,
            "mode": mode
        }
        
        return PDFUploadResponse(
            message=f"PDF loaded and indexed successfully! Generated {mode.upper()} content.",
            generated_content=generated,   
            file_id=file_id
        )
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if request.file_id not in file_states:
        raise HTTPException(status_code=404, detail="File not found. Please upload a PDF first.")
    
    state = file_states[request.file_id]
    
    try:
        # Retrieve relevant chunks
        rets = rag.retrieve(
            request.question, 
            state["index"], 
            state["chunks"], 
            k=3
        )
        
        # Generate answer
        answer = rag.answer_with_rag(request.question, rets)
        
        return QuestionResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": rag_state.embedder is not None and rag_state.llm is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)