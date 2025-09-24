from pydantic import BaseModel
from typing import List, Optional

class PDFUploadResponse(BaseModel):
    message: str
    generated_content: str
    file_id: str

class QuestionRequest(BaseModel):
    question: str
    file_id: str

class QuestionResponse(BaseModel):
    answer: str