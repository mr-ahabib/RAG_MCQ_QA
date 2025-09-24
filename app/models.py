from dataclasses import dataclass
from typing import List


@dataclass
class Chunk: 
    id:int
    page: int
    text: str
    n_words: int
    n_chars: int

@dataclass
class Retrieval:
    chunk: Chunk
    distance: float


class RAGState:
    def __init__(self):
        self.index=None
        self.chunks=None
        self.pages=None
        self.embedder=None
        self.llm=None


rag_state=RAGState()