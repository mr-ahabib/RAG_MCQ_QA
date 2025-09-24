import os
import re
import numpy as np
import faiss
import PyPDF2
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from huggingface_hub import login

from .models import Chunk, Retrieval, rag_state
from .utils import clean_text  

HF_TOKEN = ""



def initialize_models():
    if rag_state.embedder is None:
        rag_state.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    if rag_state.llm is None:
        if HF_TOKEN:
            login(token=HF_TOKEN)

        model_name = "meta-llama/Llama-3.2-1B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        rag_state.llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        rag_state.tokenizer = tokenizer



def safe_truncate(text: str, max_chars=3000) -> str:
    """Truncate text to avoid exceeding model input limits."""
    return text[:max_chars]


def clean_qa_output(raw: str) -> str:
    """Ensure model output stays in strict Q/A format only."""
    qa_pairs = re.findall(r"(Q\d+:.*?A\d+:.*?)(?=Q\d+:|$)", raw, flags=re.S)
    if not qa_pairs:  
        return raw.strip()
    return "\n\n".join(p.strip() for p in qa_pairs)

def extract_pages(pdf_path: str) -> List[str]:
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(clean_text(txt))
    return pages


def chunk_pages(pages: List[str], chunk_size=180, overlap=30) -> List[Chunk]:
    chunks, cid = [], 0
    for p_idx, page in enumerate(pages):
        words = page.split()
        if not words:
            continue
        for i in range(0, len(words), chunk_size - overlap):
            window = words[i:i + chunk_size]
            text = " ".join(window)
            if not text:
                continue
            chunks.append(Chunk(cid, p_idx + 1, text, len(window), len(text)))
            cid += 1
    return chunks


def build_index(chunks: List[Chunk]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    texts = [c.text for c in chunks]
    embeddings = rag_state.embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=True
    ).astype("float32")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index, embeddings


def local_llm_generate(prompt: str, max_tokens=250, temperature=0.2) -> str:
    formatted_prompt = (
        f"<|system|>\nYou are a helpful AI assistant.</s>\n"
        f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    )

    response = rag_state.llm(
        formatted_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=rag_state.tokenizer.eos_token_id
    )

    return response[0]['generated_text'].replace(formatted_prompt, "").strip()


def generate_content(pages: List[str], chunks: List[Chunk], mode: str = "qa") -> str:
    """
    Generate either QA pairs or MCQs depending on mode.
    mode = "qa" -> Generates 10 Question & Answer pairs
    mode = "mcq" -> Generates 10 MCQs with 4 options each
    """
    sample_text = "\n".join(c.text for c in chunks[:5])
    sample_text = safe_truncate(sample_text, max_chars=3000)

    if mode == "qa":
        prompt = f"""Generate 10 question and answer pairs strictly from the following text.
Do NOT invent anything beyond this text. 

Text:
{sample_text}

Respond ONLY in this format:
Q1: ...
A1: ...
Q2: ...
A2: ...
...
Q10: ...
A10: ..."""

    elif mode == "mcq":
        prompt = f"""Generate 10 multiple choice questions (MCQs) strictly from the following text.
Each question should have 4 options (A-D). 
Only one option must be correct. 
Mark the correct answer clearly with (Correct).

Text:
{sample_text}

Respond ONLY in this format:
Q1: ...
A) ...
B) ...
C) ...
D) ...
Answer: (Correct: X)

Q2: ...
A) ...
B) ...
C) ...
D) ...
Answer: (Correct: Y)

...
Q10: ...
A) ...
B) ...
C) ...
D) ...
Answer: (Correct: Z)"""

    else:
        return "Invalid mode. Use 'qa' or 'mcq'."

    raw = local_llm_generate(prompt, max_tokens=800, temperature=0.2)
    return raw.strip()


def retrieve(query: str, index, chunks: List[Chunk], k=3) -> List[Retrieval]:
    q = rag_state.embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q, min(k, len(chunks)))
    rets = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0:
            continue
        rets.append(Retrieval(chunks[idx], float(dist)))
    return rets


def answer_with_rag(query: str, rets: List[Retrieval]) -> str:
    ctx = "\n\n".join(f"[p{r.chunk.page}] {r.chunk.text}" for r in rets)

    prompt = f"""Based only on the retrieved context below, answer the question. 
You are a **RAG assistant**. 
Do not add extra details beyond the context.

Context:
{ctx}

Question: {query}

Answer:"""

    return local_llm_generate(prompt, max_tokens=250, temperature=0.2)
