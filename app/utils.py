import os
import re
import aiofiles
from typing import List

def clean_text(s: str) -> str:
    if not s: 
        return ""
    s = s.replace("\r", " ").replace("\u00a0", " ")
    s = "\n".join(line.strip() for line in s.splitlines())
    s = " ".join(s.split())
    return s.strip()

async def save_uploaded_file(file, file_id: str) -> str:
    file_path = f"data/uploaded_files/{file_id}.pdf"
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    return file_path

def generate_file_id() -> str:
    import uuid
    return str(uuid.uuid4())