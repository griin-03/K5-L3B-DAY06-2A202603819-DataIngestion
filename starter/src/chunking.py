"""Chunking strategies: Fixed-size and Structure/Paragraph-aware."""
import re
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split page text into fixed-size overlapping character windows."""
    text = page.text
    step = size - overlap
    chunks: list[Chunk] = []
    
    if not text:
        return chunks
        
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"fixed_{page.document_id}_p{page.page}_{chunk_index}"
            chunk_metadata = {
                **page.metadata,
                "strategy": "fixed",
                "chunk_size": len(chunk_text),
                "window_start": start,
                "window_end": end
            }
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=chunk_metadata
            ))
            chunk_index += 1
            
        if end >= len(text):
            break
        start += step
        
    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Split page text by natural paragraphs, respecting headings and boundaries."""
    # Split text on double newlines or newline followed by heading/bullet
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', page.text) if p.strip()]
    
    chunks: list[Chunk] = []
    current_buffer: list[str] = []
    current_len = 0
    chunk_index = 0
    
    def flush_buffer():
        nonlocal current_buffer, current_len, chunk_index
        if current_buffer:
            combined = "\n\n".join(current_buffer).strip()
            chunk_id = f"para_{page.document_id}_p{page.page}_{chunk_index}"
            chunk_metadata = {
                **page.metadata,
                "strategy": "paragraph",
                "chunk_size": len(combined)
            }
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=combined,
                metadata=chunk_metadata
            ))
            chunk_index += 1
            current_buffer = []
            current_len = 0

    for para in raw_paras:
        # If single paragraph exceeds max_size, split by sentences
        if len(para) > max_size:
            flush_buffer()
            # Split by sentence end punctuation
            sentences = re.split(r'(?<=[.?!])\s+', para)
            sent_buf: list[str] = []
            sent_len = 0
            for s in sentences:
                if sent_len + len(s) > max_size and sent_buf:
                    joined_s = " ".join(sent_buf).strip()
                    chunk_id = f"para_{page.document_id}_p{page.page}_{chunk_index}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=joined_s,
                        metadata={**page.metadata, "strategy": "paragraph", "chunk_size": len(joined_s)}
                    ))
                    chunk_index += 1
                    sent_buf = []
                    sent_len = 0
                sent_buf.append(s)
                sent_len += len(s) + 1
            if sent_buf:
                joined_s = " ".join(sent_buf).strip()
                chunk_id = f"para_{page.document_id}_p{page.page}_{chunk_index}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=joined_s,
                    metadata={**page.metadata, "strategy": "paragraph", "chunk_size": len(joined_s)}
                ))
                chunk_index += 1
        else:
            if current_len + len(para) > max_size:
                flush_buffer()
            current_buffer.append(para)
            current_len += len(para) + 2

    flush_buffer()
    return chunks
