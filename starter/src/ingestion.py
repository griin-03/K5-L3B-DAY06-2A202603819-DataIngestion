"""Data ingestion and normalization for PDF documents."""
import re
import unicodedata
from pathlib import Path
import pypdf
from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving textual semantics."""
    if not text:
        return ""
    # NFC Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Rejoin words broken across lines by hyphens (e.g., 'trans-\nformer' -> 'transformer')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace non-breaking and unusual spaces with normal space
    text = text.replace('\u00a0', ' ').replace('\u200b', '')
    # Convert consecutive whitespace characters into a single space while keeping paragraph breaks
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def extract_section_for_page(page_num: int, page_text: str, current_section: str) -> str:
    """Determine the top-level section of the survey based on page text or carry-over."""
    section_patterns = [
        (r'1\.\s+Introduction', '1. Introduction'),
        (r'2\.\s+Background', '2. Background'),
        (r'3\.\s+Large Language Models', '3. Large Language Models'),
        (r'4\.\s+Model Configurations', '4. Model Configurations'),
        (r'5\.\s+Datasets and Evaluation', '5. Datasets and Evaluation'),
        (r'6\.\s+Applications', '6. Applications'),
        (r'7\.\s+Challenges and Future Directions', '7. Challenges and Future Directions'),
        (r'8\.\s+Conclusion', '8. Conclusion')
    ]
    for pattern, sec_name in section_patterns:
        if re.search(pattern, page_text):
            return sec_name
    return current_section


def load_pdf(path: Path, document_id: str, source: str, max_pages: int = 35) -> list[DocumentPage]:
    """Extract substantive pages from PDF with metadata."""
    reader = pypdf.PdfReader(str(path))
    pages: list[DocumentPage] = []
    
    current_sec = "1. Introduction"
    for idx in range(min(max_pages, len(reader.pages))):
        page_num = idx + 1
        raw_text = reader.pages[idx].extract_text() or ""
        norm_text = normalize_text(raw_text)
        
        current_sec = extract_section_for_page(page_num, raw_text, current_sec)
        
        doc_page = DocumentPage(
            document_id=document_id,
            source=source,
            page=page_num,
            text=norm_text,
            metadata={
                "section": current_sec,
                "char_count": len(norm_text)
            }
        )
        pages.append(doc_page)
        
    return pages
