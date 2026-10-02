"""Reading PDFs (and .txt/.md notes) from a folder so the Librarian can mine them for ideas."""

from pathlib import Path

from pypdf import PdfReader

TEXT_SUFFIXES = {".txt", ".md"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def find_documents(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and (p.suffix.lower() == ".pdf" or p.suffix.lower() in TEXT_SUFFIXES))


def chunks(text: str, size: int = 12_000, overlap: int = 500) -> list[str]:
    """Split long text into overlapping pieces small enough for a local model's context."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out
