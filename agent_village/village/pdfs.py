"""Reading the library: PDFs, notes, Pine Script indicators, subtitles and video transcripts."""

import re
from pathlib import Path

from pypdf import PdfReader

TEXT_SUFFIXES = {".txt", ".md", ".pine", ".pinescript", ".srt", ".vtt"}
PINE_SUFFIXES = {".pine", ".pinescript"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4a", ".mp3", ".wav", ".flac"}


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".srt", ".vtt"):
        return subtitles_to_text(text)
    return text


def is_pine(path: Path, text: str = "") -> bool:
    return path.suffix.lower() in PINE_SUFFIXES or "//@version=" in text[:2000]


def subtitles_to_text(text: str) -> str:
    """Drop cue numbers and timestamps, keep the spoken words (one line per minute marker)."""
    out, last_min = [], None
    for line in text.splitlines():
        line = line.strip()
        stamp = re.match(r"(\d{1,2}):(\d{2}):(\d{2})[.,]\d{3}\s+-->", line)
        if stamp:
            minute = int(stamp.group(1)) * 60 + int(stamp.group(2))
            if minute != last_min:
                out.append(f"\n[{minute // 60:02d}:{minute % 60:02d}]")
                last_min = minute
            continue
        if not line or line.isdigit() or line.upper().startswith(("WEBVTT", "NOTE", "KIND:", "LANGUAGE:")):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if not out or out[-1] != line:
            out.append(line)
    return " ".join(out)


def transcribe(folder: str | Path, out_dir: str | Path, model_size: str = "small", log=print) -> int:
    """Turn videos/audio into text with a local Whisper model (pip install faster-whisper).
    Runs offline after the model's one-time download. Returns the number of new transcripts."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("transcribing needs: pip install faster-whisper") from e
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    todo = [v for v in videos if not (out_dir / f"{v.stem}.transcript.txt").exists()]
    if not todo:
        log(f"Nothing new to transcribe in {folder} ({len(videos)} files already done).")
        return 0
    log(f"Loading Whisper '{model_size}' (first time downloads it) ...")
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    for video in todo:
        log(f"Transcribing {video.name} ...")
        segments, info = model.transcribe(str(video), vad_filter=True)
        lines = [f"Transcript of {video.name} (language {info.language})"]
        for seg in segments:
            m, sec = divmod(int(seg.start), 60)
            lines.append(f"[{m // 60:02d}:{m % 60:02d}:{sec:02d}] {seg.text.strip()}")
        (out_dir / f"{video.stem}.transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(todo)


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
