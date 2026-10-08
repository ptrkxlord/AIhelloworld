import json
import re
from pathlib import Path


REQUIRED_METADATA = {"title", "category", "product", "audience", "revision_date", "status"}


def read_document(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise ValueError(f"У файлі {path.name} немає блоку метаданих")
    _, metadata_text, body = content.split("---", 2)
    metadata = json.loads(metadata_text)
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"У файлі {path.name} бракує метаданих: {', '.join(sorted(missing))}")
    return metadata, body.strip()


def split_document(path: Path, chunk_size: int, overlap: int) -> list[dict]:
    metadata, body = read_document(path)
    sections = re.split(r"(?m)^##\s+", body)
    chunks = []
    for section in sections:
        if not section.strip():
            continue
        lines = section.splitlines()
        heading = lines[0].strip() if len(lines) > 1 else metadata["title"]
        text = "\n".join(lines[1:]).strip() if len(lines) > 1 else section.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            excerpt = text[start:end].strip()
            if excerpt:
                chunks.append({
                    "id": f"{path.stem}-{len(chunks) + 1}",
                    "text": excerpt,
                    "embedding_text": f"{metadata['title']}\n{heading}\n{excerpt}",
                    "source": path.name,
                    "section": heading,
                    "metadata": metadata,
                })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def load_chunks(docs_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise ValueError(f"У {docs_dir} немає документів .md")
    return [chunk for path in files for chunk in split_document(path, chunk_size, overlap)]
