from app import config
from app.chunking import load_chunks
from app.embeddings import encode_passages
from app.index_store import save_index


def main() -> None:
    chunks = load_chunks(config.DOCS_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    vectors = encode_passages([chunk["embedding_text"] for chunk in chunks])
    save_index(config.INDEX_DIR, chunks, vectors, config.EMBEDDING_MODEL)
    print(f"Створено індекс: {len(chunks)} фрагментів, модель {config.EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
