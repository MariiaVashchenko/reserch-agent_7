"""
Ingestion Pipeline: завантаження документів у векторну базу даних.

Запуск: python ingest.py
"""

import os
import pickle
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi
import nltk

from config import (
    OPENAI_API_KEY,
    DATA_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

# токенізатор для BM25
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def load_documents(data_dir: str) -> List[Document]:
    """
    Завантажує документи з директорії.
    Підтримує PDF, TXT, MD файли.
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"⚠️  Директорія {data_dir} не існує. Створюю...")
        data_path.mkdir(parents=True, exist_ok=True)
        return documents


    pdf_files = list(data_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["filename"] = pdf_file.name
            documents.extend(docs)
            print(f"  ✅ Завантажено: {pdf_file.name} ({len(docs)} сторінок)")
        except Exception as e:
            print(f"  ❌ Помилка завантаження {pdf_file.name}: {e}")


    txt_files = list(data_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "txt"
                doc.metadata["filename"] = txt_file.name
            documents.extend(docs)
            print(f"  ✅ Завантажено: {txt_file.name}")
        except Exception as e:
            print(f"  ❌ Помилка завантаження {txt_file.name}: {e}")


    md_files = list(data_path.glob("*.md"))
    for md_file in md_files:
        try:
            loader = TextLoader(str(md_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "markdown"
                doc.metadata["filename"] = md_file.name
            documents.extend(docs)
            print(f"  ✅ Завантажено: {md_file.name}")
        except Exception as e:
            print(f"  ❌ Помилка завантаження {md_file.name}: {e}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Розбиває документи на чанки з перекриттям.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def create_bm25_index(chunks: List[Document]) -> tuple:
    """
    Створює BM25 індекс для лексичного пошуку.
    """
    tokenized_docs = []
    for chunk in chunks:
        tokens = chunk.page_content.lower().split()
        tokenized_docs.append(tokens)

    bm25_index = BM25Okapi(tokenized_docs)

    return bm25_index, tokenized_docs


def save_indices(
        faiss_index: FAISS,
        bm25_index: BM25Okapi,
        chunks: List[Document],
        tokenized_docs: List[List[str]],
        output_dir: str
):
    """
    Зберігає всі індекси на диск.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Зберігаємо FAISS індекс
    faiss_index.save_local(str(output_path / "faiss_index"))
    print(f"  ✅ FAISS індекс збережено")

    # Зберігаємо BM25 індекс та пов'язані дані
    bm25_data = {
        "bm25_index": bm25_index,
        "tokenized_docs": tokenized_docs,
        "chunks": [(chunk.page_content, chunk.metadata) for chunk in chunks]
    }

    with open(output_path / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"  ✅ BM25 індекс збережено")


def main():
    """
    Головна функція ingestion pipeline.
    """
    print("=" * 60)
    print("📚 Document Ingestion Pipeline")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY не знайдено у .env файлі!")
        return

    # 1. Завантаження документів
    print(f"\n📂 Завантаження документів з {DATA_DIR}/...")
    documents = load_documents(DATA_DIR)

    if not documents:
        print("⚠️  Документи не знайдено. Додайте PDF/TXT/MD файли у папку data/")
        return

    print(f"\n📄 Всього завантажено: {len(documents)} документів")

    # 2. Chunking
    print(f"\n✂️  Розбиття на чанки (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents)
    print(f"   Створено {len(chunks)} чанків")

    # 3. Створення embeddings та FAISS індексу
    print(f"\n🧮 Створення embeddings ({EMBEDDING_MODEL})...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    print("   Індексація у FAISS...")
    faiss_index = FAISS.from_documents(chunks, embeddings)

    # 4. Створення BM25 індексу
    print("\n📊 Створення BM25 індексу...")
    bm25_index, tokenized_docs = create_bm25_index(chunks)

    # 5. Збереження індексів
    print(f"\n💾 Збереження індексів у {VECTOR_DB_DIR}/...")
    save_indices(faiss_index, bm25_index, chunks, tokenized_docs, VECTOR_DB_DIR)

    print("\n" + "=" * 60)
    print("✅ Ingestion завершено успішно!")
    print(f"   - Документів: {len(documents)}")
    print(f"   - Чанків: {len(chunks)}")
    print(f"   - Індекси збережено у: {VECTOR_DB_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
