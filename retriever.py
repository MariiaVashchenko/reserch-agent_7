"""
Hybrid Retrieval з Reranking для RAG-системи.

Компоненти:
1. Semantic Search — FAISS з OpenAI embeddings
2. BM25 Search — лексичний пошук
3. Ensemble — об'єднання результатів
4. Reranking — cross-encoder для фільтрації шуму
"""

import os
# Вимикаємо конфлікти дублювання бібліотек Intel MKL
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Обмежуємо кількість потоків для стабільності на Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # <-- ВИПРАВЛЕНО

from rank_bm25 import BM25Okapi
import numpy as np

from config import (
    OPENAI_API_KEY,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
    RERANKER_MODEL,
    USE_RERANKER,
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
)


@dataclass
class RetrievalResult:
    """Результат пошуку з метаданими."""
    content: str
    metadata: dict
    score: float
    source: str  # "semantic", "bm25", або "hybrid"


class HybridRetriever:
    """
    Гібридний retriever, що об'єднує semantic та lexical пошук.
    """

    def __init__(self):
        self.faiss_index: Optional[FAISS] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[Tuple[str, dict]] = []
        self.tokenized_docs: List[List[str]] = []
        self.reranker = None
        self._loaded = False

    def load_indices(self) -> bool:
        """
        Завантажує індекси з диску.
        Повертає True якщо успішно.
        """
        vector_db_path = Path(VECTOR_DB_DIR)

        if not vector_db_path.exists():
            print(f"⚠️  Індекси не знайдено у {VECTOR_DB_DIR}/")
            print("   Запустіть спочатку: python ingest.py")
            return False

        try:

            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            self.faiss_index = FAISS.load_local(
                str(vector_db_path / "faiss_index"),
                embeddings,
                allow_dangerous_deserialization=True
            )


            with open(vector_db_path / "bm25_index.pkl", "rb") as f:
                bm25_data = pickle.load(f)

            self.bm25_index = bm25_data["bm25_index"]
            self.tokenized_docs = bm25_data["tokenized_docs"]
            self.chunks = bm25_data["chunks"]


            if USE_RERANKER:
                self._load_reranker()

            self._loaded = True
            return True

        except Exception as e:
            print(f"❌ Помилка завантаження індексів: {e}")
            return False

    def _load_reranker(self):
        """
        Завантажує cross-encoder reranker.
        """
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_MODEL)
            print(f"  ✅ Reranker завантажено: {RERANKER_MODEL}")
        except ImportError:
            print("  ⚠️  sentence-transformers не встановлено. Reranking вимкнено.")
            self.reranker = None
        except Exception as e:
            print(f"  ⚠️  Не вдалося завантажити reranker: {e}")
            self.reranker = None

    def semantic_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[RetrievalResult]:
        """
        Semantic search через FAISS.
        """
        if not self.faiss_index:
            return []

        results = self.faiss_index.similarity_search_with_score(query, k=k)

        retrieval_results = []
        for doc, score in results:
            similarity = 1 / (1 + score)

            retrieval_results.append(RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=similarity,
                source="semantic"
            ))

        return retrieval_results

    def bm25_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[RetrievalResult]:
        """
        BM25 лексичний пошук.
        """
        if not self.bm25_index or not self.chunks:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        retrieval_results = []
        for idx in top_indices:
            if scores[idx] > 0:
                content, metadata = self.chunks[idx]
                retrieval_results.append(RetrievalResult(
                    content=content,
                    metadata=metadata,
                    score=float(scores[idx]),
                    source="bm25"
                ))

        return retrieval_results

    def ensemble_results(
            self,
            semantic_results: List[RetrievalResult],
            bm25_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Об'єднує результати semantic та BM25 пошуку через RRF.
        """
        combined = {}
        k = 60

        for rank, result in enumerate(semantic_results):
            content_key = result.content[:100]
            if content_key not in combined:
                combined[content_key] = {
                    "result": result,
                    "rrf_score": 0
                }
            combined[content_key]["rrf_score"] += SEMANTIC_WEIGHT * (1 / (k + rank + 1))

        for rank, result in enumerate(bm25_results):
            content_key = result.content[:100]
            if content_key not in combined:
                combined[content_key] = {
                    "result": result,
                    "rrf_score": 0
                }
            combined[content_key]["rrf_score"] += BM25_WEIGHT * (1 / (k + rank + 1))

        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        final_results = []
        for item in sorted_results:
            result = item["result"]
            result.score = item["rrf_score"]
            result.source = "hybrid"
            final_results.append(result)

        return final_results

    def rerank(
            self,
            query: str,
            results: List[RetrievalResult],
            top_k: int = TOP_K_FINAL
    ) -> List[RetrievalResult]:
        """
        Reranking через cross-encoder.
        """
        if not self.reranker or not results:
            return results[:top_k]

        pairs = [(query, r.content) for r in results]
        scores = self.reranker.predict(pairs)

        for i, result in enumerate(results):
            result.score = float(scores[i])

        reranked = sorted(results, key=lambda x: x.score, reverse=True)

        return reranked[:top_k]

    def search(self, query: str, top_k: int = TOP_K_FINAL) -> List[RetrievalResult]:
        """
        Головний метод пошуку: hybrid search + reranking.
        """
        if not self._loaded:
            if not self.load_indices():
                return []

        semantic_results = self.semantic_search(query, k=TOP_K_RETRIEVAL)
        bm25_results = self.bm25_search(query, k=TOP_K_RETRIEVAL)
        combined_results = self.ensemble_results(semantic_results, bm25_results)

        if USE_RERANKER and self.reranker:
            final_results = self.rerank(query, combined_results, top_k)
        else:
            final_results = combined_results[:top_k]

        return final_results


_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Повертає singleton екземпляр retriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def search_knowledge_base(query: str) -> str:
    """Функція для використання в tools."""
    retriever = get_retriever()
    results = retriever.search(query)

    if not results:
        return "Нічого не знайдено в базі знань. Спробуйте інший запит або пошук в інтернеті."

    formatted = []
    for i, result in enumerate(results, 1):
        source_info = result.metadata.get("filename", "Unknown")
        page_info = result.metadata.get("page", "")
        if page_info:
            source_info += f", сторінка {page_info + 1}"

        formatted.append(
            f"[Результат {i}] (Джерело: {source_info})\n"
            f"{result.content}\n"
        )

    return "\n---\n".join(formatted)
