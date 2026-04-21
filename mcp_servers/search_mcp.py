"""
SearchMCP Server — інструменти пошуку як MCP сервер.
Порт: 8901
Tools: web_search, read_url, knowledge_search
Resources: resource://knowledge-base-stats
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from ddgs import DDGS
import trafilatura
from datetime import datetime

from config import (
    MAX_SEARCH_RESULTS,
    MAX_PAGE_CONTENT_LENGTH,
    SEARCH_MCP_PORT,
)
from retriever import search_knowledge_base, get_retriever

# Ініціалізація FastMCP сервера
mcp = FastMCP(
    name="SearchMCP",
    instructions="MCP сервер з інструментами пошуку для дослідницьких агентів.",
)


# ============================================================
# TOOLS
# ============================================================

@mcp.tool()
def web_search(query: str) -> str:
    """
    Пошук в інтернеті через DuckDuckGo.

    Args:
        query: Пошуковий запит

    Returns:
        Список результатів пошуку
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
            if not results:
                return "Нічого не знайдено."
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[{i}] {r.get('title', 'Без заголовка')}\n"
                    f"    URL: {r.get('href', 'N/A')}\n"
                    f"    {r.get('body', 'Без опису')}"
                )
            return "\n\n".join(formatted)
    except Exception as e:
        return f"Помилка пошуку: {e}"


@mcp.tool()
def read_url(url: str) -> str:
    """
    Отримує повний текст веб-сторінки за URL.

    Args:
        url: Адреса веб-сторінки

    Returns:
        Текстовий вміст сторінки
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Не вдалося завантажити: {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return "Не вдалося витягти текст."
        if len(text) > MAX_PAGE_CONTENT_LENGTH:
            text = text[:MAX_PAGE_CONTENT_LENGTH] + "\n\n[... обрізано ...]"
        return text
    except Exception as e:
        return f"Помилка: {e}"


@mcp.tool()
def knowledge_search(query: str) -> str:
    """
    Пошук у локальній базі знань (RAG).

    Args:
        query: Пошуковий запит

    Returns:
        Знайдені фрагменти документів з джерелами
    """
    try:
        result = search_knowledge_base(query)
        return result
    except Exception as e:
        return f"Помилка пошуку в базі знань: {e}"


# ============================================================
# RESOURCES
# ============================================================

@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """
    Статистика локальної бази знань.
    Повертає кількість документів та дату останнього оновлення.
    """
    try:
        retriever = get_retriever()
        if not retriever._loaded:
            retriever.load_indices()

        if retriever._loaded:
            doc_count = len(retriever.chunks)
            # Дата останнього оновлення індексу
            import os
            from pathlib import Path
            from config import VECTOR_DB_DIR
            faiss_path = Path(VECTOR_DB_DIR) / "faiss_index"
            if faiss_path.exists():
                mtime = os.path.getmtime(faiss_path)
                last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_updated = "невідомо"

            return (
                f"Кількість документів (чанків): {doc_count}\n"
                f"Дата останнього оновлення: {last_updated}"
            )
        else:
            return "База знань не завантажена. Запустіть: python ingest.py"
    except Exception as e:
        return f"Помилка отримання статистики: {e}"


if __name__ == "__main__":
    print(f"🔍 SearchMCP запускається на порту {SEARCH_MCP_PORT}...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=SEARCH_MCP_PORT)
