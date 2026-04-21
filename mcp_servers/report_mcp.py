"""
ReportMCP Server — інструменти збереження звітів як MCP сервер.
Порт: 8902
Tools: save_report
Resources: resource://output-dir
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from pathlib import Path
from config import OUTPUT_DIR, REPORT_MCP_PORT

mcp = FastMCP(
    name="ReportMCP",
    instructions="MCP сервер для збереження дослідницьких звітів.",
)


# ============================================================
# TOOLS
# ============================================================

@mcp.tool()
def save_report(filename: str, content: str) -> str:
    """
    Зберігає Markdown-звіт у файл.

    Args:
        filename: Назва файлу (наприклад, 'report.md')
        content: Текст звіту у Markdown

    Returns:
        Підтвердження збереження
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Звіт збережено: {filepath}"
    except Exception as e:
        return f"Помилка збереження: {e}"


# ============================================================
# RESOURCES
# ============================================================

@mcp.resource("resource://output-dir")
def output_dir_info() -> str:
    """
    Інформація про директорію виводу.
    Повертає шлях та список збережених звітів.
    """
    try:
        output_path = Path(OUTPUT_DIR)
        if not output_path.exists():
            return f"Директорія {OUTPUT_DIR}/ ще не існує (жодного звіту не збережено)."

        reports = sorted(output_path.glob("*.md"))
        report_list = "\n".join(f"  - {r.name}" for r in reports) or "  (порожньо)"

        return (
            f"Директорія: {output_path.resolve()}\n"
            f"Кількість звітів: {len(reports)}\n"
            f"Звіти:\n{report_list}"
        )
    except Exception as e:
        return f"Помилка: {e}"


if __name__ == "__main__":
    print(f"📄 ReportMCP запускається на порту {REPORT_MCP_PORT}...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=REPORT_MCP_PORT)
