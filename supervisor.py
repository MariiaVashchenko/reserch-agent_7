"""
Supervisor Agent — оркеструє команду через HTTP виклики до ACP сервера.
"""
import asyncio
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import fastmcp
from config import (
    OPENAI_API_KEY,
    MODEL_NAME,
    SUPERVISOR_PROMPT,
    ACP_SERVER_URL,
    REPORT_MCP_URL,
)

_current_request = ""
_revision_count = 0


# ============================================================
# ACP helper
# ============================================================

def _call_acp_agent(agent_name: str, message: str) -> str:
    """Викликає ACP агента через HTTP POST."""
    url = f"{ACP_SERVER_URL}/agents/{agent_name}/runs"
    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, json={"message": message})
        response.raise_for_status()
        return response.json()["result"]


# ============================================================
# SUPERVISOR TOOLS
# ============================================================

@tool
def delegate_to_planner(request: str) -> str:
    """
    Делегує задачу планування до Planner Agent через ACP.

    Args:
        request: Запит користувача для дослідження

    Returns:
        Структурований план дослідження
    """
    global _current_request
    _current_request = request
    print(f"\n[Supervisor → ACP → Planner]")
    result = _call_acp_agent("planner", request)
    print(f"  📎 Planner відповів ({len(result)} chars)")
    return result


@tool
def delegate_to_researcher(request: str) -> str:
    """
    Делегує задачу дослідження до Research Agent через ACP.

    Args:
        request: План або запит для дослідження

    Returns:
        Знахідки дослідження
    """
    global _revision_count
    _revision_count += 1
    print(f"\n[Supervisor → ACP → Researcher]  (round {_revision_count})")
    result = _call_acp_agent("researcher", request)
    print(f"  📎 Researcher відповів ({len(result)} chars)")
    return result


@tool
def delegate_to_critic(findings: str) -> str:
    """
    Делегує задачу критики до Critic Agent через ACP.

    Args:
        findings: Знахідки від Research Agent

    Returns:
        Результат критики з вердиктом APPROVE або REVISE
    """
    global _current_request
    print(f"\n[Supervisor → ACP → Critic]")
    message = f"ORIGINAL_REQUEST: {_current_request}\nFINDINGS: {findings}"
    result = _call_acp_agent("critic", message)
    print(f"  📎 Critic відповів ({len(result)} chars)")
    return result


@tool
def save_report(filename: str, content: str) -> str:
    """
    Зберігає Markdown-звіт через ReportMCP.
    ПОТРЕБУЄ ПІДТВЕРДЖЕННЯ КОРИСТУВАЧА (HITL).

    Args:
        filename: Назва файлу (наприклад, 'report.md')
        content: Текст звіту у Markdown

    Returns:
        Підтвердження збереження
    """
    print(f"\n[Supervisor → MCP → save_report]")

    async def _call():
        async with fastmcp.Client(REPORT_MCP_URL) as client:
            result = await client.call_tool(
                "save_report",
                {"filename": filename, "content": content},
            )
            if hasattr(result, "content") and result.content:
                return result.content[0].text
            return str(result)

    return asyncio.run(_call())


# ============================================================
# SUPERVISOR AGENT
# ============================================================

def create_supervisor_agent():
    global _revision_count
    _revision_count = 0

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )

    memory = MemorySaver()

    supervisor_tools = [
        delegate_to_planner,
        delegate_to_researcher,
        delegate_to_critic,
        save_report,
    ]

    agent = create_react_agent(
        model=llm,
        tools=supervisor_tools,
        checkpointer=memory,
        prompt=SUPERVISOR_PROMPT,
    )
    return agent


def reset_session():
    global _revision_count, _current_request
    _revision_count = 0
    _current_request = ""
