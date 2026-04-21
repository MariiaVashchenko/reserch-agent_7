"""
ACP Server — три агенти через асинхронний FastAPI сервер.
"""
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import fastmcp
from config import SEARCH_MCP_URL, ACP_SERVER_PORT
from mcp_utils import mcp_tools_to_langchain
from agents.planner import create_planner_agent, run_planner
from agents.research import create_research_agent, run_researcher
from agents.critic import create_critic_agent, run_critic
from schemas import ResearchPlan, CritiqueResult

app = FastAPI(title="ACP Server", version="1.0.0")


class RunRequest(BaseModel):
    message: str


class RunResponse(BaseModel):
    agent: str
    result: str


async def get_search_tools() -> list:
    """Підключається до SearchMCP та повертає асинхронні LangChain tools."""
    client = fastmcp.Client(SEARCH_MCP_URL)
    tools = await mcp_tools_to_langchain(client)
    return tools


@app.post("/agents/planner/runs", response_model=RunResponse)
async def run_planner_endpoint(request: RunRequest):
    try:
        print(f"\n[Planner] Отримано запит...")
        tools = await get_search_tools()
        agent = create_planner_agent(tools)


        research_plan: ResearchPlan = await run_planner(agent, request.message)

        result = (
            f"Мета: {research_plan.goal}\n"
            f"Пошукові запити: {', '.join(research_plan.search_queries)}\n"
            f"Джерела: {', '.join(research_plan.sources_to_check)}\n"
            f"Формат звіту: {research_plan.output_format}"
        )
        return RunResponse(agent="planner", result=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/researcher/runs", response_model=RunResponse)
async def run_researcher_endpoint(request: RunRequest):
    try:
        print(f"\n[Researcher] Отримано запит...")
        tools = await get_search_tools()
        agent = create_research_agent(tools)

        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Добавлен await
        findings = await run_researcher(agent, request.message)

        return RunResponse(agent="researcher", result=findings)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/critic/runs", response_model=RunResponse)
async def run_critic_endpoint(request: RunRequest):
    try:
        print(f"\n[Critic] Отримано запит...")
        raw = request.message
        original_request = ""
        findings = raw

        if "ORIGINAL_REQUEST:" in raw and "FINDINGS:" in raw:
            parts = raw.split("FINDINGS:", 1)
            original_request = parts[0].replace("ORIGINAL_REQUEST:", "").strip()
            findings = parts[1].strip()

        tools = await get_search_tools()
        agent = create_critic_agent(tools)


        critique_result: CritiqueResult = await run_critic(agent, findings, original_request)

        result = (
            f"Вердикт: {critique_result.verdict}\n"
            f"Актуальність: {critique_result.is_fresh}\n"
            f"Запити на доопрацювання: {critique_result.revision_requests}"
        )
        return RunResponse(agent="critic", result=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ACP_SERVER_PORT)