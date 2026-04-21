"""
Planner Agent — декомпозує запит у структурований план.
Запускається у складі ACP сервера.
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config import OPENAI_API_KEY, MODEL_NAME, PLANNER_PROMPT
from schemas import ResearchPlan


def create_planner_agent(tools: list):
    """
    Створює Planner Agent з MCP tools.

    Args:
        tools: LangChain tools отримані з SearchMCP
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=PLANNER_PROMPT,
    )
    return agent


async def run_planner(agent, request: str) -> ResearchPlan:
    """
    Запускає Planner асинхронно і парсить результат у ResearchPlan.
    """

    result = await agent.ainvoke({"messages": [("user", request)]})
    last_message = result["messages"][-1].content

    # Парсимо у структурований формат
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(ResearchPlan)

    # Викликаємо парсер також асинхронно для однорідності
    plan = await structured_llm.ainvoke(
        f"Витягни план дослідження з цього тексту:\n\n{last_message}"
    )

    return plan