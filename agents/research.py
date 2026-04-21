"""
Research Agent — виконує дослідження згідно з планом.
Запускається у складі ACP сервера.
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config import OPENAI_API_KEY, MODEL_NAME, RESEARCHER_PROMPT


def create_research_agent(tools: list):
    """
    Створює Research Agent з MCP tools.

    Args:
        tools: LangChain tools отримані з SearchMCP
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )
    # Створюємо React-агента, який вміє викликати інструменти
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RESEARCHER_PROMPT,
    )
    return agent


async def run_researcher(agent, request: str) -> str:
    """
    Запускає Research Agent асинхронно і повертає знахідки.

    Зверніть увагу: ми використовуємо ainvoke, оскільки MCP інструменти
    вимагають асинхронного середовища виконання.
    """
    # Використовуємо await та ainvoke замінюючи синхронний invoke
    result = await agent.ainvoke({"messages": [("user", request)]})

    # Повертаємо контент останнього повідомлення в ланцюжку (відповідь агента)
    return result["messages"][-1].content