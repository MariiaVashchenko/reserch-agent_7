"""
Critic Agent — оцінює якість дослідження.
Запускається у складі ACP сервера.
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config import OPENAI_API_KEY, MODEL_NAME, CRITIC_PROMPT
from schemas import CritiqueResult


def create_critic_agent(tools: list):
    """
    Створює Critic Agent з MCP tools.

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
        prompt=CRITIC_PROMPT,
    )
    return agent


async def run_critic(agent, findings: str, original_request: str) -> CritiqueResult:
    """
    Запускає Critic Agent асинхронно і повертає структурований результат.
    """
    critique_request = f"""
Оригінальний запит користувача: {original_request}

Знахідки дослідження:
{findings}

Оціни це дослідження за трьома вимірами: freshness, completeness, structure.
Зроби верифікаційні пошуки за потреби.
"""

    result = await agent.ainvoke({"messages": [("user", critique_request)]})
    last_message = result["messages"][-1].content


    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(CritiqueResult)

    # Викликаємо структурований вивід асинхронно
    critique = await structured_llm.ainvoke(
        f"Витягни результат критики з цього тексту:\n\n{last_message}"
    )
    return critique