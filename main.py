"""
Entry point — REPL з HITL для мультиагентної системи.
"""
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from supervisor import create_supervisor_agent, reset_session
from retriever import get_retriever


def format_tool_call(tool_call: dict) -> str:
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})
    formatted_args = []
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 60:
            value = value[:60] + "..."
        formatted_args.append(f'{key}="{value}"')
    return f"{name}({', '.join(formatted_args)})"


def handle_hitl_interrupt(pending_tool_call: dict, thread_id: str, agent):
    """
    Обробляє HITL interrupt для save_report.
    """
    args = pending_tool_call.get("args", {})
    filename = args.get("filename", "report.md")
    content = args.get("content", "")

    print("\n" + "=" * 60)
    print("⏸️  ACTION REQUIRES APPROVAL")
    print("=" * 60)
    print(f"  Tool: save_report")
    print(f"  File: {filename}")
    print("\n📄 Превью звіту (перші 500 символів):")
    print("-" * 40)
    print(content[:500] + ("..." if len(content) > 500 else ""))
    print("-" * 40)

    while True:
        decision = input("\n👉 approve / edit / reject: ").strip().lower()

        if decision == "approve":
            result = agent.invoke(
                Command(resume={"action": "approve"}),
                config={"configurable": {"thread_id": thread_id}},
            )
            print("\n✅ Approved!")
            return result

        elif decision == "edit":
            feedback = input("✏️  Your feedback: ").strip()
            result = agent.invoke(
                Command(resume={"action": "edit", "feedback": feedback}),
                config={"configurable": {"thread_id": thread_id}},
            )
            return result

        elif decision == "reject":
            reason = input("❌ Reason (optional): ").strip()
            result = agent.invoke(
                Command(resume={"action": "reject", "reason": reason}),
                config={"configurable": {"thread_id": thread_id}},
            )
            print("\n❌ Rejected.")
            return result

        else:
            print("⚠️  Введіть: approve, edit, або reject")


def run_supervisor_with_hitl(agent, user_input: str, thread_id: str) -> str:
    """
    Запускає Supervisor зі стрімінгом та HITL.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
    }

    print("\n" + "─" * 50)
    final_response = None
    pending_save_report = None

    for event in agent.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="values",
    ):
        messages = event.get("messages", [])
        if not messages:
            continue

        last_msg = messages[-1]

        if isinstance(last_msg, AIMessage):
            if last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_name = tc.get("name", "")
                    print(f"🔧 {format_tool_call(tc)}")
                    if tool_name == "save_report":
                        pending_save_report = tc
            else:
                if last_msg.content:
                    final_response = last_msg.content

        elif isinstance(last_msg, ToolMessage):
            content = last_msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"📎 {content}\n")

    # Обробляємо HITL після завершення стріму
    # (LangGraph з MemorySaver зупиниться на interrupt)
    if pending_save_report:
        result = handle_hitl_interrupt(pending_save_report, thread_id, agent)
        if result:
            msgs = result.get("messages", [])
            if msgs:
                last = msgs[-1]
                if hasattr(last, "content") and last.content:
                    final_response = last.content

    return final_response or "Агент завершив роботу."


def main():
    print("=" * 60)
    print("🔬 Мультиагентна дослідницька система (MCP + ACP)")
    print("=" * 60)
    print("\nАрхітектура:")
    print("  User → Supervisor → ACP → [Planner | Researcher | Critic]")
    print("                   → MCP → [SearchMCP | ReportMCP]")
    print()
    print("Перед запуском переконайтесь що працюють:")
    print("  python mcp_servers/search_mcp.py   # порт 8901")
    print("  python mcp_servers/report_mcp.py   # порт 8902")
    print("  python acp_server.py               # порт 8903")
    print()
    print("Команди: 'exit' — вийти | 'new' — нова сесія")
    print("=" * 60)

    try:
        agent = create_supervisor_agent()
        print("\n✅ Supervisor готовий!")

        retriever = get_retriever()
        if retriever.load_indices():
            print(f"✅ База знань: {len(retriever.chunks)} чанків")
        else:
            print("⚠️  База знань порожня. Запустіть: python ingest.py")
        print()
    except Exception as e:
        print(f"❌ Помилка ініціалізації: {e}")
        import traceback
        traceback.print_exc()
        return

    session_counter = 1
    session_id = f"session_{session_counter}"

    while True:
        try:
            user_input = input("📝 Ти: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 До побачення!")
                break

            if user_input.lower() == "new":
                session_counter += 1
                session_id = f"session_{session_counter}"
                reset_session()
                agent = create_supervisor_agent()
                print("🔄 Нова сесія.\n")
                continue

            if user_input.lower() == "status":
                retriever = get_retriever()
                if retriever._loaded:
                    print(f"📊 База знань: {len(retriever.chunks)} чанків")
                else:
                    print("📊 База знань не завантажена")
                continue

            print("\n🤖 Supervisor координує команду...")
            response = run_supervisor_with_hitl(agent, user_input, session_id)

            print("\n" + "─" * 50)
            print("🤖 Фінальна відповідь:")
            print("─" * 50)
            print(response)
            print("─" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 До побачення!")
            break
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
