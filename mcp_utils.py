"""
Утиліти для роботи з MCP протоколом.
Конвертує MCP tools у LangChain-сумісний формат.
"""
from typing import Any
from langchain_core.tools import StructuredTool
from pydantic import create_model
import fastmcp


async def mcp_tools_to_langchain(client: fastmcp.Client) -> list[StructuredTool]:
    """
    Підключається до MCP сервера та конвертує його tools у LangChain формат.

    Args:
        client: Ініціалізований fastmcp.Client

    Returns:
        Список LangChain StructuredTool
    """
    tools = []

    async with client:
        mcp_tools = await client.list_tools()

        for mcp_tool in mcp_tools:
            # Будуємо Pydantic модель для аргументів
            fields = {}
            input_schema = mcp_tool.inputSchema or {}
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            for param_name, param_info in properties.items():
                param_type = str  # За замовчуванням str
                if param_info.get("type") == "integer":
                    param_type = int
                elif param_info.get("type") == "boolean":
                    param_type = bool
                elif param_info.get("type") == "number":
                    param_type = float

                description = param_info.get("description", "")
                if param_name in required:
                    fields[param_name] = (param_type, ...)
                else:
                    fields[param_name] = (param_type, None)

            ArgsModel = create_model(f"{mcp_tool.name}_args", **fields)

            # Замикання для захоплення імені tool
            tool_name = mcp_tool.name
            tool_description = mcp_tool.description or ""

            def make_tool_func(name: str):
                async def tool_func(**kwargs: Any) -> str:
                    async with client:
                        result = await client.call_tool(name, kwargs)
                        # Витягуємо текст з результату
                        if hasattr(result, "content"):
                            parts = result.content
                            if parts and hasattr(parts[0], "text"):
                                return parts[0].text
                        return str(result)
                return tool_func

            lc_tool = StructuredTool(
                name=tool_name,
                description=tool_description,
                args_schema=ArgsModel,
                coroutine=make_tool_func(tool_name),
            )
            tools.append(lc_tool)

    return tools
