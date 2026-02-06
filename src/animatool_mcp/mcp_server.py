"""
animatool-mcp MCP Server (stdio).

启动方式：
    animatool-mcp

或（开发/未安装时）：
    python -m animatool_mcp.mcp_server
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from .client import AnimaExecutor
from .config import AnimaToolConfig
from .resources import load_tool_parameters_schema, load_tool_schema


server = Server("animatool-mcp")

_executor: AnimaExecutor | None = None


def get_executor() -> AnimaExecutor:
    global _executor
    if _executor is None:
        _executor = AnimaExecutor(config=AnimaToolConfig())
    return _executor


_TOOL_META = load_tool_schema()
_TOOL_NAME = str(_TOOL_META.get("name") or "generate_anima_image")
_TOOL_DESC = str(_TOOL_META.get("description") or "Generate image via ComfyUI + Anima workflow.")
_TOOL_INPUT_SCHEMA = load_tool_parameters_schema()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=_TOOL_NAME,
            description=_TOOL_DESC,
            inputSchema=_TOOL_INPUT_SCHEMA,
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
    if name != _TOOL_NAME:
        return [TextContent(type="text", text=f"未知工具: {name}")]

    try:
        ex = get_executor()
        result = await asyncio.to_thread(ex.generate, arguments or {})
        if not result.get("success"):
            return [TextContent(type="text", text=f"生成失败: {result}")]

        contents: list[TextContent | ImageContent] = []

        info_text = (
            "生成成功！\n"
            f"- prompt_id: {result.get('prompt_id')}\n"
            f"- 分辨率: {result.get('width')} x {result.get('height')}\n"
            f"- 图片数量: {len(result.get('images') or [])}\n"
        )
        contents.append(TextContent(type="text", text=info_text))

        for img in result.get("images") or []:
            b64 = img.get("base64")
            mime = img.get("mime_type")
            if b64 and mime:
                contents.append(ImageContent(type="image", data=b64, mimeType=mime))

        return contents
    except Exception as e:
        return [TextContent(type="text", text=f"错误: {str(e)}")]


async def _amain() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

