from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict


def _read_text(rel_path: str) -> str:
    base = resources.files(__package__)
    p = base.joinpath(rel_path)
    return p.read_text(encoding="utf-8", errors="replace")


def _read_json(rel_path: str) -> Dict[str, Any]:
    return json.loads(_read_text(rel_path))


def load_tool_schema() -> Dict[str, Any]:
    """返回完整 tool schema 文件（含 name/description/parameters）。"""
    return _read_json("schemas/tool_schema_universal.json")


def load_tool_parameters_schema() -> Dict[str, Any]:
    """
    返回 MCP Tool 所需的 inputSchema（通常为 tool_schema['parameters']）。
    """
    obj = load_tool_schema()
    params = obj.get("parameters")
    if isinstance(params, dict):
        return params
    # 兼容未来可能的结构变动
    return obj


def load_workflow_template() -> Dict[str, Any]:
    """返回 ComfyUI API JSON workflow 模板。"""
    return _read_json("workflow/workflow_template.json")


def load_knowledge() -> Dict[str, str]:
    """返回内置知识库（若缺失则返回空字符串）。"""
    out: Dict[str, str] = {}
    for name in ("anima_expert", "artist_list", "prompt_examples"):
        try:
            out[name] = _read_text(f"knowledge/{name}.md")
        except Exception:
            out[name] = ""
    return out

