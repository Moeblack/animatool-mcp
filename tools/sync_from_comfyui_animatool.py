from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def _ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def _same_content(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False
    return _read_bytes(a) == _read_bytes(b)


def _copy_file(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    shutil.copy2(src, dst)


def _detect_source_root(workspace_root: Path) -> Path:
    """
    兼容两种布局：
    1) 单仓库布局（推荐）：
       repo_root/
         executor/
         schemas/
         knowledge/
         animatool-mcp/
    2) 本地 ComfyUI 工作区布局（你现在的目录）：
       workspace_root/ComfyUI-Main/custom_nodes/ComfyUI-AnimaTool/
    """
    repo_like = workspace_root
    if (repo_like / "schemas" / "tool_schema_universal.json").exists() and (repo_like / "executor" / "workflow_template.json").exists():
        return repo_like

    comfy_custom_node = workspace_root / "ComfyUI-Main" / "custom_nodes" / "ComfyUI-AnimaTool"
    if (comfy_custom_node / "schemas" / "tool_schema_universal.json").exists():
        return comfy_custom_node

    raise SystemExit(
        "Unable to auto-detect source_root. Please pass --source (a dir containing executor/schemas/knowledge).\n"
        f"workspace_root={workspace_root}"
    )


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    animatool_mcp_dir = script_dir.parent
    workspace_root = animatool_mcp_dir.parent

    parser = argparse.ArgumentParser(description="Sync shared animatool assets into animatool-mcp package resources.")
    parser.add_argument("--source", default=None, help="source_root（包含 executor/schemas/knowledge 的目录）")
    parser.add_argument("--check", action="store_true", help="只检查是否一致，不写入")
    parser.add_argument("--write", action="store_true", help="执行同步写入")
    parser.add_argument("--sync-anima-tool", action="store_true", help="同时同步到 workspace_root/anima-tool（若存在）")
    args = parser.parse_args(argv)

    if not args.check and not args.write:
        # 默认安全：CI 用 check；本地可显式 --write
        args.check = True

    source_root = Path(args.source).resolve() if args.source else _detect_source_root(workspace_root)

    # target: animatool-mcp 资源目录
    target_resources = animatool_mcp_dir / "src" / "animatool_mcp" / "resources"
    if not target_resources.exists():
        raise SystemExit(f"Target resources dir not found: {target_resources}")

    mappings: list[tuple[Path, Path]] = []

    # schema
    mappings.append(
        (
            source_root / "schemas" / "tool_schema_universal.json",
            target_resources / "schemas" / "tool_schema_universal.json",
        )
    )

    # workflow template
    mappings.append(
        (
            source_root / "executor" / "workflow_template.json",
            target_resources / "workflow" / "workflow_template.json",
        )
    )

    # knowledge: copy all *.md from source knowledge
    source_knowledge = source_root / "knowledge"
    target_knowledge = target_resources / "knowledge"
    if source_knowledge.exists():
        for src_md in sorted(source_knowledge.glob("*.md")):
            mappings.append((src_md, target_knowledge / src_md.name))

    # optional: sync to legacy anima-tool folder (workspace only)
    anima_tool_root = workspace_root / "anima-tool"
    if args.sync_anima_tool and anima_tool_root.exists():
        mappings.append(
            (
                source_root / "schemas" / "tool_schema_universal.json",
                anima_tool_root / "schemas" / "tool_schema_universal.json",
            )
        )
        mappings.append(
            (
                source_root / "executor" / "workflow_template.json",
                anima_tool_root / "executor" / "workflow_template.json",
            )
        )
        if source_knowledge.exists():
            for src_md in sorted(source_knowledge.glob("*.md")):
                mappings.append((src_md, anima_tool_root / "knowledge" / src_md.name))

    missing = [(s, d) for (s, d) in mappings if not s.exists()]
    if missing:
        msg = "\n".join([f"- missing src: {s} -> {d}" for s, d in missing])
        raise SystemExit(f"Missing required files under source_root:\n{msg}\nsource_root={source_root}")

    mismatched: list[tuple[Path, Path]] = []
    updated: list[tuple[Path, Path]] = []

    for src, dst in mappings:
        if _same_content(src, dst):
            continue
        if args.check:
            mismatched.append((src, dst))
            continue
        _copy_file(src, dst)
        updated.append((src, dst))

    if args.check:
        if mismatched:
            print("[FAIL] Resources out of sync (run --write):")
            for src, dst in mismatched:
                print(f"- {src} -> {dst}")
            return 1
        print("[OK] Resources are synced")
        return 0

    # write mode
    if updated:
        print("[OK] Synced files:")
        for src, dst in updated:
            print(f"- {src} -> {dst}")
    else:
        print("[OK] Already up-to-date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

