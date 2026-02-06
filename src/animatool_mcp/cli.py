from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .client import AnimaExecutor
from .config import AnimaToolConfig


def _load_json_arg(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
    except Exception as e:
        raise SystemExit(f"--json 解析失败：{e}") from e
    if not isinstance(obj, dict):
        raise SystemExit("--json 必须是一个 JSON object")
    return obj


def _load_json_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到文件：{p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("--json-file 内容必须是一个 JSON object")
    return obj


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="animatool-mcp CLI (ComfyUI Server client)")

    parser.add_argument("--comfyui-url", default=None, help="ComfyUI base URL（覆盖 COMFYUI_URL）")
    parser.add_argument("--bearer-token", default=None, help="Bearer token（覆盖 ANIMATOOL_BEARER_TOKEN）")
    parser.add_argument("--headers-json", default=None, help="自定义 headers JSON（覆盖 ANIMATOOL_HEADERS_JSON）")
    parser.add_argument("--basic-user", default=None, help="Basic auth user（覆盖 ANIMATOOL_BASIC_USER）")
    parser.add_argument("--basic-pass", default=None, help="Basic auth pass（覆盖 ANIMATOOL_BASIC_PASS）")
    parser.add_argument(
        "--ssl-verify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SSL 校验（覆盖 ANIMATOOL_SSL_VERIFY）",
    )
    parser.add_argument(
        "--download-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否把图片保存到本地（覆盖 ANIMATOOL_DOWNLOAD_IMAGES）",
    )
    parser.add_argument("--output-dir", default=None, help="保存目录（覆盖 ANIMATOOL_OUTPUT_DIR）")

    parser.add_argument("--json", default=None, help="直接传入 JSON object 字符串")
    parser.add_argument("--json-file", default=None, help="从文件读取 JSON object")

    args = parser.parse_args(argv)

    if not args.json and not args.json_file:
        raise SystemExit("必须提供 --json 或 --json-file")
    if args.json and args.json_file:
        raise SystemExit("只能二选一：--json 或 --json-file")

    payload = _load_json_arg(args.json) if args.json else _load_json_file(args.json_file)

    cfg_kwargs: Dict[str, Any] = {}
    if args.comfyui_url:
        cfg_kwargs["comfyui_url"] = str(args.comfyui_url)
    if args.bearer_token:
        cfg_kwargs["bearer_token"] = str(args.bearer_token)
    if args.headers_json:
        cfg_kwargs["headers_json"] = str(args.headers_json)
    if args.basic_user is not None:
        cfg_kwargs["basic_user"] = str(args.basic_user)
    if args.basic_pass is not None:
        cfg_kwargs["basic_pass"] = str(args.basic_pass)
    if args.ssl_verify is not None:
        cfg_kwargs["ssl_verify"] = bool(args.ssl_verify)
    if args.download_images is not None:
        cfg_kwargs["download_images"] = bool(args.download_images)
    if args.output_dir:
        cfg_kwargs["output_dir"] = Path(args.output_dir)

    cfg = AnimaToolConfig(**cfg_kwargs)
    ex = AnimaExecutor(config=cfg)
    result = ex.generate(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

