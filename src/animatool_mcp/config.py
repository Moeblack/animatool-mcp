from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


def _get_env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _get_env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _parse_headers_json(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"ANIMATOOL_HEADERS_JSON 不是合法 JSON：{e}") from e
    if not isinstance(obj, dict):
        raise ValueError("ANIMATOOL_HEADERS_JSON 必须是 JSON object")
    headers: Dict[str, str] = {}
    for k, v in obj.items():
        if v is None:
            continue
        headers[str(k)] = str(v)
    return headers


# 默认模型文件名（与 ComfyUI-AnimaTool 对齐）
DEFAULT_UNET_NAME = "anima-preview.safetensors"
DEFAULT_CLIP_NAME = "qwen_3_06b_base.safetensors"
DEFAULT_VAE_NAME = "qwen_image_vae.safetensors"


@dataclass
class AnimaToolConfig:
    """
    独立 MCP/CLI 的配置（通过环境变量覆盖）。

    基础：
    - COMFYUI_URL: ComfyUI Server base URL（默认 http://127.0.0.1:8188）
    - ANIMATOOL_TIMEOUT: 请求超时秒数（默认 600）
    - ANIMATOOL_POLL_INTERVAL: 轮询 history 间隔秒（默认 1）
    - ANIMATOOL_SSL_VERIFY: SSL 校验（默认 true；自签名可设 false）

    鉴权/Headers（可选）：
    - ANIMATOOL_BEARER_TOKEN: 自动注入 Authorization: Bearer ...
    - ANIMATOOL_BASIC_USER / ANIMATOOL_BASIC_PASS: HTTP Basic Auth
    - ANIMATOOL_HEADERS_JSON: JSON object，合并到请求 header（推荐通用方案）

    模型/模板：
    - ANIMATOOL_UNET_NAME / ANIMATOOL_CLIP_NAME / ANIMATOOL_VAE_NAME

    输出：
    - ANIMATOOL_DOWNLOAD_IMAGES: 是否把图片保存到本地（默认 false；MCP 仍会下载 bytes 以返回 base64）
    - ANIMATOOL_OUTPUT_DIR: 保存目录（默认 ./animatool_outputs）

    预检查（仅本机/可访问文件系统场景）：
    - COMFYUI_MODELS_DIR: ComfyUI models 目录（开启预检查需要）
    - ANIMATOOL_CHECK_MODELS: 是否启用预检查（默认 true，但未设置 COMFYUI_MODELS_DIR 会自动跳过）
    """

    comfyui_url: str = field(default_factory=lambda: os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))

    timeout_s: float = field(default_factory=lambda: _get_env_float("ANIMATOOL_TIMEOUT", 600.0))
    poll_interval_s: float = field(default_factory=lambda: _get_env_float("ANIMATOOL_POLL_INTERVAL", 1.0))
    ssl_verify: bool = field(default_factory=lambda: _get_env_bool("ANIMATOOL_SSL_VERIFY", True))

    bearer_token: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_BEARER_TOKEN", "").strip())
    basic_user: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_BASIC_USER", "").strip())
    basic_pass: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_BASIC_PASS", "").strip())
    headers_json: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_HEADERS_JSON", "").strip())

    download_images: bool = field(default_factory=lambda: _get_env_bool("ANIMATOOL_DOWNLOAD_IMAGES", False))
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("ANIMATOOL_OUTPUT_DIR", "")).expanduser()
        if os.environ.get("ANIMATOOL_OUTPUT_DIR")
        else (Path.cwd() / "animatool_outputs")
    )

    # 分辨率生成：仅提供 aspect_ratio 时用
    target_megapixels: float = field(default_factory=lambda: _get_env_float("ANIMATOOL_TARGET_MP", 1.0))
    round_to: int = field(default_factory=lambda: _get_env_int("ANIMATOOL_ROUND_TO", 16))

    # 模型文件名（可被 prompt 参数覆盖）
    unet_name: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_UNET_NAME", DEFAULT_UNET_NAME))
    clip_name: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_CLIP_NAME", DEFAULT_CLIP_NAME))
    vae_name: str = field(default_factory=lambda: os.environ.get("ANIMATOOL_VAE_NAME", DEFAULT_VAE_NAME))

    comfyui_models_dir: Optional[Path] = field(
        default_factory=lambda: Path(os.environ["COMFYUI_MODELS_DIR"]).expanduser()
        if os.environ.get("COMFYUI_MODELS_DIR")
        else None
    )
    check_models: bool = field(default_factory=lambda: _get_env_bool("ANIMATOOL_CHECK_MODELS", True))

    def base_url(self) -> str:
        # urljoin 需要 base 以 '/' 结尾，否则带路径前缀时会丢路径
        return self.comfyui_url.rstrip("/") + "/"

    def get_request_headers(self) -> Dict[str, str]:
        """
        生成每次请求都要带的 headers（合并自定义 header + bearer token）。

        优先级：ANIMATOOL_HEADERS_JSON 覆盖默认；若未提供 Authorization，且设置了 bearer_token，则注入。
        """
        headers: Dict[str, str] = {}

        # 用户自定义 headers
        if self.headers_json:
            headers.update(_parse_headers_json(self.headers_json))

        # bearer token（若用户没显式提供 Authorization）
        if self.bearer_token and not any(k.lower() == "authorization" for k in headers.keys()):
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        # 默认 UA（便于排查）
        if not any(k.lower() == "user-agent" for k in headers.keys()):
            headers["User-Agent"] = "animatool-mcp/0.1.0"

        return headers

    def get_basic_auth(self) -> Optional[Tuple[str, str]]:
        if self.basic_user and self.basic_pass:
            return (self.basic_user, self.basic_pass)
        return None

    def get_model_paths(self) -> dict:
        return {
            "unet": ("diffusion_models", self.unet_name),
            "clip": ("text_encoders", self.clip_name),
            "vae": ("vae", self.vae_name),
        }

    def check_models_exist(self) -> Tuple[bool, List[str]]:
        if not self.comfyui_models_dir:
            return True, []
        models_dir = Path(self.comfyui_models_dir)
        if not models_dir.exists():
            return False, [f"ComfyUI models 目录不存在: {models_dir}"]
        missing = []
        for model_type, (subdir, filename) in self.get_model_paths().items():
            p = models_dir / subdir / filename
            if not p.exists():
                missing.append(f"{model_type}: {subdir}/{filename}")
        return len(missing) == 0, missing

