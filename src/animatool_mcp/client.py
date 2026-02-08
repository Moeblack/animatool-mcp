"""
AnimaExecutor — 独立客户端。

通过标准 ComfyUI Server API（/prompt, /history, /view, /models）
在本地或云端 ComfyUI 上执行 Anima 工作流，获取输出图片。

相比本体 (ComfyUI-AnimaTool/executor)，本模块：
- 使用 requests + 云端鉴权（Bearer / Basic / 自定义 Headers / SSL）
- 不依赖 ComfyUI custom_nodes 安装
"""
from __future__ import annotations

import base64
import json
import math
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin

import requests

from .config import AnimaToolConfig
from .history import HistoryManager
from .resources import load_workflow_template


# ============================================================
# Utilities
# ============================================================

def _round_up(x: int, base: int) -> int:
    if base <= 1:
        return x
    return int(math.ceil(x / base) * base)


def _parse_aspect_ratio(ratio: str) -> float:
    s = (ratio or "").strip()
    if ":" not in s:
        raise ValueError(f"aspect_ratio 必须形如 '16:9'，收到：{ratio!r}")
    a_str, b_str = s.split(":", 1)
    a = float(a_str.strip())
    b = float(b_str.strip())
    if a <= 0 or b <= 0:
        raise ValueError(f"aspect_ratio 两边必须 > 0，收到：{ratio!r}")
    return a / b


def estimate_size_from_ratio(
    *,
    aspect_ratio: str,
    target_megapixels: float = 1.0,
    round_to: int = 16,
) -> Tuple[int, int]:
    r = _parse_aspect_ratio(aspect_ratio)
    target_px = max(1.0, float(target_megapixels)) * 1_000_000.0
    w = int(math.sqrt(target_px * r))
    h = int(math.sqrt(target_px / r))
    w = _round_up(max(64, w), round_to)
    h = _round_up(max(64, h), round_to)
    return w, h


def align_dimension(value: int, round_to: int = 16) -> int:
    return _round_up(max(64, int(value)), round_to)


def _join_csv(*parts: str) -> str:
    cleaned: List[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        cleaned.append(s)
    return ", ".join(cleaned)


def build_anima_positive_text(prompt_json: Dict[str, Any]) -> str:
    """按 Anima 推荐顺序拼接正面提示词。"""
    return _join_csv(
        prompt_json.get("quality_meta_year_safe", ""),
        prompt_json.get("count", ""),
        prompt_json.get("character", ""),
        prompt_json.get("series", ""),
        prompt_json.get("artist", ""),
        prompt_json.get("style", ""),
        prompt_json.get("appearance", ""),
        prompt_json.get("tags", ""),
        prompt_json.get("environment", ""),
        prompt_json.get("nltags", ""),
    )


@dataclass(frozen=True)
class GeneratedImage:
    filename: str
    subfolder: str
    folder_type: str
    view_url: str
    saved_path: Optional[str] = None
    content: Optional[bytes] = None


# ============================================================
# AnimaExecutor
# ============================================================

class AnimaExecutor:
    """将结构化 JSON 注入 ComfyUI workflow 并执行，获取输出图片。"""

    _SUPPORTED_MODEL_TYPES = ("loras", "diffusion_models", "vae", "text_encoders")

    def __init__(self, config: Optional[AnimaToolConfig] = None):
        self.config = config or AnimaToolConfig()
        self._client_id = str(uuid.uuid4())
        self._workflow_template: Dict[str, Any] = load_workflow_template()

        # 远端模型路径分隔符缓存
        self._remote_model_path_sep_cache: Dict[str, str] = {}

        # 线程局部 Session（避免跨线程共享 requests.Session）
        self._tls = threading.local()

        # 生成历史管理器
        self.history = HistoryManager()

    # -------------------------
    # requests helper (云端鉴权)
    # -------------------------
    def _get_session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is None:
            s = requests.Session()
            setattr(self._tls, "session", s)
        return s

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        expect_json: bool = True,
    ):
        url = urljoin(self.config.base_url(), path)
        headers = self.config.get_request_headers()
        auth = self.config.get_basic_auth()
        session = self._get_session()
        r = session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.config.timeout_s,
            verify=self.config.ssl_verify,
            auth=auth,
        )
        r.raise_for_status()
        if not expect_json:
            return r.content
        return r.json()

    def _http_post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, json_body=payload, expect_json=True)

    def _http_get_json(self, path: str) -> Any:
        return self._request("GET", path, expect_json=True)

    def _http_get_bytes(self, path: str) -> bytes:
        return self._request("GET", path, expect_json=False)

    # -------------------------
    # Model listing
    # -------------------------
    def _detect_remote_model_path_sep(self, model_type: str) -> str:
        """探测远端 ComfyUI 返回的模型名路径分隔符。"""
        if model_type in self._remote_model_path_sep_cache:
            return self._remote_model_path_sep_cache[model_type]

        files = self._http_get_json(f"models/{model_type}")
        sep = "/"
        if isinstance(files, list):
            for it in files:
                if not isinstance(it, str):
                    continue
                if "\\" in it:
                    sep = "\\"
                    break
                if "/" in it:
                    sep = "/"
                    break

        self._remote_model_path_sep_cache[model_type] = sep
        return sep

    def _normalize_remote_model_name(self, name: str, model_type: str) -> str:
        """将用户输入的 name 规范化为远端 ComfyUI 可接受的模型名格式。"""
        import os
        s = (name or "").strip()
        if not s:
            return s

        remote_sep = self._detect_remote_model_path_sep(model_type)
        s = s.replace("/", remote_sep).replace("\\", remote_sep)

        while s.startswith(remote_sep):
            s = s[len(remote_sep):]
        while s.startswith(os.sep):
            s = s[len(os.sep):]

        return s

    def _read_lora_metadata(self, lora_name: str) -> Optional[Dict[str, Any]]:
        """读取 LoRA 的 sidecar 元数据文件（同名 .json）。仅本地可用。"""
        import os
        if not self.config.comfyui_models_dir:
            return None

        models_dir = Path(self.config.comfyui_models_dir)
        clean_name = lora_name.strip().replace("/", os.sep).replace("\\", os.sep)
        while clean_name.startswith(os.sep):
            clean_name = clean_name[1:]

        search_paths = [
            models_dir / "loras" / f"{clean_name}.json",
            models_dir / "loras" / f"{os.path.splitext(clean_name)[0]}.json",
        ]

        for meta_path in search_paths:
            if meta_path and meta_path.exists():
                try:
                    return json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return None

    def list_models(self, model_type: str) -> List[Dict[str, Any]]:
        """列出 ComfyUI 模型文件。"""
        model_type = (model_type or "").strip()
        if model_type not in self._SUPPORTED_MODEL_TYPES:
            raise ValueError(f"不支持的 model_type={model_type!r}，仅支持：{self._SUPPORTED_MODEL_TYPES}")

        files = self._http_get_json(f"models/{model_type}")

        if not isinstance(files, list):
            raise RuntimeError(f"ComfyUI /models/{model_type} 返回异常：{files!r}")

        results: List[Dict[str, Any]] = []
        for raw_name in files:
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue

            normalized_name = raw_name.replace("\\", "/")
            item: Dict[str, Any] = {"name": normalized_name}

            if model_type == "loras":
                meta = self._read_lora_metadata(raw_name)
                if not meta:
                    # 强制要求：不提供 json sidecar 的 LoRA 不允许被 list 出来
                    continue
                item["metadata"] = meta

            results.append(item)

        return results

    # -------------------------
    # LoRA injection
    # -------------------------
    def _inject_loras(self, wf: Dict[str, Any], loras: Any) -> None:
        """在 UNET 与 KSampler(model) 之间注入多 LoRA（仅 UNET）。"""
        if not loras:
            return
        if not isinstance(loras, list):
            raise ValueError("loras 必须是数组：[{name, weight}, ...]")

        if "19" not in wf or "inputs" not in wf["19"] or "model" not in wf["19"]["inputs"]:
            raise RuntimeError("workflow_template.json 缺少 KSampler(19).inputs.model，无法注入 LoRA")

        prev_model = wf["19"]["inputs"]["model"]

        numeric_ids = [int(k) for k in wf.keys() if str(k).isdigit()]
        next_id = (max(numeric_ids) + 1) if numeric_ids else 1

        for i, lora in enumerate(loras):
            if not isinstance(lora, dict):
                continue
            name = str(lora.get("name") or "").strip()
            if not name:
                continue
            name = self._normalize_remote_model_name(name, "loras")
            weight = float(lora.get("weight", 1.0))

            node_id = str(next_id + i)
            wf[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": prev_model,
                    "lora_name": name,
                    "strength_model": weight,
                },
            }
            prev_model = [node_id, 0]

        wf["19"]["inputs"]["model"] = prev_model

    # -------------------------
    # Core workflow injection
    # -------------------------
    def _inject(self, prompt_json: Dict[str, Any]) -> Dict[str, Any]:
        wf = deepcopy(self._workflow_template)

        # 模型文件
        clip_name = prompt_json.get("clip_name") or self.config.clip_name
        unet_name = prompt_json.get("unet_name") or self.config.unet_name
        vae_name = prompt_json.get("vae_name") or self.config.vae_name

        wf["45"]["inputs"]["clip_name"] = str(clip_name)
        wf["44"]["inputs"]["unet_name"] = str(unet_name)
        wf["15"]["inputs"]["vae_name"] = str(vae_name)

        # LoRA 注入
        self._inject_loras(wf, prompt_json.get("loras"))

        # 文本
        positive = (prompt_json.get("positive") or "").strip()
        if not positive:
            positive = build_anima_positive_text(prompt_json)
        negative = (prompt_json.get("neg") or prompt_json.get("negative") or "").strip()

        wf["11"]["inputs"]["text"] = positive
        wf["12"]["inputs"]["text"] = negative

        # 分辨率
        width = prompt_json.get("width")
        height = prompt_json.get("height")
        aspect_ratio = (prompt_json.get("aspect_ratio") or "").strip()
        round_to = int(prompt_json.get("round_to") or self.config.round_to)

        if (width is None or height is None) and aspect_ratio:
            w, h = estimate_size_from_ratio(
                aspect_ratio=aspect_ratio,
                target_megapixels=float(prompt_json.get("target_megapixels") or self.config.target_megapixels),
                round_to=round_to,
            )
            width, height = w, h
        elif width is not None and height is not None:
            width = align_dimension(width, round_to)
            height = align_dimension(height, round_to)

        if width is None or height is None:
            width, height = 1024, 1024

        wf["28"]["inputs"]["width"] = int(width)
        wf["28"]["inputs"]["height"] = int(height)
        wf["28"]["inputs"]["batch_size"] = int(prompt_json.get("batch_size") or 1)

        # 采样参数
        seed = prompt_json.get("seed")
        if seed is None:
            seed = int.from_bytes(uuid.uuid4().bytes[:4], "big", signed=False)
        wf["19"]["inputs"]["seed"] = int(seed)

        wf["19"]["inputs"]["steps"] = int(prompt_json.get("steps") or wf["19"]["inputs"]["steps"])
        wf["19"]["inputs"]["cfg"] = float(prompt_json.get("cfg") or wf["19"]["inputs"]["cfg"])
        wf["19"]["inputs"]["sampler_name"] = str(prompt_json.get("sampler_name") or wf["19"]["inputs"]["sampler_name"])
        wf["19"]["inputs"]["scheduler"] = str(prompt_json.get("scheduler") or wf["19"]["inputs"]["scheduler"])
        wf["19"]["inputs"]["denoise"] = float(prompt_json.get("denoise") or wf["19"]["inputs"]["denoise"])

        wf["52"]["inputs"]["filename_prefix"] = str(prompt_json.get("filename_prefix") or wf["52"]["inputs"]["filename_prefix"])

        return wf

    # -------------------------
    # Health check
    # -------------------------
    def check_comfyui_health(self) -> Tuple[bool, str]:
        try:
            self._http_get_json("system_stats")
            return True, f"ComfyUI 运行正常 ({self.config.comfyui_url})"
        except Exception as e:
            msg = str(e)
            if "Connection refused" in msg or "连接" in msg:
                return False, (
                    f"无法连接到 ComfyUI ({self.config.comfyui_url})\n"
                    f"请确认：\n"
                    f"  1. ComfyUI 已启动且可访问\n"
                    f"  2. 地址和端口正确（COMFYUI_URL）\n"
                    f"  3. 云端有鉴权时已设置 token/header\n"
                )
            if "timeout" in msg.lower() or "超时" in msg:
                return False, f"连接 ComfyUI 超时 ({self.config.comfyui_url})"
            return False, f"ComfyUI 连接错误: {msg}"

    def check_models(self) -> Tuple[bool, str]:
        if not self.config.check_models or not self.config.comfyui_models_dir:
            return True, "模型检查已跳过（未配置 COMFYUI_MODELS_DIR）"
        ok, missing = self.config.check_models_exist()
        if ok:
            return True, "所有模型文件已就绪"
        missing_str = "\n".join(f"  - {m}" for m in missing)
        return False, f"缺少以下模型文件：\n{missing_str}"

    # -------------------------
    # ComfyUI execution
    # -------------------------
    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        payload = {"prompt": prompt, "client_id": self._client_id}
        try:
            resp = self._http_post_json("prompt", payload)
        except Exception as e:
            is_ok, health_msg = self.check_comfyui_health()
            if not is_ok:
                raise RuntimeError(health_msg) from e
            raise

        prompt_id = str(resp.get("prompt_id") or "")
        if not prompt_id:
            err = resp.get("error") or resp.get("node_errors")
            if err:
                raise RuntimeError(f"ComfyUI 执行错误：{err}")
            raise RuntimeError(f"ComfyUI /prompt 返回异常：{resp}")
        return prompt_id

    def wait_history(self, prompt_id: str) -> Dict[str, Any]:
        deadline = time.time() + float(self.config.timeout_s)
        last = None
        while time.time() < deadline:
            data = self._http_get_json(f"history/{prompt_id}")
            last = data
            if isinstance(data, dict) and prompt_id in data:
                return data[prompt_id]
            time.sleep(float(self.config.poll_interval_s))
        raise TimeoutError(f"等待 ComfyUI 生成超时：prompt_id={prompt_id}, last={last}")

    def _extract_images(self, history_item: Dict[str, Any]) -> List[GeneratedImage]:
        outputs = history_item.get("outputs") or {}
        images: List[GeneratedImage] = []
        for node_id, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for im in node_out.get("images") or []:
                filename = str(im.get("filename") or "")
                subfolder = str(im.get("subfolder") or "")
                folder_type = str(im.get("type") or "output")
                if not filename:
                    continue
                qs = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
                view_url = urljoin(self.config.base_url(), f"view?{qs}")
                images.append(
                    GeneratedImage(
                        filename=filename,
                        subfolder=subfolder,
                        folder_type=folder_type,
                        view_url=view_url,
                        saved_path=None,
                    )
                )
        return images

    def _download_images(self, images: List[GeneratedImage]) -> List[GeneratedImage]:
        out_dir = Path(self.config.output_dir)
        downloaded: List[GeneratedImage] = []

        if self.config.download_images:
            out_dir.mkdir(parents=True, exist_ok=True)

        for im in images:
            content = self._http_get_bytes(im.view_url)

            saved_path = None
            if self.config.download_images:
                sub_dir = out_dir / (im.subfolder or "")
                sub_dir.mkdir(parents=True, exist_ok=True)
                dst = sub_dir / im.filename
                dst.write_bytes(content)
                saved_path = str(dst)

            downloaded.append(
                GeneratedImage(
                    filename=im.filename,
                    subfolder=im.subfolder,
                    folder_type=im.folder_type,
                    view_url=im.view_url,
                    saved_path=saved_path,
                    content=content,
                )
            )
        return downloaded

    def _get_mime_type(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/png")

    def generate(self, prompt_json: Dict[str, Any]) -> Dict[str, Any]:
        models_ok, models_msg = self.check_models()
        if not models_ok:
            raise RuntimeError(models_msg)

        prompt = self._inject(prompt_json)
        prompt_id = self.queue_prompt(prompt)
        history_item = self.wait_history(prompt_id)
        images = self._extract_images(history_item)
        images = self._download_images(images)

        images_data = []
        for im in images:
            mime_type = self._get_mime_type(im.filename)
            b64 = base64.b64encode(im.content).decode("ascii") if im.content else None
            images_data.append(
                {
                    "filename": im.filename,
                    "subfolder": im.subfolder,
                    "type": im.folder_type,
                    "url": im.view_url,
                    "view_url": im.view_url,
                    "file_path": im.saved_path,
                    "saved_path": im.saved_path,
                    "base64": b64,
                    "mime_type": mime_type,
                    "data_url": f"data:{mime_type};base64,{b64}" if b64 else None,
                    "markdown": f"![{im.filename}]({im.view_url})",
                }
            )

        actual_seed = int(prompt["19"]["inputs"]["seed"])
        actual_width = int(prompt["28"]["inputs"]["width"])
        actual_height = int(prompt["28"]["inputs"]["height"])

        result = {
            "success": True,
            "prompt_id": prompt_id,
            "positive": prompt["11"]["inputs"]["text"],
            "negative": prompt["12"]["inputs"]["text"],
            "seed": actual_seed,
            "width": actual_width,
            "height": actual_height,
            "images": images_data,
        }

        # 记录到历史
        record = self.history.add(
            params=prompt_json,
            positive_text=result["positive"],
            negative_text=result["negative"],
            prompt_id=prompt_id,
            seed=actual_seed,
            width=actual_width,
            height=actual_height,
        )
        result["history_id"] = record.id

        return result
