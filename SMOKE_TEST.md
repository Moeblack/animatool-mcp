# 冒烟测试（本地 / 云端）

本文件给出两套最短验证路径：**本地 ComfyUI（无鉴权）** 与 **云端/反代（有鉴权/自定义 Header）**。

---

## A. 本地 ComfyUI（无鉴权）

### 0) 前置

- ComfyUI Server 正常运行（例如 `http://127.0.0.1:8188`）
- 云端/本机的模型文件已放到 ComfyUI 的 `models/` 对应目录（文件名匹配工作流模板）

### 1) 安装

```bash
pip install comfyui-animatool
```

### 2) CLI 一键验证

在仓库根目录执行：

```bash
animatool-generate --comfyui-url http://127.0.0.1:8188 --json-file animatool-mcp/examples/payload.example.json
```

期望：输出 JSON 中 `success=true`，并包含 `images[0].base64` 与 `images[0].mime_type`。

### 3) 可选：保存图片到本地

```bash
set ANIMATOOL_DOWNLOAD_IMAGES=true
set ANIMATOOL_OUTPUT_DIR=./animatool_outputs
animatool-generate --comfyui-url http://127.0.0.1:8188 --json-file animatool-mcp/examples/payload.example.json
```

---

## B. 云端/反代（有鉴权/自定义 Header）

### 0) 快速连通性检查

浏览器或 curl 访问：

- `COMFYUI_URL/system_stats`

如果这一步都不通，MCP/CLI 一定会失败。

### 1) Bearer Token

```bash
animatool-generate ^
  --comfyui-url https://your-host.example.com:8188 ^
  --bearer-token YOUR_TOKEN ^
  --json-file animatool-mcp/examples/payload.example.json
```

### 2) 自定义 Header（推荐）

适配 Cloudflare Access / API Gateway / 自定义鉴权等：

```bash
animatool-generate ^
  --comfyui-url https://your-host.example.com:8188 ^
  --headers-json "{\"X-API-Key\":\"xxx\"}" ^
  --json-file animatool-mcp/examples/payload.example.json
```

### 3) 常见坑：/prompt 通了但 /view 被拦

该工具生成流程是：`/prompt` → `/history/<id>` → `/view?...` 拉图。

如果你的反向代理只放行了 `/prompt`，而 `/view` 返回 401/403，会表现为：

- prompt_id 有了
- 但最终报错“取图失败”或下载图片时报 401/403

解决：把 `/view`、`/history` 与 `/system_stats` 一并纳入同样的鉴权/放行规则。

