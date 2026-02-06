# animatool-mcp

> [!WARNING]
> **暂不支持 Cherry Studio**  
> Cherry Studio 的 MCP 客户端未正确处理 `ImageContent` 类型，会将 base64 图片数据作为纯文本字符串返回，而非渲染为图片。请使用正确实现 MCP 规范的客户端。

一个**独立的 MCP Server + Python 客户端**：通过**标准 ComfyUI Server API**（`/prompt`、`/history/<id>`、`/view?...`）在本地或云端 ComfyUI 上执行 Anima 工作流，并把生成图片以 **base64** 形式返回给 Cursor/Claude 等 MCP 客户端原生显示。

> 重要：本项目不需要安装到 `ComfyUI/custom_nodes/`。但你仍然需要一台可访问的 ComfyUI Server（本机或云端），并且云端已经把模型权重放到 `models/` 下（按文件名加载）。

---

## 安装

```bash
pip install comfyui-animatool
```

---

## Cursor 配置（MCP）

在你的项目中创建/编辑 `.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "animatool": {
      "command": "animatool-mcp",
      "env": {
        "COMFYUI_URL": "http://127.0.0.1:8188"
      }
    }
  }
}
```

然后重启 Cursor，确保 MCP 状态为 Running。

---

## 云端/远程 ComfyUI（推荐做法）

### COMFYUI_URL 该填什么？

- **本机默认**：`http://127.0.0.1:8188`
- **带路径前缀的反代**：例如 `https://example.com/comfy/`（注意建议以 `/` 结尾，避免丢路径前缀）

### 1) 最佳实践：用隧道/VPN 让 ComfyUI 变成“内网可达”

ComfyUI 默认几乎没有鉴权能力，把它直接暴露到公网非常危险。建议使用：

- Tailscale / ZeroTier / Cloudflare Tunnel / 自建 VPN
- 或者反向代理 + HTTPS + 鉴权（Bearer / Basic / 自定义 Header）

### 2) 配置鉴权（可选）

本项目支持“有些云端需要、有些不需要”的混合场景。你可以在 `.cursor/mcp.json` 里通过 `env` 注入：

#### Bearer Token

```json
{
  "mcpServers": {
    "animatool": {
      "command": "animatool-mcp",
      "env": {
        "COMFYUI_URL": "https://your-host.example.com:8188",
        "ANIMATOOL_BEARER_TOKEN": "YOUR_TOKEN"
      }
    }
  }
}
```

#### Basic Auth

```json
{
  "mcpServers": {
    "animatool": {
      "command": "animatool-mcp",
      "env": {
        "COMFYUI_URL": "https://your-host.example.com:8188",
        "ANIMATOOL_BASIC_USER": "user",
        "ANIMATOOL_BASIC_PASS": "pass"
      }
    }
  }
}
```

#### 自定义 Header（通用，推荐）

```json
{
  "mcpServers": {
    "animatool": {
      "command": "animatool-mcp",
      "env": {
        "COMFYUI_URL": "https://your-host.example.com:8188",
        "ANIMATOOL_HEADERS_JSON": "{\"X-API-Key\":\"xxx\",\"CF-Access-Client-Id\":\"...\",\"CF-Access-Client-Secret\":\"...\"}"
      }
    }
  }
}
```

> Windows/JSON 转义提示：`ANIMATOOL_HEADERS_JSON` 本身是一个 JSON 字符串，所以内部的双引号需要转义（如上所示）。

### 3) 自签名证书/SSL 校验

如果你的云端是自签名证书，可以临时关闭校验（不推荐长期使用）：

```json
{
  "mcpServers": {
    "animatool": {
      "command": "animatool-mcp",
      "env": {
        "COMFYUI_URL": "https://your-host.example.com:8188",
        "ANIMATOOL_SSL_VERIFY": "false"
      }
    }
  }
}
```

---

## CLI（可选）

用于本地自测（不依赖 MCP 客户端）：

```bash
animatool-generate --comfyui-url http://127.0.0.1:8188 --json-file payload.json
```

---

## 环境变量速查

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `COMFYUI_URL` | ComfyUI Server base URL | `http://127.0.0.1:8188` |
| `ANIMATOOL_TIMEOUT` | 请求超时秒 | `600` |
| `ANIMATOOL_POLL_INTERVAL` | 轮询间隔秒 | `1` |
| `ANIMATOOL_SSL_VERIFY` | SSL 校验 | `true` |
| `ANIMATOOL_BEARER_TOKEN` | Bearer Token | *(空)* |
| `ANIMATOOL_BASIC_USER` / `ANIMATOOL_BASIC_PASS` | Basic Auth | *(空)* |
| `ANIMATOOL_HEADERS_JSON` | 自定义 Header JSON | *(空)* |
| `ANIMATOOL_DOWNLOAD_IMAGES` | 是否保存图片到本地 | `false` |
| `ANIMATOOL_OUTPUT_DIR` | 本地保存目录 | `./animatool_outputs` |
| `COMFYUI_MODELS_DIR` | models 目录（用于本机预检查） | *(空)* |
| `ANIMATOOL_CHECK_MODELS` | 是否启用预检查 | `true` |

## 常见问题

### Q: 报错连接不上 /prompt？

- 检查 `COMFYUI_URL` 是否可达（云端建议先在浏览器访问 `COMFYUI_URL/system_stats`）
- 如果有鉴权，请确认 `/prompt`、`/history`、`/view` 都在同一套鉴权规则下
  - 常见坑：反代只给 `/prompt` 放行了，但 `/view` 被 401/403 拦住，会导致“生成成功但取图失败”

### Q: 报错找不到模型？

本工具提交的是一个包含 `UNETLoader/CLIPLoader/VAELoader` 的工作流模板，加载的是云端磁盘上的文件名。请确认云端 `models/` 下至少存在：

- `models/diffusion_models/anima-preview.safetensors`
- `models/text_encoders/qwen_3_06b_base.safetensors`
- `models/vae/qwen_image_vae.safetensors`

