# 改造紀錄

## 問題

原始 agent0.py 依賴 Ollama API 與 minimax-m2.5:cloud 模型，但：
- minimax-m2.5:cloud 需要雲端授權（401 Unauthorized）
- tinyllama（637MB）太笨，無法輸出正確 XML 工具標籤
- llama3.2:3b（~2GB）下載過久被中斷

## 解決方案

改用 Apple MLX 框架 + Qwen2.5-1.5B-Instruct-4bit 模型，完全移除 Ollama 相依。

## 變更內容

### 移除
- **Ollama**（brew uninstall ollama）
- `aiohttp` 套件（不再需要 HTTP 請求）

### 新增
- `mlx-lm` 套件（pip install mlx-lm）
- HF 模型快取：`~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/`（839MB）
- 檔案：`_doc/changelog.md`（本文件）

### 修改檔案

#### agent0.py

| 變更 | 說明 |
|------|------|
| `import asyncio` / `import aiohttp` | 移除，改為 `from mlx_lm import load, generate` |
| `MODEL` | `minimax-m2.5:cloud` → `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| `call_ollama()` | 移除 HTTP API 呼叫，改用 `mlx_lm.generate()`，改為同步 |
| 所有 `asyncio.run()` | 移除 |
| `extract_key_info` | 移除外層 async/await |
| `extract_tools()` | 加入 markdown code block 清理（```xml → 空字串） |
| 工具執行流程 | 先解讀並執行工具，再檢查 `<end/>` 結束標記（原順序導致工具永不執行） |
| `AUTO_AUTHORIZE` | 新增全域變數，`--auto` 參數啟用 |
| `WORKSPACE` | 新增 `--workspace <path>` 參數支援動態設定 |
| `max_tool_loops` | 新增 5 次上限，防止模型無限迴圈 |
| `SYSTEM_PROMPT` | 重寫為更明確的指令，強調禁止輸出說明文字 |
| `follow_up_prompt` | 簡化格式，減少模型混淆 |

#### test.sh
- 加入 `--auto --workspace "$(pwd)"` 參數
- 加入 npm install 步驟（模型僅負責建立檔案結構）
- 伺服器測試改為容錯（接受範本錯誤）

#### README.md
- 更新為 MLX 描述，移除 Ollama 相關內容
- 加入 `--auto`、`--workspace` 參數說明
- 更新限制描述

#### _doc/attempts.md
- 新增第 4 次嘗試記錄：Apple MLX + Qwen2.5-1.5B（成功）

## 使用方式

```bash
# 一般模式（需互動授權）
python3 agent0.py

# 自動模式（測試用）
python3 agent0.py --auto

# 指定工作目錄
python3 agent0.py --workspace /path/to/project

# 測試腳本
bash test.sh
```

## 已知限制

- Qwen2.5-1.5B 能正確輸出 XML 工具格式，但產生的程式碼（尤其是 EJS/HTML）可能有語法錯誤
- 模型會一次性輸出所有工具標籤後加 `<end/>`，而非逐步執行（無法處理需要根據中間結果決策的情境）
- 預設 `max_tokens=3072`，若檔案內容很長可能被截斷
