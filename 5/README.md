# Agent0 — 本地 AI 代理程式

基於 **Apple MLX** 的命令列 AI 代理，搭載 `Qwen2.5-1.5B-Instruct-4bit` 量化模型，完全本地執行，無需雲端 API。

## 系統架構

```
使用者輸入 → 記憶模組 → MLX 模型(Qwen2.5-1.5B) → XML 工具解析 → 執行工具 → 回饋迴圈
                                  ↑                                    |
                                  └────────── 最多 5 次工具迴圈 ────────┘
```

- 使用 `mlx-lm` 取代 Ollama，模型約 839MB（HF Hub 快取）
- 同步執行，無需 `asyncio` / `aiohttp`
- 工具迴圈上限 5 次，防止無限迴圈

## 安全機制

### 沙盒路徑保護

所有檔案操作經 `os.path.commonpath()` 強制檢查，防止 `../../etc/passwd` 等目錄穿越：

| 區域 | read_file / write_file | shell 命令 |
|------|----------------------|-----------|
| 工作區內 | ✅ 自動放行 | ❓ 每次詢問 |
| 工作區外 | ❓ 詢問核可 | ❓ 每次詢問 |

- 授權預設為 `N`，拒絕時回傳 `Error: Permission denied by user.`
- 所有 shell 命令執行前完整印出，供使用者審閱
- `--auto` 參數可跳過所有授權檢查

## 工具系統

模型透過 XML 標籤呼叫工具：

| 工具 | 格式 | 說明 |
|------|------|------|
| `shell` | `<shell>command</shell>` | 執行 shell 命令（cwd 鎖定於工作區） |
| `read_file` | `<read_file path="..."/>` | 讀取檔案 |
| `write_file` | `<write_file path="...">content</write_file>` | 寫入檔案 |
| `end` | `<end/>` | 標記任務結束 |

工具執行順序：先解析所有工具標籤 → 依序執行 → 再檢查 `<end/>` 結束標記（修正自 v2，原順序導致工具永不執行）。

## 記憶系統

- **對話記錄**：保留最近 5 輪互動，做為短期上下文
- **關鍵資訊提取**：每次工具執行後自動提取需長期記憶的資訊，存入 `key_info` 串列
- 可透過 `/memory` 指令檢視當前記憶內容

## 使用方式

```bash
# 一般模式（互動授權）
python3 agent0.py

# 自動模式（跳過授權，適合測試）
python3 agent0.py --auto

# 指定工作目錄
python3 agent0.py --workspace /path/to/project

# 測試腳本（含自動授權 + npm install + 伺服器測試）
bash test.sh
```

### 指令

| 指令 | 說明 |
|------|------|
| `/quit` `/exit` `/q` | 離開程式 |
| `/memory` | 顯示長期記憶的關鍵資訊 |

## 限制

- Qwen2.5-1.5B 能正確輸出 XML 工具格式，但產生的程式碼（尤其是 EJS/HTML）可能有語法錯誤
- 模型會一次性輸出所有工具標籤後加 `<end/>`，而非逐步執行（無法處理需要根據中間結果決策的情境）
- 預設 `max_tokens=3072`，若檔案內容很長可能被截斷
- 授權詢問為同步阻塞，不適合無人值守場景（可使用 `--auto` 解決）

## 開發紀錄

歷次嘗試記錄於 `_doc/attempts.md`，最新變更請見 `_doc/changelog.md`。
