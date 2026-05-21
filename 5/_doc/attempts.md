# agent0.py 測試失敗紀錄

## 環境

- macOS arm64, Ollama via Homebrew
- model 設定在 agent0.py 第 14 行：`MODEL = "minimax-m2.5:cloud"`

## 嘗試記錄

### 1. minimax-m2.5:cloud
- **結果**: `{"error": "unauthorized"}`
- **原因**: 此為雲端模型，需要 Ollama 雲端授權（API key 或登入）
- **時間**: 2026-05-21

### 2. tinyllama (637MB)
- **結果**: LLM 有回應，但無法正確使用 XML 工具格式（`<shell>`、`<write_file>`、`<read_file>`）
- **原因**: 模型太小，不擅長遵循結構化工具呼叫指令
- **時間**: 2026-05-21

### 3. llama3.2:3b (~2GB)
- **結果**: 下載過久，使用者中斷
- **時間**: 2026-05-21

## 問題總結

1. **agent0.py 依賴 Ollama API** — 需要本地 Ollama 服務執行於 localhost:11434
2. **原始模型 minimax-m2.5:cloud 需授權** — 無法在無網路/無授權環境使用
3. **小而快的本地模型不夠力** — tinyllama 等小型模型無法正確輸出 XML 工具標籤
4. **較大的本地模型下載耗時** — 3B 以上模型需 GB 級下載
5. **test.sh 未考慮互動授權** — agent0.py 對每個 shell 指令都需要使用者輸入 `y` 核可

### 4. Apple MLX + Qwen2.5-1.5B-Instruct-4bit (✅ 成功)
- **結果**: 成功建立 blog2/ 專案，包含 app.js、views/index.ejs、views/new.ejs
- **模型大小**: 839MB（4-bit 量化）
- **下載時間**: ~2.5 分鐘
- **優點**: 使用 Apple MLX 框架，無需 Ollama；M4 晶片推理速度快；模型能正確輸出 XML 工具標籤
- **缺點**: 1.5B 模型產生的程式碼品質有限（EJS 範本有語法錯誤），需較大模型改善
- **時間**: 2026-05-21

## 問題總結

1. **agent0.py 原依賴 Ollama API** — 已改為 Apple MLX 本機推理
2. **原始模型 minimax-m2.5:cloud 需授權** — 已改用無授權需求的 HF 開源模型
3. **小而快的本地模型不夠力** — Qwen2.5-1.5B 足夠輸出 XML，但程式碼品質有限
4. **較大的本地模型下載耗時** — MLX 4-bit 量化模型（~839MB）是良好折衷
5. **test.sh 未考慮互動授權** — 已加入 `--auto` 參數支援自動化測試
6. **工具執行順序問題** — `<end/>` 檢查需在工具執行之後，否則工具不會被執行

## 現行方案

- **模型**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`（MLX 4-bit 量化）
- **框架**: Apple MLX（mlx-lm）
- **執行**: `python3 agent0.py`（首次自動下載模型至 HF 快取）
