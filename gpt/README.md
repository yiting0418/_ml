# microgpt_bilingual_gpu

基於 [Karpathy microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 的雙語 GPT（繁體中文 + English），使用 PyTorch 實作，支援 GPU 加速。

從零實作 GPT-2 架構，無依賴任何高階 LLM 框架（如 HuggingFace Transformers）。

## 快速開始

```bash
cd gpt
pip install torch
python microgpt_bilingual_gpu.py
```

首次執行會自動下載英文名字資料集。在 MPS (Apple Silicon) 上約 2 分鐘完成訓練。

## 資料集

| 語言 | 類型 | 數量 | 長度 |
|------|------|------|------|
| 繁體中文 | 生活短句（問候、哲理、日常） | 94 句 | 8-15 字 |
| English | 名字（makemore dataset） | 94 個 | 3-12 字母 |

中英文比例 1:1 平衡，每步隨機取樣。

## 架構

標準 GPT-2 架構：

```
Token Embedding + Position Embedding
    → RMSNorm
    → [Transformer Block] × 3
        → Multi-Head Causal Self-Attention
        → RMSNorm + MLP (ReLU)
        → Residual Connections
    → RMSNorm
    → LM Head (vocab projection)
```

## 超參數

| 參數 | 值 |
|------|-----|
| n_embd | 64 |
| n_layer | 3 |
| n_head | 4 |
| block_size | 20 |
| batch_size | 128 |
| learning_rate | 5e-4 (cosine decay to 1e-5) |
| optimizer | AdamW |
| weight_decay | 0.05 |
| dropout | 0.1 |
| grad_clip | 1.0 |
| training steps | 2000 |

## 輸出範例

訓練完成後自動生成 30 筆樣本：

```
[中文] 好好休息明天才有精神
[中文] 大自然就是最好的醫生
[中文] 吃飽了覺得好滿足
[中文] 慢慢走小心安全
[EN]   kemorah
[EN]   josephyne
[EN]   phoenix
[EN]   avaneesh
```

## 參考

- [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Karpathy 的 200 行純 Python GPT
- [micrograd](https://github.com/karpathy/micrograd) — 自動微分引擎
- [makemore](https://github.com/karpathy/makemore) — 名字資料集
- [nanoGPT](https://github.com/karpathy/nanoGPT) — 更完整的 GPT 實作
