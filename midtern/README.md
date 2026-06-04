# Tixcraft Captcha Auto Solver (機器學習期中專案)

![AI Inference](https://img.shields.io/badge/Inference-On--Device-green)
![Tech](https://img.shields.io/badge/Tech-PyTorch%20%7C%20ONNX%20%7C%20WASM-blue)

本專案旨在實作深度學習模型於真實場景的端到端 (End-to-End) 部署。系統針對拓元售票系統的圖形驗證碼進行自動辨識，並將推論過程完全移植至瀏覽器本地端執行，實現無伺服器 (Serverless) 的邊緣運算應用。

> **學術用途聲明 (Disclaimer)**
> 本專案為機器學習課程之期中專案，相關程式碼、模型架構與技術探討僅供學術研究與技術交流使用。嚴禁將本系統應用於任何破壞平台公平性、商業牟利或干擾售票系統正常運作之行為。使用者若因不當操作衍生任何法律爭議，概與本專案開發者無關，請務必遵守相關法律與平台使用規範。

## 系統核心特性
- 高效能推論：採用輕量化的 ResNet-18 模型，針對 4 位元英數字進行多標籤分類 (Multi-label Classification)，精準度達 97% 以上。
- 邊緣運算 (Edge Computing)：導入 ONNX Runtime Web 技術，利用 WebAssembly (WASM) 於瀏覽器內直接加載神經網路權重，免除 API 網路延遲與伺服器建置成本。
- 靜默自動化：整合 Chrome Extension 的 MutationObserver API，精準捕捉 DOM 樹變化，於圖片載入完成瞬間進行張量處理與推論。

---

## 開發歷程與技術克服 (Development History)

### 第一階段：模型開發與訓練 (Model Training)
- 架構選擇：使用預訓練的 ResNet-18 作為特徵提取器 (Feature Extractor)，修改全連接層以適應 36 類 (0-9, a-z) 的輸出。
- 訓練策略：採用 Seed Hunting 穩定隨機梯度下降的初始狀態，歷經 60 Epochs 的微調 (Fine-tuning)，確保模型收斂並防止過擬合 (Overfitting)。

### 第二階段：API 遠端推論架構 (Client-Server Model)
- 實作方式：以 FastAPI 建置 Python 後端推論伺服器，並透過 Ngrok 進行內網穿透。
- 資料流優化：為避免重新請求圖片網址導致驗證碼刷新失效，前端擴充功能採用 HTML5 Canvas 擷取圖片原始像素，將 Base64 編碼傳遞至後端進行推論。

### 第三階段：無伺服器邊緣運算架構 (Serverless Edge Computing)
為追求最低延遲並保護使用者隱私，將系統重構為純前端推論架構，期間克服了以下技術與環境限制：
1. 模型轉換：將 PyTorch (`.pth`) 靜態圖導出為 `.onnx` 跨平台格式。
2. 突破擴充功能安全策略 (CSP Bypass)：
   - Chrome Manifest V3 嚴格限制動態載入外部腳本 (`import()`)。為此，捨棄最新版引擎，降級採用 v1.14.0 穩定版 ONNX Runtime。
   - 將 WASM 引擎二進制檔 (`ort-wasm.wasm`) 直接封裝進擴充功能內存，強制以單執行緒 (Single-thread) 模式運行，成功繞過瀏覽器網路攔截。
3. ONNX IR 版本向下相容：因引擎降級導致無法解析最新 PyTorch 導出的模型格式 (IR Version 10)，透過撰寫 Python 轉換腳本，直接修改並覆寫模型定義檔，成功將 IR Version 降級至 8，解決環境不匹配之報錯。

---

## 技術說明
- Deep Learning Framework: PyTorch, Torchvision
- Model Export & Manipulation: ONNX, ONNX-Script
- Edge Inference Engine: ONNX Runtime Web (WASM)
- Browser Integration: Chrome Extension API (Manifest V3), JavaScript, HTML5 Canvas