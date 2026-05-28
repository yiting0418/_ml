# nn0 — 從零開始的神經網路學習套件

`nn0.py` 是一個**純 Python** 實作的自動微分引擎（autograd engine）與小型深度學習工具組，專為**理解機器學習底層原理**而設計。它沒有使用任何第三方數值運算庫（如 NumPy、PyTorch），讓你能真正看懂每個環節在做什麼。

## 核心概念

### 計算圖 (Computation Graph) 與自動微分 (Autograd)

`Value` 類別是整個套件的核心。它不只儲存一個數值（`.data`），還會追蹤：

- **運算歷程**：這個值是從哪些值、透過什麼運算產生的（`_children`）
- **局部梯度**：每個運算對其輸入的偏微分（`_local_grads`）

當你寫 `d = (a + b) * c`，背後建立了一個計算圖：

```
    a
     \
      [+] --> 中間節點 --> [*] --> d
     /                    /
    b                    c
```

呼叫 `d.backward()` 時，它會走訪整個計算圖，用**連鎖律（chain rule）** 一路把梯度傳回每個葉節點。

```python
from nn0 import Value

a = Value(2.0)
b = Value(3.0)
d = (a + b) * Value(4.0)
d.backward()
print(a.grad)  # ∂d/∂a = 4.0
print(b.grad)  # ∂d/∂b = 4.0
```

### 支援的運算

| 運算 | 說明 | 梯度 |
|------|------|------|
| `+` | 加法 | 上游梯度直接通過 (乘以 1) |
| `*` | 乘法 | `∂/∂x = y`, `∂/∂y = x` |
| `**` | 乘冪 | `∂/∂x = n * x^(n-1)` |
| `log()` | 自然對數 | `∂/∂x = 1/x` |
| `exp()` | 指數 | `∂/∂x = e^x` |
| `relu()` | ReLU 激勵函數 | `∂/∂x = 1` (若 x>0)，否則 0 |

### Adam 優化器

`Adam` 類別實作了 Adaptive Moment Estimation 演算法，結合了 Momentum 和 RMSProp 的優點：

- **m**：梯度的一階動量（慣性項）
- **v**：梯度的二階動量（適應性學習率）
- **偏差修正**：解決初期估計偏零的問題
- **學習率衰減**：支援 `lr_override` 實現線性衰減

```python
from nn0 import Adam

w = Value(0.0)
b = Value(0.0)
opt = Adam([w, b], lr=0.1)

# 每次迭代：
# loss.backward() → 計算梯度
# opt.step()      → 更新參數並清空梯度
```

## 工具函數

### `linear(x, w)` — 線性層 (矩陣乘法)

```python
from nn0 import linear, Value

x = [Value(1.0), Value(2.0)]
w = [[Value(0.5), Value(0.3)],  # 2 個輸出神經元
     [Value(0.1), Value(0.4)]]  # 每個神經元有 2 個權重
y = linear(x, w)  # 回傳 [y0, y1]
```

### `softmax(logits)` — Softmax 激勵函數

將任意實數向量轉換為機率分佈（總和為 1）：

```python
from nn0 import softmax, Value

logits = [Value(2.0), Value(1.0), Value(0.1)]
probs = softmax(logits)  # 每項在 [0,1] 之間，總和 = 1
```

實作使用 **數值穩定技巧**：先減去最大值再取 exp，避免溢位。

### `cross_entropy(logits, target_id)` — 交叉熵損失

直接對 logits 計算交叉熵，不使用中間 softmax，避免 `log(0)` 的數值問題：

```
Loss = log(sum(exp(x_i - M))) - (x_target - M)
     = -log( e^{x_target} / sum(e^{x_i}) )
```

其中 M = max(logits)，即 Log-Sum-Exp 技巧。

### `rmsnorm(x)` — RMS 層正規化

將啟用值除以其 RMS（均方根），讓輸出維持在穩定的數值範圍：

```python
from nn0 import rmsnorm, Value

x = [Value(1.5), Value(-3.2), Value(0.7)]
normed = rmsnorm(x)  # 正規化後 RMS ≈ 1
```

## 範例說明

### 範例 1：`examples/01_basics.py` — 自動微分基礎

最簡單的入門。展示如何建立計算圖、執行反向傳播、以及讀取梯度來理解連鎖律。

學習重點：
- 計算圖如何建立
- `backward()` 如何傳播梯度
- `relu()` 和 `log()` 合成的計算圖

### 範例 2：`examples/02_linear_regression.py` — 線性回歸

從資料中學習 `y = 2x + 1` 的真實參數。使用 **MSE 損失** 和 **Adam 優化器**。

學習重點：
- 監督式學習的完整流程：forward → loss → backward → update
- 參數初始化、最佳化迭代
- 損失函數如何引導參數收斂

### 範例 3：`examples/03_softmax_classification.py` — Softmax 分類

4 維輸入 → 3 類別分類。使用 `cross_entropy` 作為損失函數。

學習重點：
- 多類別分類的架構
- Cross-entropy loss 與 softmax 的搭配
- `linear()` 作為權重層

### 範例 4：`examples/04_rmsnorm_demo.py` — RMS 正規化

展示 RMS Normalization 如何改變資料分佈，驗證正規化後 RMS ≈ 1。

學習重點：
- 正規化 (Normalization) 的作用
- 為什麼現代神經網路需要正規化層

## 如何執行

```bash
# 在專案目錄下執行
cd nn0

# 執行範例
python examples/01_basics.py
python examples/02_linear_regression.py
python examples/03_softmax_classification.py
python examples/04_rmsnorm_demo.py

# 或直接操作 nn0
python -c "from nn0 import Value; print(Value(42))"
```

不需要安裝任何第三方套件，只需要 Python 3 標準函式庫。

## 建議學習路徑

```
Value 類別 (自動微分)
    → 範例 1 (計算圖與梯度)
    → Adam 優化器 + 範例 2 (線性回歸)
    → softmax / cross_entropy + 範例 3 (分類)
    → rmsnorm + 範例 4 (正規化)
    → gd() 與完整訓練流程 (閱讀 nn0.py 最底層)
```

## 下一步

理解 nn0.py 後，你可以：

1. **閱讀 PyTorch 的 `autograd` 文件**，你會發現核心概念完全相同
2. **加入更多 layer 類型**：如 Conv2d、LayerNorm
3. **實作更複雜的網路**：MLP、RNN、Transformer
4. **研究現代實作**：看看 PyTorch 如何用 C++ 加速相同的 autograd 機制

nn0.py 的全部程式碼只有 **~160 行**，每一行都有具體的教育意義。讀懂它，你就掌握了深度學習框架的最小核心。
