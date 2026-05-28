# nn0.py — 從零開始學自動微分

這份程式碼用不到 200 行的純 Python，實作了一個**自動微分引擎**（autograd）。沒有 TensorFlow、沒有 PyTorch，只有最基本的 Python 與 math 模組。

目標不是寫出高效能的程式，而是讓你清楚看見**反向傳播（backpropagation）的核心機制**。

---

## 目錄

1. [為什麼需要自動微分？](#1-為什麼需要自動微分)
2. [Value 類別：計算圖的節點](#2-value-類別計算圖的節點)
3. [反向傳播的兩步驟](#3-反向傳播的兩步驟)
4. [實際運作範例](#4-實際運作範例)
5. [Adam 優化器](#5-adam-優化器)
6. [完整 API 一覽](#6-完整-api-一覽)
7. [延伸閱讀](#7-延伸閱讀)

---

## 1. 為什麼需要自動微分？

訓練神經網路的核心迴圈很單純：

```
1. 給定輸入 → 計算輸出（forward）
2. 比較輸出與正確答案 → 得到 loss
3. 計算 loss 對每個參數的「梯度」（gradient）
4. 沿著梯度反方向調整參數（gradient descent）
```

其中第 3 步是最麻煩的。當網路很深時，手算鏈鎖律（chain rule）既繁瑣又容易出錯。

**自動微分**的做法是：讓每個數值都記得「自己是怎麼算出來的」，求梯度的時候順著這個計算軌跡反向走一遍即可。

這就是 `Value` 類別做的事。

---

## 2. Value 類別：計算圖的節點

`Value` 包裝一個普通的數字，但它多了兩樣東西：

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
```

| 屬性 | 用途 |
|------|------|
| `data` | 實際的數值 |
| `grad` | 累積的梯度（初始為 0） |
| `_children` | 這個值是「從哪些節點算出來的」 |
| `_local_grads` | 對每個 child 的「局部梯度」 |

每次你做加、乘、ReLU 等運算，`Value` 不只算出結果的數值，還會**記錄**：

- 這個結果是由哪些輸入（`_children`）產生的
- 結果對每個輸入的偏微分是多少（`_local_grads`）

例如 `c = a + b`：

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data,    # 前向計算
                 (self, other),              # 記錄 children
                 (1, 1))                     # ∂c/∂a = 1, ∂c/∂b = 1
```

又如 `c = a * b`：

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data,    # 前向計算
                 (self, other),              # 記錄 children
                 (other.data, self.data))    # ∂c/∂a = b, ∂c/∂b = a
```

這些節點與節點之間的連線，形成一個**有向無環圖（DAG）**——這就是計算圖。

---

## 3. 反向傳播的兩步驟

呼叫 `backward()` 時，做了兩件事：

**步驟一：拓撲排序（topological sort）**

把計算圖中所有節點排成一個序列，確保每個節點都在它的依賴項之後出現。這樣反向走的時候，梯度才能正確傳遞。

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)
        topo.append(v)        # children 先加入，parent 後加入
```

**步驟二：沿反向傳播梯度**

```python
self.grad = 1                        # d loss / d loss = 1
for v in reversed(topo):             # 從 loss 開始往回走
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad   # 鏈鎖律
```

這裡的核心就是**鏈鎖律**：如果 `y = f(g(x))`，那麼 `dy/dx = (df/dg) * (dg/dx)`。程式碼中的 `local_grad` 就是 `df/dg`，`v.grad` 就是上層傳下來的 `dg/dx`。

---

## 4. 實際運作範例

```python
from nn0 import Value

a = Value(2.0)                     # a = 2
b = Value(3.0)                     # b = 3
c = a * b                          # c = 6, ∂c/∂a = 3, ∂c/∂b = 2
d = c + a                          # d = 8, ∂d/∂c = 1, ∂d/∂a = 1

d.backward()                       # 反向傳播

print(a.grad)  # 梯度是怎麼累積的？
```

當你呼叫 `d.backward()`，程式會先拓撲排序，然後反向走：

```
∂d/∂d = 1
∂d/∂c = 1  →  c.grad += 1 * 1 = 1
  ∂c/∂a = 3 → a.grad += 3 * 1 = 3   (經由 c)
  ∂c/∂b = 2 → b.grad += 2 * 1 = 2
∂d/∂a = 1  →  a.grad += 1 * 1 = 1   (經由 d 直接)
```

最終 `a.grad = 3 + 1 = 4`。

---

## 5. Adam 優化器

有了梯度之後，下一步就是更新參數。這裡實作了 Adam 優化器。

Adam 維護每個參數的一階動量（`m`）與二階動量（`v`），更新規則為：

```python
m = β₁·m + (1 − β₁)·grad
v = β₂·v + (1 − β₂)·grad²
m_hat = m / (1 − β₁^t)
v_hat = v / (1 − β₂^t)
param -= lr · m_hat / (√v_hat + ε)
```

每次呼叫 `step()` 後會自動將所有參數的 `grad` 歸零。

```python
from nn0 import Value, Adam

w = Value(0.0)
b = Value(0.0)
opt = Adam([w, b], lr=0.1)

# ... 計算 loss ...
loss.backward()
opt.step()   # 更新 w, b，並清除 gradient
```

---

## 6. 完整 API 一覽

### Value — 自動微分節點

| 運算 | 語法 | 梯度 |
|------|------|------|
| 加法 | `a + b` | ∂/∂a = 1, ∂/∂b = 1 |
| 乘法 | `a * b` | ∂/∂a = b, ∂/∂b = a |
| 冪次 | `a ** k` | ∂/∂a = k·a^(k−1) |
| ReLU | `a.relu()` | ∂/∂a = 1 若 a > 0，否則 0 |
| 指數 | `a.exp()` | ∂/∂a = e^a |
| 對數 | `a.log()` | ∂/∂a = 1/a |
| 負號 | `-a` | ∂/∂a = −1 |
| 減法 | `a - b` | ∂/∂a = 1, ∂/∂b = −1 |
| 除法 | `a / b` | ∂/∂a = 1/b, ∂/∂b = −a/b² |

### 高層函數

| 函數 | 說明 |
|------|------|
| `linear(x, w)` | 矩陣乘法 `y = W @ x`，`w` 是二維 list |
| `softmax(logits)` | 數值穩定的 softmax |
| `rmsnorm(x)` | RMS 正規化 |
| `cross_entropy(logits, target_id)` | 使用 Log-Sum-Exp 技巧的 cross-entropy loss |
| `gd(model, optimizer, tokens, step, num_steps)` | 單步梯度下降（給 transformer 訓練用） |

### 建議閱讀順序

1. 先跑 `examples/01_basics.py` — 理解計算圖與梯度傳遞
2. 再跑 `examples/02_linear_regression.py` — 看 Adam 如何學習
3. 接著 `examples/03_softmax_classification.py` — softmax + cross-entropy
4. 最後 `examples/04_rmsnorm_demo.py` — 正規化層的效用

---

## 7. 延伸閱讀

如果你理解了這份程式碼，推薦繼續研究：

- **Andrej Karpathy — micrograd**：本程式的靈感來源，也是 nn0.py 的進階版本
- **CS231n 筆記 — 反向傳播**：對鏈鎖律與計算圖的視覺化解釋
- **Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)**：Adam 原始論文

---

## 授權

MIT
