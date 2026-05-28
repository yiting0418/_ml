"""
範例 3：Softmax 分類 — 使用 cross_entropy
模擬 4 維輸入 → 3 類別分類
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn0 import Value, linear, softmax, cross_entropy, Adam

# 資料：4 個輸入特徵，3 個類別
x = [Value(0.5), Value(1.2), Value(-0.8), Value(0.3)]
target = 2  # 正確類別索引

# 權重：3x4 矩陣
w = [[Value(0.01 * (i * 4 + j)) for j in range(4)] for i in range(3)]

# 收集所有參數
params = [wi for row in w for wi in row]
opt = Adam(params, lr=0.1)

num_steps = 50
for step in range(1, num_steps + 1):
    logits = linear(x, w)
    loss = cross_entropy(logits, target)

    loss.backward()
    opt.step()

    if step % 10 == 0 or step == 1:
        probs = softmax(logits)
        probs_str = " ".join(f"{p.data:.4f}" for p in probs)
        print(f"step {step:3d}, loss = {loss.data:.6f}, probs = [{probs_str}]")

print(f"\n最終預測: 類別 {max(range(3), key=lambda i: softmax(linear(x, w))[i].data)}")
