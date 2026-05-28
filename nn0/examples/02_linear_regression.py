"""
範例 2：線性回歸 — 使用 Adam 優化器
目標：學習 y = 2x + 1
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn0 import Value, Adam

# 真實參數
true_w, true_b = 2.0, 1.0

# 訓練資料
xs = [Value(i) for i in range(-10, 11)]
ys = [Value(true_w * x.data + true_b) for x in xs]

# 模型參數 (隨機初始化)
w = Value(0.0)
b = Value(0.0)
opt = Adam([w, b], lr=0.1)

num_steps = 100
for step in range(1, num_steps + 1):
    preds = [w * x + b for x in xs]
    loss = sum((pred - y) ** 2 for pred, y in zip(preds, ys)) / len(xs)

    loss.backward()
    opt.step()

    if step % 20 == 0 or step == 1:
        print(f"step {step:3d}, loss = {loss.data:.6f}, w = {w.data:.4f}, b = {b.data:.4f}")

print(f"\n最終結果: w = {w.data:.4f} (真實: {true_w}), b = {b.data:.4f} (真實: {true_b})")
