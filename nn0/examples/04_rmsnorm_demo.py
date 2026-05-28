"""
範例 4：RMS Normalization 示範
比較正規化前後的資料分佈。
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn0 import Value, rmsnorm

# 模擬一層的啟用值
activations = [Value(1.5), Value(-3.2), Value(0.7), Value(2.1), Value(-0.4)]

print("原始啟用值:")
print("  " + " ".join(f"{a.data:7.4f}" for a in activations))

normed = rmsnorm(activations)
print("RMS 正規化後:")
print("  " + " ".join(f"{a.data:7.4f}" for a in normed))

# 驗證 RMS ≈ 1
rms = (sum(a.data ** 2 for a in normed) / len(normed)) ** 0.5
print(f"\n正規化後的 RMS: {rms:.6f} (應接近 1)")
