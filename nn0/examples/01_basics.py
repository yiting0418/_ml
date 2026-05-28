"""
範例 1：Value 基礎運算與自動微分
展示計算圖建立、反向傳播、梯度計算。
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn0 import Value

a = Value(2.0)
b = Value(3.0)
c = Value(4.0)

# 建立計算圖: d = (a + b) * c
d = (a + b) * c
d.backward()

print(f"d = {d.data}")
print(f"∂d/∂a = {a.grad}  (預期: {c.data})")
print(f"∂d/∂b = {b.grad}  (預期: {c.data})")
print(f"∂d/∂c = {c.grad}  (預期: {a.data + b.data})")

print()

# 鏈鎖律展示
e = Value(1.0)
f = e.relu().log()
f.backward()
print(f"e = {e.data}, f = {f.data}")
print(f"∂f/∂e = {e.grad}  (預期: 1/{e.data} = 1.0)")
