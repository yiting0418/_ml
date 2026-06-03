# 爬山演算法解決旅行推銷員問題 (TSP)

## 問題概述

旅行推銷員問題（Traveling Salesman Problem, TSP）是一個經典的組合最佳化問題：給定 n 個城市，推銷員從起點出發，每個城市恰好拜訪一次後回到起點，求最短的路徑。

本實作使用 **爬山演算法 (Hill Climbing)** 搭配 **2-opt 鄰居交換** 來求解。

## 演算法說明

### 爬山演算法

```
Start → 產生初始解 s
        ↓
    重複直到收斂或達到最大代數：
        snew = s.neighbor()    ← 取得鄰近解
        if snew.height() >= s.height():
            s = snew           ← 往更好的方向移動
        else:
            fails += 1         ← 失敗次數 +1
```

爬山演算法是一種局部搜尋（Local Search）方法，每次從當前解的鄰域中找一個更好的解移動，直到無法再改進為止。

### 2-opt 交換（neighbor）

選兩條邊 (a,b) 和 (c,d)，交換成 (a,d) 和 (b,c)，相當於反轉兩點之間的路徑段：

```
原始：...→a→b→...→c→d→...
交換：...→a→d→...→c→b→...
```

實作方式：隨機選兩個索引 i < j，將 `path[i+1:j+1]` 反轉。

### 高度函數（height）

`height = -total_distance`，距離越短則高度越高，符合爬山演算法最大化高度值的需求。

## 程式架構

```
ml/1/
├── tsp_hillClimbing.py     # 主程式
└── README.md               # 本文件
```

### TspSolution 類別

| 方法 | 說明 |
|------|------|
| `__init__(cities, path)` | 初始解為 1→2→...→n→1 |
| `neighbor()` | 2-opt 交換產生鄰近解 |
| `height()` | 回傳 -total_distance |
| `total_distance()` | 計算完整迴路距離 |
| `str()` | 格式化輸出路徑與距離 |

### hillClimbing 函數

參數：
- `s`：初始解
- `maxGens`：最大疊代代數（預設 1000）
- `maxFails`：連續失敗次數上限（預設 100）

## 執行結果

使用 20 個隨機城市（座標範圍 0~100）：

```
Initial distance: 1087.09

start:  path: 1=>2=>3=>...=>20=>1, distance=1087.09
...
solution:  path: 1=>19=>5=>12=>7=>16=>18=>9=>10=>4=>3=>11=>2=>17=>8=>6=>14=>15=>20=>13=>1, distance=493.74

Final distance: 493.74
```

距離從 **1087.09** 降至 **493.74**，改善約 **54.6%**。

## 與 opencode 對話過程

以下為使用 opencode CLI 工具完成此任務的完整流程：

### Step 1: 理解需求

使用者提出需求：「用爬山演算法解決旅行推銷員問題」，並提供參考連結：
- [爬山演算法物件導向框架](https://github.com/ccc114b/py2cs/blob/master/02-機器學習/01-優化算法/01-傳統優化/01-優化/01-爬山演算法/04-爬山物件導向框架/hillClimbing.py)
- [維基百科 TSP 條目](https://zh.wikipedia.org/zh-tw/旅行推銷員問題)

提示要求：
- `neighbor`：用 2-opt 交換（選兩邊 (a,b)(c,d) → (a,d),(b,c)）
- `height`：用旅行距離 × -1
- 初始解：1→2→3→...→n→1
- 檔案寫在 `ml/1` 資料夾

### Step 2: 爬取參考程式碼

```bash
# opencode 使用 WebFetch 工具取得 hillClimbing.py 原始碼
# 確認爬蟲演算法主體：hillClimbing(s, maxGens, maxFails)
# 介面要求：s.neighbor(), s.height(), s.str()
```

### Step 3: 撰寫程式

opencode 根據參考框架實作 `TspSolution` 類別與 `hillClimbing` 函數，寫入 `ml/1/tsp_hillClimbing.py`。

### Step 4: 執行程式驗證

```bash
python3 ml/1/tsp_hillClimbing.py
```

輸出正常，距離從 1087.09 降至 493.74，演算法正確收斂。

### Step 5: 撰寫 README

使用者要求撰寫本說明文件，記錄程式碼架構與與 opencode 的協作過程。
