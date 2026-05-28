"""
microgpt_bilingual_gpu.py — 雙語 GPT（繁體中文 + English），PyTorch GPU 版

完全從零實作 GPT-2 架構，基於 PyTorch，支援 MPS (Apple Silicon) 與 CUDA。
Base on Karpathy's microgpt (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
"""

import os, math, random, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 裝置 / Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"裝置 / device: {device}")

random.seed(42)
torch.manual_seed(42)

# ============================================================
# 繁體中文短句資料集（8-15 字）
# ============================================================
chinese_sentences = [
    # 打招呼 / Greetings
    "你好嗎今天過得如何", "早安祝你有个好心情",
    "晚安祝你好夢", "好久不見最近好嗎",
    "謝謝你的幫忙", "不客氣這是我應該做的",
    "對不起我不是故意的", "沒關係大家互相體諒",
    "請坐慢慢來", "再見一路順風",
    "明天見好好休息", "下次見要保持聯絡",

    # 生活日常 / Daily life
    "今天天氣真好適合出門走走", "早睡早起身體好",
    "記得要多喝水對身體好", "慢慢走小心安全",
    "好好休息明天才有精神", "開開心心過每一天",
    "平常心面對一切", "健康就是最大的財富",
    "喝杯咖啡放鬆一下", "出去散步曬曬太陽",
    "早點睡覺不要熬夜", "按時吃飯對胃比較好",

    # 簡單感受 / Simple feelings
    "我今天很開心", "工作有點累但很充實",
    "沒問題交給我來處理", "你做得太好了真厲害",
    "這家餐廳的東西好好吃", "這部電影真的好好看",
    "今天天氣好冷要多穿衣服", "吃飽了覺得好滿足",
    "人生就是不斷學習", "每天都讓自己進步一點",

    # 學習工作 / Study & Work
    "認真學習才能進步", "努力工作實現夢想",
    "慢慢來不要著急", "你做得非常好繼續加油",
    "堅持下去就會成功", "我相信你一定做得到",
    "不懂就要問不要怕", "多多練習就會越來越好",
    "好好讀書天天向上", "努力賺錢孝順父母",

    # 祝福 / Wishes
    "祝你生日快樂健康平安", "新年快樂恭喜發財",
    "祝你萬事如意心想事成", "平安喜樂天天開心",
    "祝你身體健康長命百歲", "一路順風事事順利",
    "祝你美夢成真", "大吉大利好運連連",
    "百年好合永結同心", "白頭偕老幸福美滿",

    # 自然 / Nature
    "今天的天空好藍", "花都開了好漂亮",
    "夕陽好美好浪漫", "今晚的月亮好圓好亮",
    "春天的風吹得好舒服", "秋天的落葉好美",
    "下雨了記得帶傘", "出太陽了去外面走走",
    "大自然就是最好的醫生", "好山好水好放鬆",

    # 人生哲理 / Life wisdom
    "人生就像一場旅行", "過程比結果更重要",
    "珍惜當下把握每一刻", "知足常樂平安是福",
    "簡單生活也是一種幸福", "心存善念好事會發生",
    "對自己好一點", "照顧好自己才能照顧別人",
    "學會放下才能前進", "每一天都是新的開始",

    # 家庭朋友 / Family & Friends
    "家人是最重要的", "朋友是一輩子的財富",
    "常常關心身邊的人", "有你們真好我很幸福",
    "孝順父母及時行孝", "陪伴就是最好的禮物",
    "我愛你爸爸媽媽", "有你真好謝謝你",

    # 食物 / Food
    "這碗湯好好喝", "今天一起來煮飯",
    "想吃什麼我請客", "自己煮的最好吃",
    "飯前洗手飯後漱口", "多吃水果身體健康",

    # 時間 / Time
    "時間過得真快", "不知不覺又一年",
    "把握好每一個今天", "不要後悔過去的選擇",
    "展望未來充滿希望", "活在當下就是最好的",
]

# 下載英文名字 / Download English names
if not os.path.exists('input_english.txt'):
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input_english.txt')
english_docs = [l.strip() for l in open('input_english.txt') if l.strip()]

random.shuffle(english_docs)
english_docs = english_docs[:len(chinese_sentences)]

docs = english_docs + chinese_sentences
random.shuffle(docs)

# ============================================================
# 分詞器 / Tokenizer
# ============================================================
uchars = sorted(set(''.join(docs)))
stoi = {ch: i for i, ch in enumerate(uchars)}
itos = {i: ch for ch, i in stoi.items()}
BOS = len(uchars)
vocab_size = len(uchars) + 1

def encode(s):
    return [BOS] + [stoi[c] for c in s] + [BOS]

print(f"詞彙量 / vocab size: {vocab_size}")
print(f"資料數 / docs: {len(docs)} ({len(chinese_sentences)} 中文 + {len(english_docs)} 英文)")

# ============================================================
# 資料載入 / Data Loading (batch)
# ============================================================
block_size = 20
batch_size = 128

def get_batch(split='train'):
    batch_docs = docs
    xs, ys = [], []
    for _ in range(batch_size):
        doc = random.choice(batch_docs)
        tokens = encode(doc)
        n = min(block_size, len(tokens) - 1)
        x = torch.tensor(tokens[:n], dtype=torch.long)
        y = torch.tensor(tokens[1:n+1], dtype=torch.long)
        xs.append(x)
        ys.append(y)
    xb = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0).to(device)
    yb = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1).to(device)
    return xb, yb

# ============================================================
# GPT-2 模型（PyTorch）
# ============================================================
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, block_size=16, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.ln_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=-1)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == BOS:
                break
        return idx

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        head_dim = n_embd // n_head
        self.ln1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, head_dim, dropout)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(y)
        return self.wo(y)

# ============================================================
# 初始化模型 / Initialize Model
# ============================================================
model = GPT(vocab_size, block_size=block_size).to(device)
print(f"參數總數 / params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)

# ============================================================
# 訓練迴圈 / Training Loop
# ============================================================
num_steps = 2000
print(f"\n開始訓練 / Training ({num_steps} steps)...\n")

for step in range(num_steps):
    model.train()
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if (step + 1) % 200 == 0 or step == 0:
        lr_now = scheduler.get_last_lr()[0]
        print(f"step {step+1:4d}/{num_steps} | loss {loss.item():.4f} | lr {lr_now:.2e}")

# ============================================================
# 推理 / Inference
# ============================================================
model.eval()
temperature = 0.6
print("\n--- 生成結果 / Generated samples ---\n")

with torch.no_grad():
    for i in range(30):
        start = torch.tensor([[BOS]], dtype=torch.long, device=device)
        out = model.generate(start, max_new_tokens=block_size, temperature=temperature)
        tokens = out[0].cpu().tolist()
        if BOS in tokens[1:]:
            tokens = tokens[1:tokens.index(BOS, 1)]
        else:
            tokens = tokens[1:]
        text = ''.join(itos[t] for t in tokens if t in itos)
        has_cn = any('\u4e00' <= c <= '\u9fff' for c in text)
        prefix = "中文" if has_cn else "EN"
        print(f"[{prefix:4s}] {text}")
