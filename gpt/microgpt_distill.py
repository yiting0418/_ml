"""
microgpt_distill.py — 雙語 GPT：預訓練 + 知識蒸餾（PyTorch + GPU）

三步驟：
  1. Pretrain: 訓練較大的 Teacher 模型
  2. Distill:  用 Teacher 的軟標籤訓練較小的 Student 模型
  3. Infer:    用 Student 模型生成中英文短句

Base on Karpathy's microgpt (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
"""

import os, math, random, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# ── 裝置 / Device ──────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'使用裝置 / device: {device}')

# ═══════════════════════════════════════════════════════════
# 1. 資料集 / Dataset
# ═══════════════════════════════════════════════════════════

# 英文名字 / English names
if not os.path.exists('input_english.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(url, 'input_english.txt')
english_docs = [l.strip() for l in open('input_english.txt') if l.strip()]

# 繁體中文短句（5-10 字）/ Traditional Chinese short sentences
chinese_docs = [
    "人生如夢", "平安就是福", "知足常樂",
    "一分耕耘一分收穫", "天下沒有白吃的午餐",
    "活到老學到老", "健康就是財富", "失敗為成功之母",
    "時間就是金錢", "誠信為做人之本",
    "團結力量大", "遠親不如近鄰", "家和萬事興",
    "助人為快樂之本", "百善孝為先",
    "知識就是力量", "勤能補拙", "不忘初心方得始終",
    "行百里者半九十", "千里之行始於足下",
    "不經一番寒徹骨", "梅花香自苦寒來",
    "塞翁失馬焉知非福", "授人以魚不如授人以漁",
    "冰凍三尺非一日之寒", "十年樹木百年樹人",
    "一日之計在於晨", "一年之計在於春",
    "今天天氣真好", "一起去吃飯吧",
    "早安你好", "晚安祝好夢", "謝謝你的幫忙",
    "好久不見", "最近過得如何",
    "生日快樂", "新年快樂", "恭喜發財",
    "萬事如意", "一路順風", "早日康復",
    "加油你一定行", "保持平常心",
    "順其自然", "隨遇而安", "一切隨緣",
    "開心過每一天",
    "夕陽無限好", "只是近黃昏",
    "明月幾時有", "把酒問青天",
    "春江水暖鴨先知", "小橋流水人家",
    "古道西風瘦馬", "大漠孤煙直",
    "長河落日圓", "採菊東籬下",
    "悠然見南山", "海闊憑魚躍",
    "天高任鳥飛",
    "學而不思則罔", "思而不學則殆",
    "溫故而知新", "三人行必有我師",
    "己所不欲勿施於人", "人無遠慮必有近憂",
    "君子和而不同", "天下興亡匹夫有責",
    "先天下之憂而憂", "後天下之樂而樂",
    "人生自古誰無死", "留取丹心照汗青",
    "生於憂患死於安樂",
    "工欲善其事", "必先利其器",
    "欲速則不達", "見賢思齊焉",
    "見不賢而內自省", "知之為知之",
    "不知為不知", "實事求是",
    "精益求精", "推己及人",
    "將心比心", "設身處地", "反求諸己",
    "簡單就是美", "享受當下", "活在當下",
    "把握每一刻", "珍惜身邊的人",
    "愛要及時", "勇敢追夢", "堅持到底",
    "永不放棄", "做自己就好", "開心最重要",
    "平安喜樂", "自由自在", "心想事成",
    "美夢成真", "前途無量", "前程似錦",
    "風和日麗好出遊", "月有陰晴圓缺",
    "人有悲歡離合", "但願人長久",
    "千里共嬋娟", "不識廬山真面目",
    "只緣身在此山中", "春色滿園關不住",
    "一枝紅杏出牆來", "山重水複疑無路",
    "柳暗花明又一村",
    "人生得意須盡歡", "莫使金樽空對月",
    "天生我材必有用", "千金散盡還復來",
    "抽刀斷水水更流", "舉杯消愁愁更愁",
    "少年不識愁滋味", "衣帶漸寬終不悔",
    "為伊消得人憔悴", "眾裡尋他千百度",
    "此情可待成追憶", "只是當時已惘然",
    "歡喜就好", "平安順遂",
    "身體健康", "事業順利",
    "家庭美滿", "學業進步", "步步高陞",
    "財源廣進", "大吉大利",
    "新年快樂萬事如意", "恭喜發財紅包拿來",
    "中秋節快樂", "端午節安康",
    "母親節快樂", "父親節快樂",
    "情人節快樂", "聖誕節快樂",
    "開工大吉", "喜氣洋洋",
    "福如東海", "壽比南山",
    "白頭偕老", "永結同心",
    "花好月圓", "百年好合",
    "忍一時風平浪靜", "退一步海闊天空",
    "吃虧就是佔便宜", "施比受更有福",
    "船到橋頭自然直", "浪子回頭金不換",
    "一寸光陰一寸金", "寸金難買寸光陰",
    "路遙知馬力日久見人心",
    "近朱者赤近墨者黑",
    "良藥苦口利於病", "忠言逆耳利於行",
    "獨樂樂不如眾樂樂", "前人種樹後人乘涼",
    "天下無難事只怕有心人",
    "有緣千里來相會", "無緣對面不相識",
    "只羨鴛鴦不羨仙",
    "人生如戲全靠演技", "笑一笑十年少",
    "睡個好覺明天會更好",
    "沒有過不去的事情", "只有過不去的心情",
    "關關難過關關過", "事事難成事事成",
    "天天開心", "事事如意", "年年有餘",
    "步步高升", "蒸蒸日上", "心想事成",
    "五福臨門", "六六大順", "七星高照",
    "八方來財", "九九同心", "十全十美",
    "百事可樂", "千祥雲集", "萬事亨通",
    "一帆風順", "四季平安",
    "五穀豐登", "六畜興旺",
    "十分美滿", "百尺竿頭",
    "萬無一失", "一心一意",
    "四季如春", "五彩繽紛",
    "八仙過海", "九牛二虎",
    "十面埋伏", "百發百中",
    "萬水千山", "山明水秀", "鳥語花香",
    "風調雨順", "國泰民安", "安居樂業",
    "豐衣足食", "欣欣向榮",
    "日新月異", "自強不息", "厚德載物",
    "上善若水", "有容乃大", "無欲則剛",
    "海納百川", "淡泊明志",
    "寧靜致遠", "博學篤行",
    "天天好心情", "明天會更好",
    "一切都是最好的安排",
    "放下才能自在",
    "簡單就是幸福",
    "知足者富",
    "心寬路更寬",
    "退一步想",
    "進一步看",
    "少即是多",
    "慢就是快",
    "靜能生慧",
    "慧能生定",
    "定能生智",
]

random.shuffle(chinese_docs)
docs = english_docs + chinese_docs
random.shuffle(docs)
print(f'文件 / docs: {len(docs)}（英文 {len(english_docs)} + 中文 {len(chinese_docs)}）')

# ═══════════════════════════════════════════════════════════
# 2. 分詞器 / Tokenizer
# ═══════════════════════════════════════════════════════════
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
stoi = {ch: i for i, ch in enumerate(uchars)}
itos = {i: ch for i, ch in enumerate(uchars)}
itos[BOS] = '<BOS>'
print(f'詞彙量 / vocab: {vocab_size}')

def encode(s):
    return [BOS] + [stoi[ch] for ch in s] + [BOS]

def decode(toks):
    return ''.join(itos[t] for t in toks if t != BOS)

# ═══════════════════════════════════════════════════════════
# 3. GPT 模型（PyTorch）/ GPT Model (PyTorch)
# ═══════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.ln_in = nn.RMSNorm(n_embd)
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_out = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=x.device)).unsqueeze(0)
        x = self.ln_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, token_id, max_len, temperature=0.5):
        self.eval()
        device = next(self.parameters()).device
        toks = [token_id]
        out = []
        for _ in range(max_len):
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            logits = self.forward(inp)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if next_id == BOS:
                break
            out.append(next_id)
            toks.append(next_id)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════
# 4. 訓練 / Training
# ═══════════════════════════════════════════════════════════
def train_one_epoch(model, docs, batch_size, opt):
    model.train()
    indices = list(range(len(docs)))
    random.shuffle(indices)
    total_loss = 0.0
    n = 0
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        seqs = []
        for idx in batch_idx:
            toks = encode(docs[idx])
            n_tok = min(model.block_size, len(toks))
            seqs.append(torch.tensor(toks[:n_tok], dtype=torch.long))
        max_len = max(s.shape[0] for s in seqs)
        x = torch.full((len(seqs), max_len), 0, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :s.shape[0]] = s
        x = x.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               x[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ═══════════════════════════════════════════════════════════
# 5. 蒸餾 / Distillation
# ═══════════════════════════════════════════════════════════
def distill_one_epoch(teacher, student, docs, batch_size, opt, temp=2.0, alpha=0.7):
    teacher.eval()
    student.train()
    indices = list(range(len(docs)))
    random.shuffle(indices)
    total_loss = 0.0
    n = 0
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        seqs = []
        for idx in batch_idx:
            toks = encode(docs[idx])
            n_tok = min(student.block_size, len(toks))
            seqs.append(torch.tensor(toks[:n_tok], dtype=torch.long))
        max_len = max(s.shape[0] for s in seqs)
        x = torch.full((len(seqs), max_len), 0, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :s.shape[0]] = s
        x = x.to(device)

        with torch.no_grad():
            t_logits = teacher(x)
            t_soft = F.softmax(t_logits / temp, dim=-1)

        s_logits = student(x)
        s_soft = F.log_softmax(s_logits / temp, dim=-1)

        mask = (x != 0).unsqueeze(-1).expand_as(s_soft)
        s_masked = s_soft[mask].view(-1, s_soft.size(-1))
        t_masked = t_soft[mask].view(-1, t_soft.size(-1))

        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.size(-1)),
                             x[:, 1:].reshape(-1), ignore_index=0)
        kl = F.kl_div(s_masked, t_masked, reduction='batchmean')
        loss = alpha * ce + (1 - alpha) * temp * temp * kl

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ═══════════════════════════════════════════════════════════
# 6. 推理 / Inference
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def generate_samples(model, n_samples=30, temperature=0.5):
    model = model.to(device).eval()
    print()
    for idx in range(n_samples):
        tokens = model.generate(BOS, max_len=model.block_size, temperature=temperature)
        text = decode(tokens)
        has_cn = any('\u4e00' <= c <= '\u9fff' for c in text)
        label = '中文' if has_cn else 'English'
        print(f'  sample {idx+1:2d} [{label}]: {text}')


# ═══════════════════════════════════════════════════════════
# 7. 主流程 / Main Pipeline
# ═══════════════════════════════════════════════════════════
block_size = 32
batch_size = 64
lr = 1e-3

print('\n' + '='*50)
print('階段 1：預訓練 Teacher')
print('='*50)
teacher = GPT(vocab_size, n_embd=64, n_layer=2, n_head=4, block_size=block_size).to(device)
print(f'  Teacher 參數: {count_params(teacher):,}')
opt_t = torch.optim.AdamW(teacher.parameters(), lr=lr)
for epoch in range(15):
    loss = train_one_epoch(teacher, docs, batch_size, opt_t)
    print(f'  [teacher] epoch {epoch+1:2d}/15 | loss {loss:.4f}')

print('\n' + '='*50)
print('階段 2：知識蒸餾 Student')
print('='*50)
student = GPT(vocab_size, n_embd=16, n_layer=1, n_head=4, block_size=block_size).to(device)
print(f'  Student 參數: {count_params(student):,}')
opt_s = torch.optim.AdamW(student.parameters(), lr=lr)
for epoch in range(15):
    loss = distill_one_epoch(teacher, student, docs, batch_size, opt_s, temp=2.0, alpha=0.7)
    print(f'  [distill] epoch {epoch+1:2d}/15 | loss {loss:.4f}')

print('\n' + '='*50)
print('Student 生成結果')
print('='*50)
generate_samples(student, n_samples=30, temperature=0.5)

print('\n' + '='*50)
print('Teacher 生成結果（對照）')
print('='*50)
generate_samples(teacher, n_samples=10, temperature=0.5)
