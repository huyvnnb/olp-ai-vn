# 00 · Nền tảng: Transformer

> Đọc file này trước tiên.
> Mọi bài NLP = **Transformer** + **một lớp đầu ra (head)** + **một hàm loss**.

Quy ước:
- Khối `python` = code tự viết, chạy được (chạy lần lượt từ trên xuống).
- Khối `py` = code dùng thư viện có sẵn, để tham khảo.

---

## 1. Bức tranh chung

```
văn bản  →  tokenizer  →  các số (ids)  →  Transformer  →  vector cho từng token
                                                               │
                         ┌──────────────┬──────────────┬───────┴────────┐
                         ▼              ▼              ▼                ▼
                    vector cả câu   từng token     start / end      sinh từng từ
                    Phân loại       NER            Hỏi đáp (QA)     Dịch / Tóm tắt
```

Có 3 họ model:

| Họ | Ví dụ | Giỏi việc gì |
|---|---|---|
| Encoder | BERT, PhoBERT, XLM-R | hiểu văn bản: phân loại, NER, QA |
| Decoder | GPT, Llama, Qwen | sinh văn bản |
| Encoder + Decoder | T5, BART, Marian, NLLB | dịch máy, tóm tắt |

---

## 2. Tokenizer: cắt chữ thành số

Máy chỉ hiểu số. Tokenizer cắt câu thành mảnh nhỏ (token), mỗi mảnh có 1 số.

- Cắt theo **từ**: từ điển quá lớn, gặp từ lạ là bó tay.
- Cắt theo **ký tự**: chuỗi quá dài.
- Cắt theo **subword** (cách hiện nay): từ phổ biến giữ nguyên, từ hiếm tách nhỏ.
  Ví dụ: `lowest` → `low` + `est`.

**BPE** là cách học subword phổ biến nhất:
1. Bắt đầu: mỗi ký tự là 1 token.
2. Đếm cặp token đứng cạnh nhau nhiều nhất.
3. Gộp cặp đó thành 1 token mới.
4. Lặp lại bước 2–3.

```python
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)
```

Hàm gộp một cặp token trong một từ:

```python
def merge_pair(symbols, pair):
    result = []
    i = 0
    while i < len(symbols):
        is_pair = i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair
        if is_pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result
```

Học BPE:

```python
def train_bpe(sentences, num_merges):
    # mỗi từ -> danh sách ký tự, thêm "</w>" đánh dấu cuối từ
    word_freq = Counter()
    for sentence in sentences:
        for word in sentence.split():
            word_freq[tuple(word) + ("</w>",)] += 1

    merges = []
    for _ in range(num_merges):
        # đếm các cặp đứng cạnh nhau
        pair_freq = Counter()
        for word, freq in word_freq.items():
            for a, b in zip(word, word[1:]):
                pair_freq[(a, b)] += freq
        if len(pair_freq) == 0:
            break

        # gộp cặp nhiều nhất
        best_pair = pair_freq.most_common(1)[0][0]
        merges.append(best_pair)

        new_word_freq = Counter()
        for word, freq in word_freq.items():
            new_word = tuple(merge_pair(list(word), best_pair))
            new_word_freq[new_word] += freq
        word_freq = new_word_freq
    return merges


def bpe_encode(word, merges):
    symbols = list(word) + ["</w>"]
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols


sentences = ["low lower lowest low", "new newer newest new", "học sinh học bài học"]
merges = train_bpe(sentences, num_merges=20)

print(bpe_encode("lowest", merges))
print(bpe_encode("newer", merges))
```

**Lưu ý tiếng Việt:**
- PhoBERT học trên văn bản **đã tách từ** (`học_sinh`) → phải tách từ bằng `pyvi` trước khi đưa vào.
- XLM-R không cần tách từ.

Thư viện có sẵn:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
print(tokenizer("Tôi đi học"))      # input_ids, attention_mask
```

---

## 3. Vị trí của token

Attention không biết thứ tự từ. Ta cộng thêm một vector "vị trí" vào mỗi token.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)                 # (max_len, 1)
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * freq)                      # cột chẵn: sin
        pe[:, 1::2] = torch.cos(position * freq)                      # cột lẻ: cos
        self.register_buffer("pe", pe)

    def forward(self, x):                                             # x: (batch, len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len]
```

---

## 4. Attention

Ý tưởng: mỗi token **nhìn** các token khác và lấy thông tin từ token liên quan nhất.

Mỗi token tạo ra 3 vector:
- **Q (query)**: tôi đang tìm gì?
- **K (key)**: tôi có gì?
- **V (value)**: nội dung tôi đưa ra.

Công thức:

```
điểm      = Q · Kᵀ / √d          (chia √d để số không quá lớn)
trọng số  = softmax(điểm)
kết quả   = trọng số · V
```

```python
def attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    if mask is not None:
        # mask = True: được nhìn; False: bị che
        scores = scores.masked_fill(mask == False, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output
```

**Mask** dùng để che bớt:

```python
def make_padding_mask(ids, pad_id):
    # che các token đệm (PAD). Kích thước (batch, 1, 1, len)
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length):
    # decoder không được nhìn tương lai -> ma trận tam giác dưới
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


print(make_causal_mask(4).int())
```

**Multi-head**: chạy nhiều attention song song, mỗi "đầu" học một kiểu quan hệ khác nhau.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (batch, len, d_model) -> (batch, heads, len, d_head)
        batch, length, _ = x.shape
        x = x.view(batch, length, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(self, x_query, x_key_value, mask=None):
        q = self.split_heads(self.w_q(x_query))
        k = self.split_heads(self.w_k(x_key_value))
        v = self.split_heads(self.w_v(x_key_value))

        out = attention(q, k, v, mask)                    # (batch, heads, len, d_head)

        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.w_out(out)
```

- **Self-attention**: query, key, value cùng lấy từ 1 câu.
- **Cross-attention**: query lấy từ câu đích, key/value lấy từ câu nguồn (dùng trong dịch máy).

---

## 5. Một lớp Transformer

```
x ──► LayerNorm ──► Attention ──► cộng x ──► LayerNorm ──► FeedForward ──► cộng ──► ra
```

Phép "cộng x" (residual) giúp train mạng sâu không bị mất tín hiệu.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, h, mask))

        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x
```

Ghép nhiều lớp lại thành **Encoder** (giống BERT, bản thu nhỏ):

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=256, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, d_ff))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids):
        mask = make_padding_mask(ids, self.pad_id)
        x = self.embedding(ids) * math.sqrt(self.d_model)
        x = self.position(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)                                  # (batch, len, d_model)


encoder = Encoder(vocab_size=100)
ids = torch.randint(1, 100, (2, 7))                          # 2 câu, mỗi câu 7 token
print(encoder(ids).shape)                                    # torch.Size([2, 7, 128])
```

---

## 6. Pretrain: Fill-Mask và Text Generation

Model lớn được "dạy đọc" trước trên rất nhiều văn bản, theo 1 trong 2 cách:

| Cách | Model | Bài tập | Task tương ứng |
|---|---|---|---|
| MLM | BERT | che 15% từ, đoán lại | **Fill-Mask** |
| CLM | GPT | đoán từ tiếp theo | **Text Generation** |

### MLM: che từ

Trong 15% token được chọn:
- 80% → thay bằng `[MASK]`
- 10% → thay bằng từ ngẫu nhiên
- 10% → giữ nguyên

Chỉ tính loss ở các vị trí được chọn (các vị trí khác đặt nhãn `-100` để bỏ qua).

```python
def mask_tokens(ids, mask_id, vocab_size, prob=0.15):
    labels = ids.clone()
    ids = ids.clone()

    chosen = torch.rand(ids.shape) < prob
    labels[~chosen] = -100                       # không tính loss ở chỗ không chọn

    r = torch.rand(ids.shape)
    to_mask = chosen & (r < 0.8)
    to_random = chosen & (r >= 0.8) & (r < 0.9)

    ids[to_mask] = mask_id
    ids[to_random] = torch.randint(vocab_size, ids.shape)[to_random]
    return ids, labels


ids = torch.randint(5, 50, (1, 12))
masked_ids, labels = mask_tokens(ids, mask_id=4, vocab_size=50)
print("gốc   :", ids[0].tolist())
print("bị che:", masked_ids[0].tolist())
print("nhãn  :", labels[0].tolist())
```

### CLM: mini-GPT đoán ký tự tiếp theo

GPT = encoder layer + **causal mask** (không nhìn tương lai).

```python
text = "học, học nữa, học mãi. " * 40
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
data = torch.tensor([char_to_id[c] for c in text])


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, context=32):
        super().__init__()
        self.context = context
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, 4 * d_model, dropout=0.0))
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)        # dự đoán token tiếp theo

    def forward(self, ids):
        length = ids.size(1)
        x = self.token_emb(ids) + self.pos_emb(torch.arange(length))
        mask = make_causal_mask(length)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))                    # (batch, len, vocab)
```

Train: input là đoạn văn, nhãn là đoạn văn **dịch sang phải 1 ký tự**.

```python
gpt = MiniGPT(len(chars))
optimizer = torch.optim.AdamW(gpt.parameters(), lr=3e-3)

for step in range(300):
    starts = torch.randint(0, len(data) - gpt.context - 1, (16,))
    x = torch.stack([data[s : s + gpt.context] for s in starts])
    y = torch.stack([data[s + 1 : s + gpt.context + 1] for s in starts])

    logits = gpt(x)
    loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("loss cuối:", round(loss.item(), 3))
```

### Cách sinh văn bản

| Cách | Ý tưởng | Dùng khi |
|---|---|---|
| Greedy | luôn chọn từ xác suất cao nhất | nhanh, dễ lặp từ |
| Beam search | giữ k câu tốt nhất cùng lúc | **dịch máy**, tóm tắt |
| Temperature | < 1 an toàn hơn, > 1 sáng tạo hơn | chat, viết văn |
| Top-k | chỉ bốc thăm trong k từ tốt nhất | chat, viết văn |

```python
def generate(model, prompt, num_chars=40, temperature=0.7, top_k=5):
    ids = torch.tensor([[char_to_id[c] for c in prompt]])
    model.eval()
    with torch.no_grad():
        for _ in range(num_chars):
            logits = model(ids[:, -model.context:])
            last = logits[0, -1] / temperature                    # logits của vị trí cuối

            top_values, top_ids = last.topk(top_k)                # chỉ giữ k từ tốt nhất
            probs = torch.softmax(top_values, dim=-1)
            pick = torch.multinomial(probs, 1)
            next_id = top_ids[pick]

            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    return "".join(chars[i] for i in ids[0])


print(generate(gpt, "học"))
```

---

## 7. Lấy vector cho cả câu (Feature Extraction)

Lấy trung bình vector các token thật (bỏ PAD) → 1 vector cho cả câu.

```python
def mean_pooling(hidden, ids, pad_id=0):
    mask = (ids != pad_id).unsqueeze(-1).float()        # 1 = token thật, 0 = PAD
    total = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1)
    return total / count


ids = torch.tensor([[5, 6, 7, 0, 0],
                    [8, 9, 10, 11, 12]])
sentence_vectors = mean_pooling(encoder(ids), ids)
print(sentence_vectors.shape)                           # (2, 128)
```

---

## 8. Mẹo khi train (gặp ở mọi bài)

| Mẹo | Để làm gì |
|---|---|
| AdamW, lr 2e-5 → 5e-5 | chuẩn khi fine-tune |
| Warmup 10% số bước | đầu train tăng lr từ từ, không phá model |
| Gradient clipping = 1.0 | tránh gradient quá lớn |
| Dropout 0.1 | chống học vẹt |
| Label smoothing 0.1 | model bớt quá tự tin (hay dùng cho dịch máy) |
| Nhãn `-100` | vị trí bị bỏ qua khi tính loss |

---

## 9. Thư viện có sẵn làm gì thay mình

| Tự viết | PyTorch | HuggingFace |
|---|---|---|
| `attention` | `F.scaled_dot_product_attention` | — |
| `MultiHeadAttention` | `nn.MultiheadAttention` | — |
| `EncoderLayer` | `nn.TransformerEncoderLayer` | — |
| `Encoder` | `nn.TransformerEncoder` | `AutoModel` |
| `MiniGPT` + `generate` | — | `AutoModelForCausalLM`, `model.generate()` |
| `mask_tokens` | — | `DataCollatorForLanguageModeling` |
| `mean_pooling` | — | `sentence-transformers` |

---

## 10. Tự kiểm tra

1. Vì sao chia cho √d trong attention?
2. Padding mask và causal mask khác nhau thế nào?
3. Vì sao BERT giới hạn 512 token? (gợi ý: attention tốn L × L)
4. Dịch máy nên dùng greedy, beam hay top-k? Vì sao?
