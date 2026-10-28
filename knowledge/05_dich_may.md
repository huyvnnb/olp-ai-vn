# 05 · Dịch máy (Machine Translation) ⭐ ưu tiên số 1

> Chạy code ở đây **sau** file `00_nen_tang_transformer.md` (dùng lại `MultiHeadAttention`, `FeedForward`, `EncoderLayer`, `PositionalEncoding`, các hàm mask).

---

## 1. Ý tưởng

```
Câu nguồn: "the red car"
        │
   ┌────▼────┐      đọc hiểu cả câu nguồn
   │ ENCODER │
   └────┬────┘
        │  (vector của từng từ nguồn)
        ▼
   ┌─────────┐      viết câu đích TỪNG TỪ MỘT,
   │ DECODER │ ◄──  mỗi bước nhìn lại câu nguồn (cross-attention)
   └────┬────┘
        ▼
Câu đích: "xe đỏ"
```

Decoder sinh lần lượt:

```
<bos>              → "xe"
<bos> xe           → "đỏ"
<bos> xe đỏ        → <eos>   (dừng)
```

Lịch sử ngắn:

| Thời kỳ | Cách làm |
|---|---|
| trước 2014 | dịch thống kê theo cụm từ (SMT) |
| 2014 | RNN encoder-decoder |
| 2015 | RNN + attention |
| 2017 → nay | **Transformer** (Marian, mBART, NLLB, T5) |
| hiện nay | LLM (GPT, Qwen…) cũng dịch được |

---

## 2. Dữ liệu đồ chơi

Tiếng Anh: tính từ đứng **trước** danh từ. Tiếng Việt: tính từ đứng **sau**.
Model phải học được cả **dịch từ** lẫn **đảo thứ tự**.

```
"the red car and the small cat"  →  "xe đỏ và mèo nhỏ"
```

```python
adjectives = {"red": "đỏ", "big": "lớn", "small": "nhỏ", "old": "cũ", "new": "mới", "blue": "xanh"}
nouns = {"car": "xe", "house": "nhà", "cat": "mèo", "dog": "chó", "book": "sách", "table": "bàn"}


def make_pair():
    src_parts = []
    tgt_parts = []
    num_phrases = random.randint(1, 3)
    for _ in range(num_phrases):
        adj = random.choice(list(adjectives))
        noun = random.choice(list(nouns))
        src_parts.append(f"the {adj} {noun}")
        tgt_parts.append(f"{nouns[noun]} {adjectives[adj]}")      # đảo thứ tự
    return " and ".join(src_parts), " và ".join(tgt_parts)


train_pairs = [make_pair() for _ in range(3000)]
test_pairs = [make_pair() for _ in range(100)]
print(train_pairs[0])
```

Từ điển (mỗi từ 1 số) và 3 token đặc biệt:

| Token | Số | Ý nghĩa |
|---|---|---|
| `<pad>` | 0 | đệm cho các câu dài bằng nhau |
| `<bos>` | 1 | bắt đầu câu đích |
| `<eos>` | 2 | kết thúc câu |

```python
PAD, BOS, EOS = 0, 1, 2


def build_vocab(sentences):
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


src_vocab = build_vocab([src for src, tgt in train_pairs])
tgt_vocab = build_vocab([tgt for src, tgt in train_pairs])
id_to_tgt_word = {i: w for w, i in tgt_vocab.items()}


def to_ids(sentence, vocab):
    return [vocab[word] for word in sentence.split()]


def pad(sequences):
    max_len = max(len(s) for s in sequences)
    padded = [s + [PAD] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded)
```

---

## 3. Mô hình Encoder–Decoder

Decoder layer có **3 khối** (encoder chỉ có 2):

```
y ─► Self-Attention (có causal mask: không nhìn từ tương lai)
  ─► Cross-Attention (query = câu đích, key/value = câu nguồn)
  ─► FeedForward
```

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask, src_mask):
        h = self.norm1(y)
        y = y + self.dropout(self.self_attn(h, h, tgt_mask))

        h = self.norm2(y)
        y = y + self.dropout(self.cross_attn(h, memory, src_mask))   # nhìn sang câu nguồn

        h = self.norm3(y)
        y = y + self.dropout(self.ff(h))
        return y
```

```python
class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=128):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD)
        self.position = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(EncoderLayer(d_model, num_heads, d_ff))
            self.decoder_layers.append(DecoderLayer(d_model, num_heads, d_ff))

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, tgt_vocab_size)          # điểm cho từng từ trong từ điển đích

    def encode(self, src):
        src_mask = make_padding_mask(src, PAD)
        x = self.position(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x), src_mask

    def decode(self, tgt_in, memory, src_mask):
        length = tgt_in.size(1)
        tgt_mask = make_padding_mask(tgt_in, PAD) & make_causal_mask(length)
        y = self.position(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, src_mask)
        return self.output(self.decoder_norm(y))                  # (batch, len, tgt_vocab)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)
```

---

## 4. Huấn luyện: Teacher Forcing

Khi train, ta đưa **câu đích đúng** vào decoder (dịch phải 1 vị trí), bắt model đoán từ tiếp theo ở **mọi vị trí cùng lúc**:

```
decoder nhận vào :  <bos>  xe   đỏ
nhãn cần đoán    :  xe     đỏ   <eos>
```

```python
def make_batch(pairs):
    src_list = []
    tgt_in_list = []
    tgt_out_list = []
    for src, tgt in pairs:
        src_ids = to_ids(src, src_vocab) + [EOS]
        tgt_ids = to_ids(tgt, tgt_vocab)
        src_list.append(src_ids)
        tgt_in_list.append([BOS] + tgt_ids)       # đầu vào decoder
        tgt_out_list.append(tgt_ids + [EOS])      # nhãn
    return pad(src_list), pad(tgt_in_list), pad(tgt_out_list)
```

Loss = cross-entropy, **bỏ qua PAD**, thêm **label smoothing** (chuẩn trong dịch máy):

```python
model = Translator(len(src_vocab), len(tgt_vocab))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

model.train()
for step in range(800):
    batch = random.sample(train_pairs, 32)
    src, tgt_in, tgt_out = make_batch(batch)

    logits = model(src, tgt_in)
    loss = loss_fn(logits.reshape(-1, len(tgt_vocab)), tgt_out.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 200 == 0:
        print("step", step, "loss", round(loss.item(), 3))
```

---

## 5. Dịch: Greedy và Beam Search

Khi dịch thật, không có câu đích → decoder phải **tự dùng output của chính nó**.

### Greedy: mỗi bước chọn từ tốt nhất

```python
def greedy_translate(model, sentence, max_len=30):
    model.eval()
    src = torch.tensor([to_ids(sentence, src_vocab) + [EOS]])
    with torch.no_grad():
        memory, src_mask = model.encode(src)            # encoder chỉ chạy 1 lần
        output = [BOS]
        for _ in range(max_len):
            logits = model.decode(torch.tensor([output]), memory, src_mask)
            next_id = logits[0, -1].argmax().item()
            if next_id == EOS:
                break
            output.append(next_id)
    return " ".join(id_to_tgt_word[i] for i in output[1:])


print(greedy_translate(model, "the big dog and the blue book"))
```

### Beam search: giữ k bản dịch tốt nhất cùng lúc

Greedy có thể chọn sai sớm rồi không sửa được. Beam search giữ `beam_size` ứng viên, cộng **log xác suất** để so sánh.

```
bước 1:  [xe -0.1]  [nhà -2.3]  [chó -3.0]           giữ 3 ứng viên tốt nhất
bước 2:  [xe đỏ -0.2] [xe lớn -2.5] [nhà đỏ -2.6]    mở rộng từng ứng viên, lại giữ 3
...
```

Câu dài có tổng log-prob thấp hơn → chia cho `độ dài ** length_penalty` để công bằng.

```python
def beam_search(model, sentence, beam_size=3, max_len=30, length_penalty=1.0):
    model.eval()
    src = torch.tensor([to_ids(sentence, src_vocab) + [EOS]])
    with torch.no_grad():
        memory, src_mask = model.encode(src)

        beams = [([BOS], 0.0)]                          # (các id, tổng log-prob)
        finished = []
        for _ in range(max_len):
            candidates = []
            for ids, score in beams:
                logits = model.decode(torch.tensor([ids]), memory, src_mask)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                top_scores, top_ids = log_probs.topk(beam_size)
                for s, i in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append((ids + [i], score + s))

            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for ids, score in candidates[:beam_size]:
                if ids[-1] == EOS:
                    length = len(ids) - 1
                    finished.append((ids, score / length ** length_penalty))
                else:
                    beams.append((ids, score))
            if len(beams) == 0:
                break

    if len(finished) == 0:
        finished = beams
    best_ids = max(finished, key=lambda c: c[1])[0]
    words = [id_to_tgt_word[i] for i in best_ids[1:] if i != EOS]
    return " ".join(words)


print(beam_search(model, "the big dog and the blue book"))
```

---

## 6. Chấm điểm: BLEU

BLEU đo **bản dịch máy trùng bao nhiêu cụm 1, 2, 3, 4 từ** với bản dịch chuẩn.

```
BLEU = BP × trung bình nhân (precision 1-gram, 2-gram, 3-gram, 4-gram)
BP   = phạt nếu bản dịch NGẮN hơn bản chuẩn (để không "ăn gian" bằng câu cực ngắn)
```

```python
def count_ngrams(words, n):
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def corpus_bleu(hypotheses, references, max_n=4):
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()
        ref_words = ref.split()
        hyp_len += len(hyp_words)
        ref_len += len(ref_words)
        for n in range(1, max_n + 1):
            hyp_ngrams = count_ngrams(hyp_words, n)
            ref_ngrams = count_ngrams(ref_words, n)
            overlap = hyp_ngrams & ref_ngrams                       # lấy min số lần xuất hiện
            matches[n - 1] += sum(overlap.values())
            totals[n - 1] += max(len(hyp_words) - n + 1, 0)

    if min(matches) == 0:
        return 0.0

    log_precision = 0.0
    for m, t in zip(matches, totals):
        log_precision += math.log(m / t)
    log_precision /= max_n

    if hyp_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - ref_len / hyp_len)
    return 100 * brevity_penalty * math.exp(log_precision)
```

```python
references = [tgt for src, tgt in test_pairs]
greedy_outputs = [greedy_translate(model, src) for src, tgt in test_pairs]
beam_outputs = [beam_search(model, src) for src, tgt in test_pairs]

print("BLEU greedy:", round(corpus_bleu(greedy_outputs, references), 2))
print("BLEU beam  :", round(corpus_bleu(beam_outputs, references), 2))
```

Các metric khác:

| Metric | Ý tưởng |
|---|---|
| **chrF / chrF++** | như BLEU nhưng đếm theo **ký tự** → hợp với tiếng Việt, ít bị ảnh hưởng tách từ |
| **TER** | số lần sửa để bản dịch thành bản chuẩn (càng thấp càng tốt) |
| **COMET** | model AI chấm điểm, gần đánh giá của người nhất |

Khi thi luôn dùng **sacreBLEU** (chuẩn, ai tính cũng ra cùng số), không tự viết.

---

## 7. Các mẹo tăng điểm (hiểu nguyên lý)

| Mẹo | Nguyên lý |
|---|---|
| Model pretrained đa ngữ | đã học dịch trên hàng trăm triệu câu → fine-tune chỉ cần ít dữ liệu |
| Làm sạch dữ liệu | bỏ câu trùng, câu lệch độ dài, câu chưa dịch → model không học sai |
| Label smoothing 0.1 | model bớt quá tự tin → beam search hoạt động tốt hơn |
| Beam 4–5 + length_penalty | tìm bản dịch tốt hơn greedy |
| **Back-translation** | dịch ngược văn bản tiếng đích → nguồn để tạo thêm cặp câu giả |
| **MBR decoding** | sinh nhiều bản dịch, chọn bản **giống các bản khác nhất** (bản "đồng thuận") |
| Ensemble | gộp ứng viên của nhiều model rồi chọn bằng MBR |

MBR, ví dụ đơn giản (độ giống = số từ chung):

```python
def word_overlap(a, b):
    common = Counter(a.split()) & Counter(b.split())
    return sum(common.values())


def mbr_pick(candidates):
    best = None
    best_score = -1
    for cand in candidates:
        score = 0
        for other in candidates:
            if other is not cand:
                score += word_overlap(cand, other)
        if score > best_score:
            best = cand
            best_score = score
    return best


print(mbr_pick(["xe đỏ và chó lớn", "xe đỏ và chó to", "xe đỏ và mèo lớn", "nhà đỏ"]))
```

---

## 8. Thư viện có sẵn

| Tự viết | Thư viện |
|---|---|
| `Translator` | `AutoModelForSeq2SeqLM` (Marian, mBART, NLLB, T5) / `nn.Transformer` |
| `make_batch` (teacher forcing) | `DataCollatorForSeq2Seq` (tự tạo `labels` và đầu vào decoder) |
| vòng lặp train | `Seq2SeqTrainer` |
| `greedy_translate`, `beam_search` | `model.generate(num_beams=5, length_penalty=1.0)` |
| `corpus_bleu` | `sacrebleu.corpus_bleu(hyps, [refs])` |

Dịch bằng model có sẵn:

```py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name)

inputs = tokenizer(["The red car is very fast."], return_tensors="pt")
outputs = model.generate(**inputs, num_beams=5)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

Model **đa ngữ** (NLLB, mBART) cần biết dịch sang ngôn ngữ nào → ép token đầu tiên của decoder là mã ngôn ngữ đích:

```py
name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(name)

inputs = tokenizer(["The red car"], return_tensors="pt")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("vie_Latn"))
```

| Model | Cách chỉ định ngôn ngữ |
|---|---|
| Marian (`opus-mt-en-vi`) | mỗi model 1 chiều, không cần |
| NLLB | `src_lang="eng_Latn"`, `forced_bos_token_id` = `vie_Latn` |
| mBART-50 | `src_lang="en_XX"`, `forced_bos_token_id` = `vi_VN` |
| EnViT5 | thêm chữ `"en: "` trước câu nguồn |

Chấm điểm:

```py
import sacrebleu
print(sacrebleu.corpus_bleu(hypotheses, [references]).score)
print(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score)   # chrF++
```

---

## 9. Tự kiểm tra

1. Teacher forcing là gì? Vì sao khi dịch thật lại không dùng được?
2. Decoder có mấy loại attention? Mỗi loại dùng mask gì?
3. Vì sao beam search cần chia cho độ dài câu?
4. BLEU = 0 có nghĩa là gì? Vì sao BLEU có brevity penalty?
5. Back-translation tạo dữ liệu như thế nào? Vì sao lại có ích?
6. Vì sao NLLB cần `forced_bos_token_id` còn `opus-mt-en-vi` thì không?
