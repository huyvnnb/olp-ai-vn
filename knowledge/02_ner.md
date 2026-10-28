# 02 · Nhận dạng thực thể (NER / Token Classification)

> Code ở đây dùng lại các lớp trong file `00_nen_tang_transformer.md`.

---

## 1. Ý tưởng

Phân loại **từng từ** thay vì cả câu.

```
Lan    sống   ở    Hà     Nội
B-PER  O      O    B-LOC  I-LOC
```

**Định dạng BIO:**

| Nhãn | Nghĩa |
|---|---|
| `B-X` | **B**ắt đầu thực thể loại X |
| `I-X` | t**I**ếp theo của thực thể X |
| `O` | không phải thực thể |

Vì sao cần B và I? Để tách 2 thực thể đứng liền nhau:
`Hà Nội Huế` → `B-LOC I-LOC B-LOC` (2 địa danh, không phải 1).

Các bài cùng dạng: **POS tagging** (từ loại), **chunking**, **tách từ tiếng Việt** (B-W / I-W).

---

## 2. Kiến trúc

```
từ:      Lan    sống    ở     Hà     Nội
          │      │      │      │      │
       ┌──▼──────▼──────▼──────▼──────▼──┐
       │            Encoder              │
       └──┬──────┬──────┬──────┬──────┬──┘
          ▼      ▼      ▼      ▼      ▼
       Linear(d, số_nhãn) cho TỪNG vị trí
          ▼      ▼      ▼      ▼      ▼
       B-PER     O      O    B-LOC  I-LOC
```

Khác phân loại câu đúng 1 chỗ: **không gộp** (pooling), mà đưa từng vector qua Linear.

---

## 3. Dữ liệu đồ chơi

```python
people = ["Lan", "Nam", "Minh", "Hoa"]
places = [["Hà", "Nội"], ["Huế"], ["Đà", "Nẵng"], ["Cần", "Thơ"]]


def make_sentence():
    person = random.choice(people)
    place = random.choice(places)
    words = [person, "sống", "ở"] + place
    tags = ["B-PER", "O", "O"] + ["B-LOC"] + ["I-LOC"] * (len(place) - 1)
    return words, tags


ner_train = [make_sentence() for _ in range(500)]
print(ner_train[0])

tag_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
tag2id = {t: i for i, t in enumerate(tag_list)}

ner_vocab = {"<pad>": 0}
for words, tags in ner_train:
    for w in words:
        if w not in ner_vocab:
            ner_vocab[w] = len(ner_vocab)
```

---

## 4. Model

```python
class TokenTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, d_model=64):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model=d_model, num_heads=4, num_layers=2, d_ff=128)
        self.head = nn.Linear(d_model, num_tags)

    def forward(self, ids):
        hidden = self.encoder(ids)        # (batch, len, d)
        return self.head(hidden)          # (batch, len, num_tags) – mỗi từ một dự đoán
```

Train: loss tính cho **mọi vị trí**, bỏ qua PAD bằng nhãn `-100`.

```python
def make_ner_batch(batch):
    max_len = max(len(words) for words, tags in batch)
    ids = []
    labels = []
    for words, tags in batch:
        pad_len = max_len - len(words)
        ids.append([ner_vocab[w] for w in words] + [0] * pad_len)
        labels.append([tag2id[t] for t in tags] + [-100] * pad_len)
    return torch.tensor(ids), torch.tensor(labels)


tagger = TokenTagger(len(ner_vocab), len(tag_list))
optimizer = torch.optim.AdamW(tagger.parameters(), lr=1e-3)

for step in range(200):
    ids, labels = make_ner_batch(random.sample(ner_train, 32))
    logits = tagger(ids)
    loss = F.cross_entropy(logits.reshape(-1, len(tag_list)), labels.reshape(-1), ignore_index=-100)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 5. Vấn đề sub-token

Model thật dùng subword → 1 từ có thể thành nhiều token. Nhưng nhãn là **theo từ**.

```
từ    :  Washington           là   thủ   đô
token :  ▁Washing   ton       ▁là  ▁thủ  ▁đô
nhãn  :  B-LOC      -100      O    O     O
```

Quy tắc: **chỉ token đầu tiên của mỗi từ giữ nhãn**, các token sau đặt `-100` (bỏ qua khi tính loss).
Khi dự đoán: lấy nhãn của token đầu tiên làm nhãn của từ.

```python
def align_labels(word_ids, word_tags):
    # word_ids: mỗi token thuộc từ thứ mấy (None = token đặc biệt)
    labels = []
    previous = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != previous:
            labels.append(tag2id[word_tags[word_id]])
        else:
            labels.append(-100)
        previous = word_id
    return labels


# ví dụ: <s> Washing ton là thủ đô </s>
print(align_labels([None, 0, 0, 1, 2, 3, None], ["B-LOC", "O", "O", "O"]))
```

---

## 6. Chấm điểm: F1 theo thực thể

**Không** tính đúng/sai từng token. Một thực thể chỉ đúng khi **đúng cả ranh giới và loại**.

```
chuẩn : B-LOC I-LOC        → thực thể (LOC, từ 3 đến 4)
dự đoán: B-LOC O            → thực thể (LOC, từ 3 đến 3)   ✗ sai ranh giới
```

```python
def get_entities(tags):
    entities = []
    start = None
    kind = None
    for i, tag in enumerate(tags + ["O"]):          # thêm "O" để đóng thực thể cuối
        starts_new = tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != kind)
        if starts_new and start is not None:
            entities.append((kind, start, i - 1))
            start = None
            kind = None
        if tag.startswith("B-") or (tag.startswith("I-") and start is None):
            start = i
            kind = tag[2:]
    return set(entities)


def entity_f1(true_sentences, pred_sentences):
    tp = 0
    num_pred = 0
    num_true = 0
    for true_tags, pred_tags in zip(true_sentences, pred_sentences):
        true_entities = get_entities(true_tags)
        pred_entities = get_entities(pred_tags)
        tp += len(true_entities & pred_entities)
        num_pred += len(pred_entities)
        num_true += len(true_entities)
    if tp == 0:
        return 0.0
    precision = tp / num_pred
    recall = tp / num_true
    return 2 * precision * recall / (precision + recall)


print(get_entities(["B-PER", "O", "O", "B-LOC", "I-LOC"]))
print(entity_f1([["B-LOC", "I-LOC"]], [["B-LOC", "O"]]))     # 0.0 vì sai ranh giới
```

---

## 7. CRF (biết khái niệm)

Model trên dự đoán **từng từ độc lập** → có thể ra chuỗi vô lý như `O I-LOC`.

**CRF** thêm lớp cuối học **điểm chuyển** giữa các nhãn (vd `O → I-LOC` bị điểm rất thấp), rồi dùng thuật toán **Viterbi** tìm chuỗi nhãn tốt nhất cho cả câu.

| | Không CRF | Có CRF |
|---|---|---|
| Dự đoán | từng từ riêng | cả chuỗi |
| Chuỗi vô lý | có thể xảy ra | không |
| Với BERT | thường đủ tốt | thêm khoảng +0.2–0.5 F1 |

Cách rẻ hơn: **sửa BIO sau khi dự đoán** (`I-X` sau `O` → `B-X`), như hàm `fix_bio` trong notebook.

---

## 8. Thư viện có sẵn

| Tự viết | Thư viện |
|---|---|
| `TokenTagger` | `AutoModelForTokenClassification` |
| `align_labels` | `tokenizer(..., is_split_into_words=True).word_ids()` |
| `make_ner_batch` | `DataCollatorForTokenClassification` (đệm nhãn bằng -100) |
| `entity_f1` | `seqeval.metrics.f1_score` |
| CRF | `pytorch-crf` (`torchcrf.CRF`) |

```py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
encoded = tokenizer(["Lan", "sống", "ở", "Hà", "Nội"], is_split_into_words=True)
print(encoded.word_ids())    # [None, 0, 1, 2, 3, 4, None]
```

---

## 9. Tự kiểm tra

1. Vì sao cần cả B và I, không chỉ dùng tên loại?
2. 1 từ bị cắt thành 3 token thì gán nhãn thế nào? Vì sao không gán nhãn cho cả 3?
3. Vì sao accuracy theo token không phản ánh đúng chất lượng NER?
4. CRF giải quyết vấn đề gì?
