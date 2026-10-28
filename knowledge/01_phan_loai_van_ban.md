# 01 · Phân loại văn bản (Text Classification)

> Code ở đây dùng lại các lớp trong file `00_nen_tang_transformer.md`.

---

## 1. Ý tưởng

Đưa vào **một đoạn văn** → trả ra **một nhãn**.

| Bài | Ví dụ nhãn |
|---|---|
| Cảm xúc | tích cực / tiêu cực / trung tính |
| Chủ đề | thể thao / kinh tế / giải trí |
| Spam, toxic | có / không |
| Intent (chatbot) | đặt vé / hỏi giá / huỷ đơn |

```
"phim này hay quá"  →  Encoder  →  vector cả câu  →  Linear  →  [0.1, 0.9]  →  "tích cực"
```

---

## 2. Dữ liệu đồ chơi

```python
positive_words = ["hay", "tuyệt", "đẹp", "thích", "ổn"]
negative_words = ["dở", "tệ", "chán", "ghét", "kém"]
subjects = ["phim", "quán", "sách", "áo", "điện thoại"]


def make_example():
    subject = random.choice(subjects)
    if random.random() < 0.5:
        return f"{subject} này {random.choice(positive_words)} quá", 1     # 1 = tích cực
    else:
        return f"{subject} này {random.choice(negative_words)} quá", 0     # 0 = tiêu cực


train_data = [make_example() for _ in range(1000)]
test_data = [make_example() for _ in range(200)]
print(train_data[:3])
```

---

## 3. Cách 1 – Baseline: Bag of Words + Logistic Regression

Ý tưởng đơn giản nhất: **đếm từ**. Câu có từ "hay" nhiều → tích cực.

```
từ điển : [phim, này, hay, dở, quá, ...]
"phim này hay quá"  →  [1, 1, 1, 0, 1, ...]
```

Cách này nhanh, không cần GPU, và **thường đã đạt điểm khá**. Luôn làm baseline này trước.

```python
word_vocab = {}
for text, label in train_data:
    for word in text.split():
        if word not in word_vocab:
            word_vocab[word] = len(word_vocab)


def bag_of_words(text):
    vector = torch.zeros(len(word_vocab))
    for word in text.split():
        if word in word_vocab:
            vector[word_vocab[word]] += 1
    return vector


X_train = torch.stack([bag_of_words(text) for text, label in train_data])
y_train = torch.tensor([label for text, label in train_data])
```

Logistic regression = **một lớp Linear** + softmax:

```python
logreg = nn.Linear(len(word_vocab), 2)
optimizer = torch.optim.AdamW(logreg.parameters(), lr=0.1)

for epoch in range(100):
    logits = logreg(X_train)
    loss = F.cross_entropy(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

X_test = torch.stack([bag_of_words(text) for text, label in test_data])
y_test = torch.tensor([label for text, label in test_data])
pred = logreg(X_test).argmax(dim=1)
print("accuracy:", (pred == y_test).float().mean().item())
```

**TF-IDF** (bản tốt hơn của đếm từ): từ xuất hiện ở **mọi** câu (như "này", "quá") ít giá trị → giảm trọng số.

```
tf-idf(từ, câu) = số lần từ xuất hiện trong câu  ×  log(tổng số câu / số câu chứa từ)
```

---

## 4. Cách 2 – Transformer

Bag of words không hiểu thứ tự: "không hay" và "hay không" như nhau. Transformer hiểu ngữ cảnh.

```
ids → Encoder → (batch, len, d) → gộp thành 1 vector/câu → Linear(d, số_lớp) → logits
```

Có 2 cách gộp:
- **[CLS]**: lấy vector của token đầu tiên (BERT làm vậy).
- **Mean pooling**: trung bình các token (dùng hàm `mean_pooling` ở file 00).

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=64):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model=d_model, num_heads=4, num_layers=2, d_ff=128)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        hidden = self.encoder(ids)                    # (batch, len, d)
        sentence = mean_pooling(hidden, ids)          # (batch, d)
        return self.head(self.dropout(sentence))      # (batch, num_classes)
```

Chuyển câu thành ids (số 0 dành cho PAD):

```python
token_vocab = {"<pad>": 0}
for text, label in train_data:
    for word in text.split():
        if word not in token_vocab:
            token_vocab[word] = len(token_vocab)


def encode_batch(texts):
    ids_list = []
    for text in texts:
        ids_list.append([token_vocab.get(word, 0) for word in text.split()])
    max_len = max(len(ids) for ids in ids_list)
    padded = [ids + [0] * (max_len - len(ids)) for ids in ids_list]
    return torch.tensor(padded)
```

Train:

```python
classifier = TextClassifier(len(token_vocab), num_classes=2)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)

classifier.train()
for step in range(300):
    batch = random.sample(train_data, 32)
    ids = encode_batch([text for text, label in batch])
    labels = torch.tensor([label for text, label in batch])

    loss = F.cross_entropy(classifier(ids), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 5. Dữ liệu lệch nhãn

Ví dụ 90% tích cực, 10% tiêu cực → model đoán "tích cực" hết cũng đúng 90%.

Cách xử lý: **class weights** – lớp hiếm được phạt nặng hơn khi đoán sai.

```
weight[lớp] = tổng số mẫu / (số lớp × số mẫu của lớp đó)
```

```python
label_counts = Counter(label for text, label in train_data)
num_samples = len(train_data)
num_classes = 2
weights = torch.tensor([num_samples / (num_classes * label_counts[c]) for c in range(num_classes)])

loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
```

---

## 6. Chấm điểm

Với 1 lớp:

```
precision = đoán là lớp A và đúng  /  số lần đoán là lớp A
recall    = đoán là lớp A và đúng  /  số mẫu thật sự là lớp A
F1        = 2 × precision × recall / (precision + recall)
```

| Kiểu F1 | Cách tính | Ý nghĩa |
|---|---|---|
| **macro** | trung bình F1 của các lớp | mọi lớp quan trọng như nhau → lớp hiếm có ảnh hưởng lớn |
| micro | tính trên tất cả mẫu | với bài 1 nhãn, bằng accuracy |
| weighted | trung bình F1 theo số mẫu mỗi lớp | lớp nhiều mẫu quan trọng hơn |

```python
def f1_per_class(y_true, y_pred, cls):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def macro_f1(y_true, y_pred):
    classes = set(y_true)
    scores = [f1_per_class(y_true, y_pred, c) for c in classes]
    return sum(scores) / len(scores)


classifier.eval()
with torch.no_grad():
    test_ids = encode_batch([text for text, label in test_data])
    predictions = classifier(test_ids).argmax(dim=1).tolist()
print("macro F1:", macro_f1([label for text, label in test_data], predictions))
```

---

## 7. Biến thể

**Multi-label** (1 câu có thể nhiều nhãn, ví dụ "giá" + "chất lượng"):
- Thay softmax bằng **sigmoid** cho từng nhãn.
- Loss: `nn.BCEWithLogitsLoss`.
- Nhãn > ngưỡng (vd 0.5) thì chọn. Nên dò ngưỡng riêng cho từng nhãn.

**Cặp câu** (2 câu có cùng nghĩa không? câu B có suy ra từ câu A không?):
- Ghép 2 câu: `[CLS] câu A [SEP] câu B [SEP]` → phân loại như bình thường.

**Zero-shot** (không có dữ liệu train):
- Dùng model NLI (đã học "câu A có suy ra câu B không").
- Với mỗi nhãn, tạo giả thuyết: *"Câu này mang ý nghĩa tích cực."*
- Nhãn nào có điểm "suy ra được" cao nhất → chọn nhãn đó.

---

## 8. Thư viện có sẵn

| Tự viết | Thư viện |
|---|---|
| `bag_of_words`, TF-IDF | `sklearn.feature_extraction.text.TfidfVectorizer` |
| Logistic regression | `sklearn.linear_model.LogisticRegression` |
| `TextClassifier` | `AutoModelForSequenceClassification` |
| `encode_batch` | `AutoTokenizer` + `DataCollatorWithPadding` |
| Vòng lặp train | `Trainer` |
| `macro_f1` | `sklearn.metrics.f1_score(average="macro")` |
| Zero-shot | `pipeline("zero-shot-classification")` |

```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

inputs = tokenizer(["phim này hay quá"], return_tensors="pt")
logits = model(**inputs).logits
```

`AutoModelForSequenceClassification` bên trong chính là: **Encoder pretrained + lấy vector [CLS] + Dropout + Linear** – giống hệt `TextClassifier` ở trên, chỉ khác là encoder đã được pretrain trên rất nhiều văn bản.

---

## 9. Tự kiểm tra

1. Vì sao nên làm baseline TF-IDF trước khi fine-tune transformer?
2. Dữ liệu lệch 95/5, accuracy 95% có phải model tốt không? Nên xem metric gì?
3. Multi-label khác multi-class ở chỗ nào (output, loss, cách chọn nhãn)?
4. Zero-shot classification hoạt động thế nào?
