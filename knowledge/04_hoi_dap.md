# 04 · Hỏi đáp trích xuất (Extractive Question Answering)

> Code ở đây dùng lại các lớp trong file `00_nen_tang_transformer.md`.

---

## 1. Ý tưởng

Câu trả lời là **một đoạn nằm sẵn trong văn bản**. Model chỉ cần chỉ ra **vị trí bắt đầu** và **vị trí kết thúc**.

```
đoạn văn :  Lan  sống  ở  Huế  .  Nam  sống  ở  Vinh  .
vị trí   :   0    1    2   3   4   5    6    7   8    9
câu hỏi  :  "Nam sống ở đâu ?"
đáp án   :  start = 8, end = 8   →  "Vinh"
```

| Loại QA | Cách làm |
|---|---|
| **Extractive** (file này) | chọn đoạn trong văn bản – SQuAD, UIT-ViQuAD |
| Abstractive / Generative | tự viết câu trả lời – dùng seq2seq hoặc LLM |
| Open-domain | tìm tài liệu trước (file 03), rồi trích đáp án |

---

## 2. Kiến trúc

```
[CLS] câu hỏi [SEP] đoạn văn [SEP]
            │
         Encoder
            │
   (batch, len, d)
            │
     Linear(d, 2)          ← 2 điểm cho mỗi token: "là điểm bắt đầu?" và "là điểm kết thúc?"
        ┌───┴───┐
  start_logits  end_logits     (batch, len)
```

Loss = cross-entropy cho vị trí bắt đầu + cross-entropy cho vị trí kết thúc (coi mỗi vị trí như một "lớp").

---

## 3. Dữ liệu đồ chơi

```python
qa_people = ["Lan", "Nam", "Minh", "Hoa", "Tuấn"]
qa_cities = ["Huế", "Vinh", "Hà_Nội", "Đà_Lạt", "Cần_Thơ"]


def make_qa_example():
    p1, p2 = random.sample(qa_people, 2)
    c1, c2 = random.sample(qa_cities, 2)
    context = f"{p1} sống ở {c1} . {p2} sống ở {c2} ."
    asked = random.choice([p1, p2])
    answer = c1 if asked == p1 else c2
    question = f"{asked} sống ở đâu ?"
    return question, context, answer


qa_vocab = {"<pad>": 0, "<cls>": 1, "<sep>": 2}
for _ in range(300):
    q, c, a = make_qa_example()
    for w in (q + " " + c).split():
        if w not in qa_vocab:
            qa_vocab[w] = len(qa_vocab)
```

Ghép thành `[CLS] câu hỏi [SEP] đoạn văn [SEP]` và tìm vị trí đáp án:

```python
def build_qa_input(question, context, answer):
    q_words = question.split()
    c_words = context.split()
    words = ["<cls>"] + q_words + ["<sep>"] + c_words + ["<sep>"]
    ids = [qa_vocab[w] for w in words]

    context_offset = len(q_words) + 2                 # đoạn văn bắt đầu sau [CLS] câu hỏi [SEP]
    answer_position = context_offset + c_words.index(answer)
    return ids, answer_position, answer_position, words   # đáp án 1 từ: start = end
```

---

## 4. Model và train

```python
class SpanQA(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model=d_model, num_heads=4, num_layers=2, d_ff=128)
        self.qa_head = nn.Linear(d_model, 2)

    def forward(self, ids):
        hidden = self.encoder(ids)                    # (batch, len, d)
        logits = self.qa_head(hidden)                 # (batch, len, 2)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        return start_logits, end_logits


qa_model = SpanQA(len(qa_vocab))
optimizer = torch.optim.AdamW(qa_model.parameters(), lr=1e-3)

for step in range(300):
    batch = [build_qa_input(*make_qa_example()) for _ in range(32)]
    ids = torch.tensor([item[0] for item in batch])           # câu cùng độ dài → không cần pad
    starts = torch.tensor([item[1] for item in batch])
    ends = torch.tensor([item[2] for item in batch])

    start_logits, end_logits = qa_model(ids)
    loss = (F.cross_entropy(start_logits, starts) + F.cross_entropy(end_logits, ends)) / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 5. Chọn đáp án

Không chỉ lấy argmax start và argmax end riêng lẻ (có thể ra end < start). Phải tìm **cặp (start, end) có tổng điểm cao nhất** thoả:
- `start ≤ end`
- độ dài ≤ giới hạn (vd 30 token)
- nằm **trong đoạn văn** (không nằm trong câu hỏi)

```python
def best_span(start_logits, end_logits, context_start, context_end, max_answer_len=10):
    best_score = float("-inf")
    best = (0, 0)
    for s in range(context_start, context_end + 1):
        for e in range(s, min(s + max_answer_len, context_end + 1)):
            score = start_logits[s] + end_logits[e]
            if score > best_score:
                best_score = score
                best = (s, e)
    return best


question, context, answer = make_qa_example()
ids, _, _, words = build_qa_input(question, context, answer)
context_start = words.index("<sep>") + 1
context_end = len(words) - 2

qa_model.eval()
with torch.no_grad():
    start_logits, end_logits = qa_model(torch.tensor([ids]))
s, e = best_span(start_logits[0].tolist(), end_logits[0].tolist(), context_start, context_end)
print(question, "→", " ".join(words[s : e + 1]), "| đúng:", answer)
```

**Câu hỏi không có đáp án** (SQuAD 2.0, ViQuAD 2.0):
- Khi train: nhãn start = end = 0 (trỏ vào `[CLS]`).
- Khi dự đoán: nếu `điểm[CLS] > điểm đoạn tốt nhất + ngưỡng` → trả lời rỗng. Ngưỡng được dò trên tập dev.

---

## 6. Đoạn văn dài: Sliding window

Model chỉ nhận tối đa 512 token. Đoạn văn dài hơn → cắt thành nhiều **cửa sổ chồng nhau**:

```
đoạn văn  : [==========================================]
cửa sổ 1  : [câu hỏi | ==============]
cửa sổ 2  : [câu hỏi |          ==============]          chồng nhau "stride" token
cửa sổ 3  : [câu hỏi |                    ==============]
```

- Mỗi cửa sổ đều có câu hỏi.
- Chồng nhau để đáp án nằm ở ranh giới không bị cắt đôi.
- Dự đoán: tìm đoạn tốt nhất trong **tất cả** cửa sổ.

**Offset mapping:** token nào ứng với ký tự thứ mấy trong văn bản gốc → để cắt đúng chữ trả lời từ văn bản gốc (giữ nguyên dấu, hoa thường).

---

## 7. Chấm điểm: EM và F1

| Metric | Cách tính |
|---|---|
| **EM** (Exact Match) | dự đoán **giống hệt** đáp án (sau khi bỏ dấu câu, viết thường) → 1, ngược lại 0 |
| **F1** | đếm **từ chung** giữa dự đoán và đáp án → precision, recall, F1 |

```
đáp án   : "thành phố Huế"
dự đoán  : "Huế"
EM = 0      F1 = 2 × (1/1 × 1/3) / (1/1 + 1/3) = 0.5
```

```python
def qa_f1(prediction, truth):
    pred_words = prediction.lower().split()
    true_words = truth.lower().split()
    common = Counter(pred_words) & Counter(true_words)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_words)
    recall = num_same / len(true_words)
    return 2 * precision * recall / (precision + recall)


print(qa_f1("Huế", "thành phố Huế"))     # 0.5
```

Nếu có nhiều đáp án chuẩn → lấy điểm **cao nhất**.

---

## 8. Thư viện có sẵn

| Tự viết | Thư viện |
|---|---|
| `SpanQA` | `AutoModelForQuestionAnswering` |
| `build_qa_input`, sliding window | `tokenizer(question, context, stride=128, return_overflowing_tokens=True, return_offsets_mapping=True)` |
| `best_span` + ghép cửa sổ | `pipeline("question-answering")` |
| `qa_f1` | `evaluate.load("squad_v2")` |

```py
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
qa(question="Nam sống ở đâu?", context="Lan sống ở Huế. Nam sống ở Vinh.")
# {'answer': 'Vinh', 'start': ..., 'end': ..., 'score': ...}
```

⚠️ QA cần tokenizer **fast** (có offset mapping) → dùng XLM-R / mDeBERTa, không dùng PhoBERT.

---

## 9. Tự kiểm tra

1. Vì sao QA trích xuất chỉ cần 2 lớp Linear output (start, end)?
2. Vì sao không lấy argmax start và argmax end riêng?
3. Sliding window giải quyết vấn đề gì? Vì sao các cửa sổ phải chồng nhau?
4. Xử lý câu hỏi không có đáp án thế nào?
5. Vì sao khởi tạo từ model đã học SQuAD lại tốt hơn nhiều so với model gốc?
