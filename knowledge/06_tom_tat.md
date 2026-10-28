# 06 · Tóm tắt văn bản (Summarization)

> Code ở đây dùng lại hàm trong file `00_nen_tang_transformer.md` (import, `Counter`, `math`).
> Phần model encoder–decoder giống hệt dịch máy → xem `05_dich_may.md`.

---

## 1. Hai cách tóm tắt

| | Trích xuất (Extractive) | Trừu tượng (Abstractive) |
|---|---|---|
| Cách làm | **chọn** các câu quan trọng nhất trong bài | **viết lại** bằng câu mới |
| Model | không cần train (TextRank) hoặc phân loại câu | encoder–decoder (ViT5, BART, mT5) |
| Ưu | nhanh, không bịa nội dung | tự nhiên, ngắn gọn |
| Nhược | câu rời rạc | có thể **bịa** thông tin (hallucination) |

```
Abstractive = Dịch máy, với:
   câu nguồn  = văn bản dài
   câu đích   = bản tóm tắt
```

---

## 2. Trích xuất: TextRank

Ý tưởng (giống PageRank của Google):
1. Mỗi câu là một **đỉnh**.
2. Hai câu càng giống nhau → cạnh nối càng nặng.
3. Câu quan trọng = câu **giống với nhiều câu quan trọng khác**.
4. Lặp tính điểm cho đến khi ổn định, chọn top câu.

```python
def sentence_similarity(a, b):
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if len(words_a) == 0 or len(words_b) == 0:
        return 0.0
    common = len(words_a & words_b)
    return common / (math.log(len(words_a) + 1) + math.log(len(words_b) + 1))


def textrank(sentences, top_n=2, damping=0.85, iterations=50):
    n = len(sentences)

    # ma trận độ giống, chuẩn hoá mỗi hàng thành tổng = 1
    weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i][j] = sentence_similarity(sentences[i], sentences[j])
        row_sum = sum(weights[i])
        if row_sum > 0:
            weights[i] = [w / row_sum for w in weights[i]]

    # lặp tính điểm
    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = []
        for i in range(n):
            incoming = sum(weights[j][i] * scores[j] for j in range(n))
            new_scores.append((1 - damping) + damping * incoming)
        scores = new_scores

    # chọn top câu, giữ thứ tự xuất hiện trong bài
    top = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_n]
    return [sentences[i] for i in sorted(top)]


article = [
    "Hà Nội hôm nay có mưa lớn.",
    "Mưa lớn khiến nhiều tuyến phố ở Hà Nội ngập sâu.",
    "Người dân đi lại khó khăn vì phố ngập.",
    "Một cửa hàng bánh mì mới khai trương.",
    "Cơ quan khí tượng dự báo mưa lớn còn kéo dài ở Hà Nội.",
]
print(textrank(article))
```

---

## 3. Trừu tượng: những điểm khác với dịch máy

Kiến trúc và cách train **giống hệt** file 05. Khác biệt:

| Vấn đề | Cách xử lý |
|---|---|
| Văn bản đầu vào rất dài | model nhận được dài (ViT5: 1024 token, LED/LongT5: 4096+), hoặc cắt đầu bài (phần quan trọng thường ở đầu) |
| Model hay **lặp lại** cụm từ | `no_repeat_ngram_size=3` khi sinh |
| Tóm tắt quá ngắn / quá dài | `length_penalty`, `min_new_tokens`, `max_new_tokens` |
| Tên riêng, số liệu hiếm bị sai | **Copy mechanism** (Pointer-Generator): mỗi bước model chọn *sinh từ mới* hay *chép từ văn bản gốc* |

### Chặn lặp n-gram

Trước khi chọn từ tiếp theo: nếu thêm từ đó tạo ra một cụm `n` từ **đã xuất hiện** → cấm từ đó.

```python
def banned_next_words(generated, n=3):
    # generated: các từ đã sinh. Trả về các từ bị cấm ở bước tiếp theo
    if len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):])          # n-1 từ cuối
    banned = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i : i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return banned


print(banned_next_words(["mưa", "lớn", "ở", "hà", "nội", "mưa", "lớn"], n=3))   # {'ở'}
```

Khi sinh: đặt logits của các từ bị cấm = `-inf` rồi mới chọn.

---

## 4. Chấm điểm: ROUGE

ROUGE đo **bản tóm tắt chuẩn được bao phủ bao nhiêu** (nghiêng về recall, khác BLEU nghiêng về precision).

| Metric | Đếm gì |
|---|---|
| ROUGE-1 | từ chung |
| ROUGE-2 | cặp 2 từ liền nhau chung |
| ROUGE-L | **chuỗi con chung dài nhất** (LCS) – các từ cùng thứ tự, không cần liền nhau |

```python
def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def rouge_l_f1(prediction, reference):
    pred = prediction.lower().split()
    ref = reference.lower().split()
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


print(rouge_l_f1("hà nội mưa lớn gây ngập", "mưa lớn gây ngập nhiều phố hà nội"))
```

⚠️ Thư viện `rouge_score` mặc định **xoá ký tự có dấu** → điểm tiếng Việt sai. Tự tính theo khoảng trắng như trên (notebook 06 đã làm sẵn).

---

## 5. Thư viện có sẵn

| Tự viết | Thư viện |
|---|---|
| `textrank` | `sumy`, `networkx.pagerank` |
| model abstractive | `AutoModelForSeq2SeqLM` (`VietAI/vit5-base`, `facebook/bart-large-cnn`) |
| `banned_next_words` | `model.generate(no_repeat_ngram_size=3)` |
| `rouge_l_f1` | `rouge_score` (chỉ an toàn với tiếng Anh) |

```py
from transformers import pipeline

summarizer = pipeline("summarization", model="VietAI/vit5-base-vietnews-summarization")
summarizer(long_text, max_new_tokens=128, num_beams=4, no_repeat_ngram_size=3)
```

---

## 6. Tự kiểm tra

1. Khác nhau giữa tóm tắt trích xuất và trừu tượng? Khi nào dùng cái nào?
2. Vì sao abstractive có thể bịa thông tin?
3. ROUGE-L khác ROUGE-2 thế nào?
4. Vì sao summarization giống dịch máy về kiến trúc?
