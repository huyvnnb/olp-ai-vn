# 03 · Độ tương đồng câu, Feature Extraction, Xếp hạng văn bản

> Code ở đây dùng lại các lớp trong file `00_nen_tang_transformer.md`.

---

## 1. Ba bài toán, một ý tưởng

| Bài | Đầu vào | Đầu ra |
|---|---|---|
| Sentence Similarity | 2 câu | điểm giống nhau |
| Feature Extraction | 1 câu | 1 vector |
| Text Ranking / Retrieval | câu hỏi + nhiều tài liệu | tài liệu xếp theo độ liên quan |

Ý tưởng chung: biến câu thành **vector** sao cho câu **cùng nghĩa → vector gần nhau**.

```
"tôi thích mèo"        ●
"tôi yêu mèo"          ●    ← gần nhau
                                         ● "giá xăng tăng"   ← xa
```

---

## 2. Bi-encoder và Cross-encoder

```
BI-ENCODER                                 CROSS-ENCODER
câu A → Encoder → vector A ┐               [câu A + câu B] → Encoder → Linear → điểm
câu B → Encoder → vector B ┴→ cosine
```

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Cách làm | mã hoá từng câu riêng | đọc 2 câu cùng lúc |
| Tốc độ | **nhanh**: mã hoá tài liệu 1 lần, lưu lại | chậm: phải chạy lại cho mỗi cặp |
| Độ chính xác | khá | **cao hơn** (2 câu nhìn thấy nhau qua attention) |
| Dùng để | tìm nhanh top 100 trong hàng triệu tài liệu | xếp hạng lại top 100 đó |

---

## 3. Cosine similarity

```
cosine(a, b) = (a · b) / (|a| × |b|)        ∈ [-1, 1]
```

Nếu chuẩn hoá vector về độ dài 1 trước → cosine = tích vô hướng.

```python
def cosine(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


print(cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])))   # ≈ 0.707
```

---

## 4. Train bi-encoder: Contrastive learning

Dữ liệu: các cặp (câu, câu cùng nghĩa). Không cần cặp "khác nghĩa" – lấy luôn **các câu khác trong batch** làm ví dụ sai (**in-batch negatives**).

```
             đ.1    đ.2    đ.3    đ.4         ma trận cosine giữa câu hỏi và đáp án
câu 1      [ ✔      ✘      ✘      ✘  ]
câu 2      [ ✘      ✔      ✘      ✘  ]        mục tiêu: đường chéo cao nhất
câu 3      [ ✘      ✘      ✔      ✘  ]        → đúng là bài toán phân loại,
câu 4      [ ✘      ✘      ✘      ✔  ]          nhãn của hàng i là cột i
```

Loss này tên **InfoNCE**, trong sentence-transformers gọi là **MultipleNegativesRankingLoss**.

```python
def in_batch_contrastive_loss(query_vectors, doc_vectors, temperature=0.05):
    query_vectors = F.normalize(query_vectors, dim=-1)
    doc_vectors = F.normalize(doc_vectors, dim=-1)
    similarity = query_vectors @ doc_vectors.T / temperature      # (batch, batch)
    labels = torch.arange(len(query_vectors))                     # đáp án đúng nằm trên đường chéo
    return F.cross_entropy(similarity, labels)
```

Dữ liệu đồ chơi: câu và câu thay từ đồng nghĩa.

```python
synonyms = {"đẹp": "xinh", "nhanh": "lẹ", "to": "lớn", "buồn": "rầu", "vui": "mừng", "ăn": "xơi"}
subjects = ["cô ấy", "con mèo", "chiếc xe", "anh ta", "em bé"]


def make_paraphrase():
    subject = random.choice(subjects)
    word = random.choice(list(synonyms))
    return f"{subject} rất {word}", f"{subject} rất {synonyms[word]}"


sim_vocab = {"<pad>": 0}
for _ in range(500):
    for sentence in make_paraphrase():
        for w in sentence.split():
            if w not in sim_vocab:
                sim_vocab[w] = len(sim_vocab)


def encode_sentences(sentences):
    ids = [[sim_vocab.get(w, 0) for w in s.split()] for s in sentences]
    max_len = max(len(x) for x in ids)
    return torch.tensor([x + [0] * (max_len - len(x)) for x in ids])
```

Model = Encoder + mean pooling (giống file 00):

```python
class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model=d_model, num_heads=4, num_layers=2, d_ff=128)

    def forward(self, ids):
        return mean_pooling(self.encoder(ids), ids)


sentence_model = SentenceEncoder(len(sim_vocab))
optimizer = torch.optim.AdamW(sentence_model.parameters(), lr=1e-3)

for step in range(200):
    pairs = [make_paraphrase() for _ in range(16)]
    anchors = encode_sentences([a for a, b in pairs])
    positives = encode_sentences([b for a, b in pairs])

    loss = in_batch_contrastive_loss(sentence_model(anchors), sentence_model(positives))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Hard negatives:** thêm câu "trông giống nhưng sai" (vd tài liệu BM25 xếp cao nhưng không phải đáp án) → model học phân biệt tinh hơn.

**CoSENT loss:** khi dữ liệu có **điểm** (0–5) thay vì cặp đúng/sai → loss bắt cặp có điểm cao phải có cosine cao hơn cặp có điểm thấp.

---

## 5. BM25: tìm theo từ khoá (không cần train)

Điểm của tài liệu cho câu hỏi = tổng điểm từng từ trong câu hỏi:

```
điểm(từ) = IDF(từ) × tf × (k1 + 1) / (tf + k1 × (1 - b + b × độ_dài_tài_liệu / độ_dài_trung_bình))
```

- **tf**: từ xuất hiện bao nhiêu lần trong tài liệu (càng nhiều càng tốt, nhưng **bão hoà** nhờ k1).
- **IDF**: từ hiếm có giá trị hơn từ phổ biến.
- **b**: phạt tài liệu dài.

```python
def bm25_scores(query, documents, k1=1.5, b=0.75):
    docs = [d.lower().split() for d in documents]
    avg_len = sum(len(d) for d in docs) / len(docs)
    n_docs = len(docs)

    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for word in query.lower().split():
            n_containing = sum(1 for d in docs if word in d)
            idf = math.log((n_docs - n_containing + 0.5) / (n_containing + 0.5) + 1)
            tf = counts[word]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)
    return scores


docs = ["mèo ăn cá", "chó ăn xương", "mèo ngủ trên ghế"]
print(bm25_scores("mèo ngủ", docs))
```

| BM25 | Dense (bi-encoder) |
|---|---|
| giỏi khớp **chính xác** tên riêng, mã số, thuật ngữ | giỏi khớp **nghĩa** ("ô tô" ↔ "xe hơi") |
| không cần train | cần model tốt |

→ Kết hợp cả hai thường tốt nhất.

---

## 6. Gộp kết quả: RRF

Điểm BM25 và cosine khác thang đo → không cộng trực tiếp được. **RRF** chỉ dùng **thứ hạng**:

```
RRF(tài liệu) = Σ  1 / (60 + thứ hạng trong từng danh sách)
```

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = Counter()
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1 / (k + rank)
    return [doc_id for doc_id, s in scores.most_common()]


print(reciprocal_rank_fusion([["A", "B", "C"], ["B", "C", "A"]]))   # B đứng đầu
```

---

## 7. Chấm điểm

| Metric | Ý nghĩa |
|---|---|
| **Recall@k** | đáp án đúng có nằm trong top k không |
| **MRR** | trung bình của 1 / (thứ hạng đáp án đúng đầu tiên) |
| **NDCG@k** | như MRR nhưng tính được nhiều tài liệu đúng, mức độ đúng khác nhau |
| **Spearman** | (cho bài chấm điểm cặp câu) thứ tự điểm model có giống thứ tự điểm chuẩn không |

```python
def mrr(rank_of_correct):
    return sum(1 / r for r in rank_of_correct) / len(rank_of_correct)


def spearman(x, y):
    def to_rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for rank, i in enumerate(order):
            ranks[i] = rank
        return ranks

    rx = to_rank(x)
    ry = to_rank(y)
    n = len(x)
    d_squared = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d_squared / (n * (n ** 2 - 1))


print(mrr([1, 3, 2]))                          # (1 + 1/3 + 1/2) / 3
print(spearman([0.1, 0.5, 0.9], [1, 2, 5]))     # 1.0 – cùng thứ tự
```

---

## 8. Thư viện có sẵn

| Tự viết | Thư viện |
|---|---|
| `SentenceEncoder` | `SentenceTransformer("...")`, `model.encode(...)` |
| `in_batch_contrastive_loss` | `losses.MultipleNegativesRankingLoss` |
| loss theo điểm | `losses.CoSENTLoss`, `losses.CosineSimilarityLoss` |
| cross-encoder | `CrossEncoder("...")`, `.predict(pairs)` |
| `bm25_scores` | `rank_bm25.BM25Okapi` |
| tìm kiếm trên hàng triệu vector | `faiss` |
| `spearman` | `scipy.stats.spearmanr` |

```py
from sentence_transformers import SentenceTransformer, CrossEncoder

bi = SentenceTransformer("intfloat/multilingual-e5-base")
vectors = bi.encode(["query: mèo ăn gì", "passage: mèo ăn cá"], normalize_embeddings=True)

cross = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
cross.predict([("mèo ăn gì", "mèo ăn cá"), ("mèo ăn gì", "giá xăng tăng")])
```

**RAG** (Retrieval-Augmented Generation) = bước tìm kiếm ở file này + LLM đọc top tài liệu rồi viết câu trả lời.

---

## 9. Tự kiểm tra

1. Vì sao không dùng cross-encoder để tìm trong 1 triệu tài liệu?
2. In-batch negatives là gì? Vì sao batch size lớn thường giúp model tốt hơn?
3. Khi nào BM25 thắng dense model?
4. Vì sao RRF dùng thứ hạng chứ không dùng điểm số?
