# Ôn NLP – Olympic AI Việt Nam (lộ trình 3 ngày)

## 1. Nên ưu tiên dạng nào?

Tiêu chí chọn: (a) **chấm điểm tự động dễ** (BTC hay ra kiểu này), (b) **train được trong vài giờ trên 1 GPU**, (c) có dữ liệu tiếng Việt phổ biến.
Số lượng model trên HuggingFace **không** phản ánh khả năng ra đề (Text Generation nhiều model nhưng khó chấm điểm).

| Mức | Dạng | Khả năng ra đề | Notebook |
|---|---|---|---|
| 🔥 Chắc chắn có | **Translation – dịch máy Anh ↔ Việt** | Rất cao | `05_machine_translation.ipynb` |
| ⭐⭐⭐ Bắt buộc | **Text Classification** (sentiment, chủ đề, toxic, intent, NLI) + Zero-shot | Rất cao | `01_text_classification.ipynb` |
| ⭐⭐⭐ Bắt buộc | **Token Classification** (NER, POS) | Cao | `02_token_classification_ner.ipynb` |
| ⭐⭐⭐ Bắt buộc | **Sentence Similarity + Text Ranking + Feature Extraction** (STS, tìm kiếm, RAG) | Cao, đang là xu hướng | `03_sentence_similarity_ranking.ipynb` |
| ⭐⭐ Nên | **Question Answering** (extractive, kiểu UIT-ViQuAD) | Trung bình–cao | `04_question_answering.ipynb` |
| ⭐ Biết làm | **Summarization** | Trung bình (tốn GPU, chấm ROUGE) | `06_seq2seq_summarization.ipynb` |
| Chỉ cần hiểu khái niệm | Fill-Mask, Text Generation, Table QA | Thấp (làm bài thi) | — |

- **Fill-Mask** = cách BERT được pretrain (MLM). Dùng được để *domain-adaptive pretraining* trước khi phân loại — chỉ làm khi thừa thời gian.
- **Text Generation**: hiểu decoder-only, sampling (greedy/beam/top-k/top-p/temperature), prompting, RAG. Nếu đề yêu cầu LLM, thường là RAG → kết hợp notebook 03.
- **Table QA**: rất hiếm, bỏ qua.

> Tôi không có đề các năm trước của cuộc thi. Hãy xem lại đề cũ / thể lệ (có internet khi thi không? GPU gì? giới hạn thời gian?) để điều chỉnh: nếu **không có internet**, tải sẵn model và dataset vào máy từ trước (`snapshot_download`).

## 2. Cách dùng tài liệu

**Notebook (`*.ipynb`)** = code thực hành để thi:
- Ô **Cấu hình** ở đầu: điền `TRAIN_FILE`, `TEST_FILE`… của đề; để `None` thì chạy dữ liệu mẫu.
- Đi theo thứ tự: baseline → model mạnh → nộp bài. Mục **Nâng cao** ở cuối chỉ làm khi còn thời gian.

**Kiến thức (`knowledge/*.md`)** = đọc để hiểu nguyên lý, code tự viết bằng PyTorch thuần + bảng đối chiếu thư viện:

| File | Nội dung |
|---|---|
| `00_nen_tang_transformer.md` | **đọc đầu tiên**: tokenizer BPE, attention, Transformer, Fill-Mask, Text Generation, vector câu |
| `05_dich_may.md` | encoder–decoder, teacher forcing, greedy/beam search, BLEU, MBR, back-translation |
| `01_phan_loai_van_ban.md` | bag-of-words, classifier, class weights, macro-F1, multi-label, zero-shot |
| `02_ner.md` | BIO, gán nhãn sub-token, F1 theo thực thể, CRF |
| `03_tuong_dong_va_xep_hang.md` | bi/cross-encoder, contrastive loss, BM25, RRF, MRR, Spearman |
| `04_hoi_dap.md` | dự đoán start/end, chọn đoạn, sliding window, EM/F1 |
| `06_tom_tat.md` | TextRank, chặn lặp n-gram, ROUGE |

Nên đọc file kiến thức **trước** rồi mới chạy notebook cùng dạng.

## 3. Lộ trình 3 ngày

Dịch máy được dồn nhiều giờ nhất vì gần như chắc chắn có trong đề.

### Ngày 1 — Dịch máy (ưu tiên số 1) + phân loại văn bản
| Giờ | Việc |
|---|---|
| Sáng (4h) | Chạy hết `05`. Nắm: làm sạch cặp câu, so sánh zero-shot các model (Marian / EnViT5 / NLLB), mã ngôn ngữ và `forced_bos_token_id`, `Seq2SeqTrainer`, label smoothing, BLEU / chrF++ / TER. Thử cả hai chiều en→vi và vi→en. |
| Chiều (3h) | Chạy `01`. Hiểu: chuẩn hoá NFC, TF-IDF word+char, StratifiedKFold, OOF, fine-tune, class weights, macro-F1. |
| Tối (2h) | **Tự gõ lại từ đầu** pipeline dịch máy (load model → dịch dev → tính BLEU → fine-tune) không nhìn code, giới hạn 60 phút. |

### Ngày 2 — Dịch máy nâng cao + NER + tương đồng câu
| Giờ | Việc |
|---|---|
| Sáng (3h) | `05` mục 5–8: dò tham số sinh, **MBR decoding**, ensemble qua MBR, back-translation, tách câu cho văn bản dài. Ghi lại mỗi bước tăng bao nhiêu BLEU. |
| Chiều (3h) | Chạy `02` (BIO, căn nhãn sub-token với `-100`, seqeval) và `03` phần A (bi-encoder vs cross-encoder, Spearman). |
| Tối (2h) | `03` phần B (BM25, RRF, rerank) + lướt `04` (offset_mapping, stride, EM/F1). |

### Ngày 3 — Thi thử + chốt template
| Giờ | Việc |
|---|---|
| Sáng (3h) | **Thi thử dịch máy bấm giờ** trên bộ dữ liệu khác (vd IWSLT'15 en-vi hoặc một phần PhoMT): 30 phút nộp bản zero-shot → fine-tune → MBR → nộp. |
| Chiều (3h) | Thi thử phân loại văn bản. Sửa template cho nhanh: hàm đọc dữ liệu, xuất submission đúng format. |
| Tối (2h) | Tải sẵn model (nếu thi offline: opus-mt-en-vi, opus-mt-vi-en, envit5-translation, nllb-200-distilled-600M, xlm-roberta…), ôn checklist, ngủ đủ. |

## 4. Checklist ngày thi (đạt độ chính xác cao)
1. Đọc kỹ **metric** và **format submission** trước khi code.
2. EDA 5 phút: số mẫu, phân phối nhãn, độ dài văn bản, ngôn ngữ, có lặp/rò rỉ train–test không.
3. **Nộp baseline sớm** (TF-IDF/BM25/model pretrained) để chắc có điểm.
4. Chia validation chuẩn (StratifiedKFold / GroupKFold nếu có nhóm). Chỉ tin điểm validation.
5. Fine-tune transformer phù hợp ngôn ngữ; lr 1e-5–5e-5, warmup 10%, 3–5 epoch, early stopping.
6. Tăng điểm theo thứ tự hiệu quả/thời gian: model tốt hơn (large) → K-fold → ensemble model khác loại → chỉnh ngưỡng/hậu xử lý → pseudo-label.
7. Cố định seed, lưu checkpoint, lưu OOF và dự đoán test để ensemble lại sau.

## 5. Lý thuyết hay bị hỏi
- Tokenization: word / BPE / WordPiece / SentencePiece; vì sao PhoBERT cần tách từ tiếng Việt.
- Self-attention, multi-head, positional encoding; encoder (BERT) vs decoder (GPT) vs encoder-decoder (T5/BART).
- Pretraining: MLM (Fill-Mask), CLM (Text Generation), span corruption (T5).
- Fine-tuning vs feature extraction vs zero-shot (NLI-based) vs few-shot prompting; LoRA/PEFT.
- Metrics: accuracy, precision/recall, macro vs micro F1, entity-level F1, EM/F1 cho QA, BLEU, ROUGE, Spearman, MRR, NDCG.
- Xử lý mất cân bằng, overfitting (dropout, weight decay, early stopping), data augmentation cho văn bản.
- Decoding: greedy, beam search, top-k, top-p, temperature, length penalty, repetition.

## 6. Cài đặt
```bash
pip install -U torch transformers datasets accelerate scikit-learn seqeval sentence-transformers rank_bm25 sacrebleu sentencepiece pyvi scipy
```
