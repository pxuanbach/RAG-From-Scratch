# 02. Embedding và vector search

Bài học: [Embedding và vector search: chọn model bằng bài test 10 câu hỏi](https://www.util-kits.io.vn/vi/blog/2026-10-22-embeddings-vector-search-explained-rag) · [English](README.md)

Đây là nơi package `ragkit/` bắt đầu. Nó gồm interface `Embedder` với bản cài đặt local dùng sentence-transformers, cache vector trên đĩa, và một smoke test 10 câu hỏi (`eval/smoke.jsonl`) đo xem bài báo đúng có nằm trong top 3 không. `compare_models.py` chạy test này cho `bge-m3` và `multilingual-e5-small`, với câu hỏi tiếng Anh và tiếng Việt trên corpus tiếng Anh. Bài viết giải thích kết quả, trang này chỉ hướng dẫn cách chạy.

| File | Nội dung |
|---|---|
| `ragkit/corpus.py` | Tải bài báo và câu hỏi của MultiHop-RAG từ Hugging Face |
| `ragkit/embed.py` | Interface `Embedder`, `LocalEmbedder`, `OpenAIEmbedder` (chưa test), cache vector trên đĩa |
| `ragkit/search.py` | Top-k vét cạn bằng một phép nhân ma trận |
| `ragkit/smoke.py` | Đọc bộ smoke, tính hit@3 và hạng của bài báo bằng chứng tốt nhất |
| `eval/smoke.jsonl` | 10 câu hỏi: bản tiếng Anh, bản dịch tiếng Việt, URL các bài báo đúng. Bài 03 dùng lại |
| `eval/build_smoke.py` | Cách chọn 10 câu hỏi (để tham khảo, không cần chạy lại) |
| `intuition.py` | Cosine, dot product và L2 trên bốn câu thật |
| `compare_models.py` | Embed cả 609 bài cho mỗi cấu hình rồi chạy smoke test |
| `results/` | `intuition.json` và `compare.json`: kết quả tôi chạy trên RTX 5060 Ti, 2026-10-14 |

## Chuẩn bị

1. Cài thư viện từ thư mục gốc của repo, theo [README gốc](../README.vi.md#cài-đặt). Bài này không cần LLM.
2. Có GPU thì tốt. Trên CPU, `compare_models.py` chậm hơn nhiều lần (lượt `bge-m3` 2048 token là lượt chậm nhất).

Lần chạy đầu tải `bge-m3` (khoảng 2.3 GB) và `multilingual-e5-small` (khoảng 0.5 GB). Embedding được cache ở `../.cache/embeddings/`, nên lần chạy thứ hai chỉ mất vài giây.

## Chạy

```bash
cd 02-embeddings
uv run python intuition.py
uv run python compare_models.py
```

Kết quả mong đợi của `intuition.py` (điểm khớp tới ba chữ số thập phân trên máy tôi):

```text
dimension: 1024, vector norm: 1.000

cosine similarity (A/B: same trial, different words; C: football; D: Vietnamese version of A)
  A vs B: 0.595
  A vs C: 0.255
  A vs D: 0.867
  B vs C: 0.305
  B vs D: 0.607
  C vs D: 0.308

normalized vectors, A vs C: cosine 0.255, dot 0.255, L2 1.220, 2 - 2*cosine = 1.489 = L2 squared 1.489
scale A by 3: dot 0.766 (changed), cosine 0.255 (same)
```

Kết quả mong đợi của `compare_models.py` (mỗi dòng thật còn kết thúc bằng thời gian embed tính theo giây, bằng 0.0 khi vector lấy từ cache; `results/compare.json` giữ thời gian của lần chạy đầu):

```text
bge-m3, 2048 tokens                      dim 1024  truncated 275/609  EN hit@3 1.0  VI hit@3 1.0  both@3 0.4/0.4  mean rank 1.3/1.1
bge-m3, 512 tokens                       dim 1024  truncated 609/609  EN hit@3 0.9  VI hit@3 0.9  both@3 0.4/0.4  mean rank 1.9/2.1
multilingual-e5-small, with prefixes     dim  384  truncated 609/609  EN hit@3 0.9  VI hit@3 0.6  both@3 0.3/0.3  mean rank 1.6/5.3
multilingual-e5-small, no prefixes       dim  384  truncated 609/609  EN hit@3 0.9  VI hit@3 0.6  both@3 0.3/0.3  mean rank 2.0/4.2
```

"hit@3" tính một câu là đúng khi có ít nhất một bài báo đúng nằm trong top 3. "both@3" đòi hỏi đủ mọi bài báo đúng, còn "mean rank" là vị trí trung bình của bài đúng xếp cao nhất trong 609 bài.

## Thử model của bạn

Thêm một dòng vào `CONFIGS` trong `compare_models.py`. Đọc model card trước để biết prefix và độ dài input tối đa:

```python
("my model", dict(model_name="org/model-name", query_prefix="query: ", doc_prefix="passage: ", max_seq_length=512)),
```

Bạn cũng có thể thêm câu hỏi của mình vào `eval/smoke.jsonl` (cùng các trường: `id`, `q_en`, `q_vi`, `expected_urls`).

`OpenAIEmbedder` trong `ragkit/embed.py` minh họa phương án dùng API. Tôi đã viết nhưng chưa chạy lần nào vì không có API key. Cài `openai`, đặt `OPENAI_API_KEY`, và coi nó là điểm xuất phát.

## Xử lý lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| CUDA out of memory trong `compare_models.py` | Giảm `batch_size` của cấu hình `bge-m3` 2048 token, hoặc đóng chương trình khác đang dùng GPU |
| Tiếng Việt bị lỗi font trên terminal Windows | Đặt `PYTHONIOENCODING=utf-8` |
| Số liệu lệch một chút so với bài viết | Phiên bản GPU và thư viện làm đổi chữ số cuối. Hạng và số câu trúng phải khớp |
| Kết quả cũ sau khi đổi cấu hình | Xóa `../.cache/embeddings/`. Khóa cache gồm model, độ dài tối đa và prefix, không gồm các thiết lập khác |
