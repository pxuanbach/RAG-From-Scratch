# 03. Ingestion và chunking

Bài học: [Chunking trong RAG bằng Python: làm sạch và cắt bài báo thật, rồi đo hiệu quả](https://www.util-kits.io.vn/vi/blog/2026-10-29-rag-chunking-strategies-python) · [English](README.md)

Thư mục này mở rộng `ragkit/` của bài 02 với bước làm sạch (`corpus.py`) và bốn chiến lược chunk (`chunk.py`). `compare_chunking.py` chia 609 bài đã làm sạch theo bảy cách, embed mọi chunk bằng `bge-m3`, rồi chấm truy xuất trên 150 câu hỏi lấy mẫu cùng bộ smoke 10 câu. Không cần LLM, trừ `astros.py --ask` (tùy chọn).

| File | Nội dung |
|---|---|
| `ragkit/corpus.py` | Hàm đọc dữ liệu của bài 02, thêm `find_boilerplate`, `clean_body`, `load_clean_articles` |
| `ragkit/chunk.py` | `Chunk`, các chiến lược (`fixed`, `recursive`, `by_paragraph`), id băm từ nội dung, `make_chunks` |
| `ragkit/smoke.py` | Smoke test của bài 02, thêm `run_chunk_eval` (hit, all, lượng bằng chứng tìm thấy, theo top-k hoặc ngân sách token) |
| `ragkit/embed.py`, `ragkit/search.py` | Giữ nguyên từ bài 02 |
| `eval/sample150.jsonl` | 150 câu hỏi (mỗi loại 50) kèm câu bằng chứng; `eval/build_sample.py` tạo ra file này |
| `eval/smoke.jsonl` | 10 câu smoke từ bài 02 |
| `junk_stats.py` | Đếm rác trong cả 609 bài, làm sạch, kiểm tra không mất câu bằng chứng nào |
| `compare_chunking.py` | Chunk, embed và chấm từng chiến lược, có và không có dòng metadata |
| `analyze.py` | Khoảng tin cậy bootstrap theo cặp: chênh lệch nào vượt nhiễu |
| `junk_in_topk.py` | Rác lọt vào top 5 bao nhiêu khi bỏ qua bước làm sạch |
| `chunk_ids.py` | Bao nhiêu id chunk còn nguyên sau một chỉnh sửa nhỏ, mã băm nội dung so với vị trí |
| `astros.py` | Câu hỏi Astros của bài 01, chạy trên cả 609 bài |
| `results/` | `junk_stats`, `chunking`, `significance`, `junk_in_topk`, `chunk_ids`, `astros` (JSON): kết quả tôi chạy trên RTX 5060 Ti, 2026-10-14 |

## Chuẩn bị

1. Cài thư viện từ thư mục gốc của repo, theo [README gốc](../README.vi.md#cài-đặt).
2. Có GPU thì tốt. `compare_chunking.py` embed tổng cộng khoảng 70.000 chunk (lần đầu khoảng 25 phút trên GPU của tôi, chạy lại với cache khoảng 4 phút).
3. Chỉ `astros.py --ask` cần LLM. Đặt các biến LM Studio hoặc Claude trong README gốc.

Embedding được cache ở `../.cache/embeddings/` (file `BAAI__bge-m3_len1024_0p.npz`, khoảng 300 MB). Xóa file này để tính lại.

## Chạy

```bash
cd 03-chunking
uv run python junk_stats.py
uv run python eval/build_sample.py      # tùy chọn, sample150.jsonl đã có sẵn
uv run python compare_chunking.py
uv run python analyze.py
uv run python junk_in_topk.py
uv run python chunk_ids.py
uv run python astros.py --ask
```

Kết quả mong đợi của `junk_stats.py` (các dòng chính):

```text
author_null: 64
author_empty_string: 4
author_string_None: 0
replacement_char_U+FFFD: 0
articles_with_invisible_chars: 16
articles_with_any_junk_paragraph: 213
paragraphs_total: 29011
paragraphs_after_cleaning: 28107
tokens_before: 1548729
tokens_after: 1537038
evidence_facts_found_in_clean_body: 6084
```

Kết quả mong đợi của `compare_chunking.py` (mỗi dòng thật còn kết thúc bằng thời gian embed tính theo giây, bằng 0.0 khi lấy từ cache; `results/chunking.json` giữ thời gian của lần chạy đầu). `hit@3` tính một câu là đúng khi một trong 3 chunk đầu đến từ bài báo đúng; `fact@k` là tỉ lệ câu bằng chứng nằm nguyên vẹn trong k chunk đầu; `fact@2000t` dùng số chunk vừa 2.000 token; `intact` là tỉ lệ câu bằng chứng không bị ranh giới chunk cắt:

```text
whole article (2048-token cut)    chunks    609  tok 2543.7  intact 1.000  hit@3 0.867  all@5 0.340  fact@3 0.536  fact@5 0.654  fact@2000t 0.255
fixed 256                         chunks   6305  tok  243.9  intact 0.870  hit@3 0.833  all@5 0.267  fact@3 0.271  fact@5 0.357  fact@2000t 0.428
fixed 512                         chunks   3303  tok  465.4  intact 0.944  hit@3 0.840  all@5 0.300  fact@3 0.320  fact@5 0.440  fact@2000t 0.354
fixed 512 + overlap 64            chunks   3641  tok  475.5  intact 1.000  hit@3 0.833  all@5 0.240  fact@3 0.326  fact@5 0.404  fact@2000t 0.349
recursive 256                     chunks   7420  tok  207.1  intact 1.000  hit@3 0.827  all@5 0.253  fact@3 0.290  fact@5 0.384  fact@2000t 0.495
recursive 512                     chunks   3543  tok  433.8  intact 1.000  hit@3 0.807  all@5 0.273  fact@3 0.311  fact@5 0.399  fact@2000t 0.371
paragraph (min 128, max 512)      chunks   9180  tok  167.4  intact 1.000  hit@3 0.840  all@5 0.313  fact@3 0.265  fact@5 0.362  fact@2000t 0.537
```

Script còn in thêm một dòng cho mỗi chiến lược với biến thể `+ header`, một dòng recursive 512 trên văn bản chưa làm sạch, cùng mean rank và kết quả smoke test; các số ở trên là số được trích trong bài.

Kết quả mong đợi của `astros.py --ask` (các dòng chính):

```text
fixed 256                      first chunk with the scheme ranks    2   in top 3: True    in a 2000-token context (7 chunks): True
paragraph (min 128, max 512)   first chunk with the scheme ranks    4   in top 3: False   in a 2000-token context (12 chunks): True
```

Cuối cùng là câu trả lời của model, câu chữ đổi theo từng lần chạy.

## Thử thêm

- Đổi `min_tokens` hoặc `max_tokens` trong `by_paragraph` rồi chạy lại `compare_chunking.py`. Chỉ các chunk mới được embed.
- Thêm một chiến lược vào `STRATEGIES` trong `ragkit/chunk.py` (ví dụ cửa sổ theo câu) và xem nó đứng ở đâu.
- Chạy `analyze.py` cho biến thể của bạn trước khi tin vào chênh lệch 0,02.

## Xử lý lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| CUDA out of memory | Giảm `batch_size` trong `compare_chunking.py` (16 cho model chunk, 4 cho baseline cả bài) |
| Lỗi font trên Windows | Đặt `PYTHONIOENCODING=utf-8` |
| Số lệch ở chữ số cuối | Phiên bản GPU và thư viện làm vector lệch nhẹ. Số chunk và thống kê làm sạch phải khớp chính xác |
| `astros.py --ask` không kết nối được | Mở LM Studio và đặt các biến trong README gốc, hoặc chạy không có `--ask` |
