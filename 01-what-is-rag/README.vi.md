# 01. RAG là gì? Tự xây RAG đầu tiên trong khoảng 50 dòng

Bài học: [RAG là gì? Tự xây RAG đầu tiên trong khoảng 50 dòng Python](https://www.util-kits.io.vn/vi/blog/2026-10-15-what-is-rag-build-first-rag-python) · [English](README.md)

`rag_50_lines.py` tải 20 bài báo từ MultiHop-RAG, giữ 3 đoạn đầu của mỗi bài rồi embed 60 đoạn đó bằng `BAAI/bge-m3`. Khi bạn hỏi, script tìm 3 đoạn gần câu hỏi nhất và gửi cho LLM để trả lời kèm trích dẫn `[n]`. Bài viết giải thích từng bước, trang này chỉ hướng dẫn cách chạy.

| File | Nội dung |
|---|---|
| `rag_50_lines.py` | Pipeline: load, embed, retrieve, generate |
| `experiment.py` | Hỏi cùng một câu theo 3 cách (không tài liệu, dán tài liệu vào prompt, RAG), ghi số token và độ trễ vào `results.json` |
| `results.json` | Kết quả mình chạy bằng Qwen2.5-7B-Instruct-1M trên LM Studio, card RTX 5060 Ti |

## Chuẩn bị

1. Cài thư viện từ thư mục gốc của repo, theo [README gốc](../README.vi.md#cài-đặt).
2. Chuẩn bị một LLM: bật server LM Studio với một model đã nạp, hoặc có Anthropic API key.

Lần chạy đầu sẽ tải corpus (12 MB) và model `bge-m3` (khoảng 2.3 GB). CPU embed 60 đoạn mất chưa tới một phút.

## Chạy

Với LM Studio:

```bash
cd 01-what-is-rag
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m   # id model hiển thị trong LM Studio
uv run python rag_50_lines.py
```

Với Claude, chỉ cần đặt `ANTHROPIC_API_KEY` bằng key của bạn. `RAG_MODEL` mặc định là `claude-haiku-5-5`.

Trên PowerShell, đặt biến bằng `$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:1234"`, các biến khác tương tự.

Kết quả mong đợi (câu chữ sẽ khác, điểm số phải giống):

```text
0.649  TechCrunch: SBF’s trial starts soon, but how did he — and FTX — get here?
0.556  TechCrunch: SBF’s trial starts soon, but how did he — and FTX — get here?
0.454  TechCrunch: SBF’s trial starts soon, but how did he — and FTX — get here?

FTX was founded in 2019 by Sam Bankman-Fried, and at its peak, its valuation hit $32 billion [1].
```

## Hỏi câu của bạn

Truyền câu hỏi làm tham số:

```bash
uv run python rag_50_lines.py "Who won the 2023 Nobel Prize in Physics?"
```

Không bài nào trong 20 bài nói về giải Nobel, nên model sẽ trả lời là không biết. Thử thêm hai câu:

- `"How did the Houston Astros cheat in 2017?"` tìm đúng bài, nhưng model vẫn nói không biết. Script chỉ giữ 3 đoạn đầu của mỗi bài, mà 3 đoạn đó không kể cách gian lận. Bài 03 sửa chỗ này bằng chunking thật.
- Câu hỏi về một bài nằm ngoài 20 bài đầu sẽ không tìm được gì hữu ích. Đổi `[:20]` trong script để nạp nhiều bài hơn.

Hỏi bằng tiếng Việt cũng được, vì `bge-m3` là model đa ngôn ngữ:

```bash
uv run python rag_50_lines.py "FTX được thành lập năm nào và lúc đỉnh cao được định giá bao nhiêu?"
```

Lần mình chạy, script vẫn tìm đúng bài TechCrunch (điểm 0.643) và model trả lời bằng tiếng Việt. Bài 02 đo xem tìm kiếm chéo ngôn ngữ chính xác đến đâu trên nhiều câu hỏi.

## Chạy lại thí nghiệm

```bash
uv run python experiment.py --retrieval-only   # không cần LLM, in ra top 3
uv run python experiment.py                    # chạy đủ, cần LLM
```

Cả hai lệnh đều ghi đè `results.json`. Với model local, bước dán cả 20 bài vào prompt sẽ lỗi vì vượt context, và script ghi lại lỗi đó. Muốn bước này chạy được thì tăng context length trong LM Studio.

## Gặp lỗi

| Triệu chứng | Cách sửa |
|---|---|
| `Connection error` | Bật server LM Studio, kiểm tra cổng 1234 |
| Không tìm thấy model | Đặt `RAG_MODEL` bằng một id lấy từ `curl http://127.0.0.1:1234/v1/models` |
| Ký tự `’` hoặc `—` bị vỡ khi pipe output trên Windows | Đặt `PYTHONIOENCODING=utf-8` |
