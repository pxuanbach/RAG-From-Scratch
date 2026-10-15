# RAG From Scratch

Code mẫu cho series RAG From Scratch trên [UtilKits](https://www.util-kits.io.vn/vi/blog). Series có 12 bài, bắt đầu từ một script 50 dòng và kết thúc bằng một RAG service có thể deploy. Mỗi thư mục chứa code của một bài và một README hướng dẫn cách chạy. Phần giải thích nằm trong bài viết, còn số liệu trong bài đều đo từ chính code này.

[English](README.md)

## Danh sách bài

| # | Bài | Level | Trạng thái |
|---|---|---|---|
| 01 | [RAG là gì? Tự xây RAG đầu tiên trong khoảng 50 dòng](01-what-is-rag/README.vi.md) | Beginner | đã có |
| 02 | [Embeddings và vector search](02-embeddings/README.vi.md) | Beginner | đã có |
| 03 | [Ingestion và chunking](03-chunking/README.vi.md) | Beginner | đã có |
| 04 | [Lưu vector với Postgres + pgvector](04-pgvector) | Beginner | sắp ra |
| 05 | [Câu trả lời grounded có trích nguồn](05-grounded-generation) | Beginner | sắp ra |
| 06 | [Đo lường RAG](06-evaluation) | Intermediate | sắp ra |
| 07 | [Hybrid search và reranking](07-hybrid-search-reranking) | Intermediate | sắp ra |
| 08 | [Biến đổi query](08-query-transformation) | Intermediate | sắp ra |
| 09 | [Contextual retrieval và parent-document](09-smarter-indexing) | Intermediate | sắp ra |
| 10 | [Agentic RAG](10-agentic-rag) | Advanced | sắp ra |
| 11 | [Bảo mật RAG](11-rag-security) | Advanced | sắp ra |
| 12 | [Ship it: đưa RAG thành service](12-ship-it) | Advanced | sắp ra |

Bài 01 đến 05 dựng pipeline trả lời được câu hỏi. Bài 06 đến 09 đo pipeline và sửa những chỗ số đo chỉ ra, còn 10 đến 12 thêm agent, bảo mật và deploy.

## Cài đặt

Cần Python 3.10+ và [uv](https://docs.astral.sh/uv/) (hoặc pip).

```bash
git clone https://github.com/pxuanbach/RAG-From-Scratch.git
cd RAG-From-Scratch
uv venv && uv pip install -r requirements.txt
```

Lần chạy đầu sẽ tải model embedding `BAAI/bge-m3`, khoảng 2.3 GB.

## Chạy bằng model self-host hoặc bằng Claude

Code gọi LLM qua SDK `anthropic`. [LM Studio](https://lmstudio.ai/docs/developer/anthropic-compat) có endpoint tương thích Anthropic, nên cùng một script chạy được trên Claude lẫn model self-host.

Chép file mẫu thành `.env` ở thư mục gốc của repo:

```bash
cp .env.template .env
```

File mẫu trỏ tới LM Studio đang chạy `qwen2.5-7b-instruct-1m`. Muốn dùng Claude, comment ba dòng đó lại rồi điền `ANTHROPIC_API_KEY` và `RAG_MODEL` ở phương án B. Mọi script đều nạp `.env` bằng `python-dotenv`, nên chúng ta chỉ cấu hình một lần cho cả series. `.env` đã nằm trong `.gitignore`.

Rồi chạy bài trong thư mục của bài đó. README của từng bài ghi lệnh chạy và kết quả mong đợi.

```bash
cd 01-what-is-rag
uv run python rag_50_lines.py
```

Biến đã đặt sẵn trong shell sẽ được ưu tiên hơn giá trị trong `.env`.

## Dữ liệu

Series dùng [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024): 609 bài báo tiếng Anh cuối năm 2023 và 2,556 câu hỏi kèm đáp án và bằng chứng. Script tự tải dataset từ Hugging Face ở lần chạy đầu, repo này không phân phối lại dữ liệu. Dataset dùng giấy phép ODC-BY 1.0, nội dung bài báo vẫn thuộc về các nhà xuất bản.

## Giấy phép

Code phát hành theo [giấy phép MIT](LICENSE). Dataset MultiHop-RAG giữ giấy phép ODC-BY 1.0 riêng của nó.
