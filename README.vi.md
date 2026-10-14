# RAG From Scratch

Code đi kèm series **RAG From Scratch** trên [UtilKits](https://www.util-kits.io.vn/vi/blog). 12 bài thực hành đưa bạn từ một script 50 dòng đến một RAG service có thể đưa lên production. Bài nào cũng dựng được thứ chạy được, và mọi con số trong bài đều đo từ chính code này.

[English](README.md)

## Danh sách bài

| # | Bài | Level | Trạng thái |
|---|---|---|---|
| 01 | [RAG là gì? Tự xây RAG đầu tiên trong khoảng 50 dòng](01-what-is-rag/README.vi.md) | Beginner | đã có |
| 02 | [Embeddings và vector search](02-embeddings) | Beginner | sắp ra |
| 03 | [Ingestion và chunking](03-chunking) | Beginner | sắp ra |
| 04 | [Lưu vector với Postgres + pgvector](04-pgvector) | Beginner | sắp ra |
| 05 | [Câu trả lời grounded có trích nguồn](05-grounded-generation) | Beginner | sắp ra |
| 06 | [Đo lường RAG](06-evaluation) | Intermediate | sắp ra |
| 07 | [Hybrid search và reranking](07-hybrid-search-reranking) | Intermediate | sắp ra |
| 08 | [Biến đổi query](08-query-transformation) | Intermediate | sắp ra |
| 09 | [Contextual retrieval và parent-document](09-smarter-indexing) | Intermediate | sắp ra |
| 10 | [Agentic RAG](10-agentic-rag) | Advanced | sắp ra |
| 11 | [Bảo mật RAG](11-rag-security) | Advanced | sắp ra |
| 12 | [Ship it: đưa RAG thành service](12-ship-it) | Advanced | sắp ra |

Xong bài 01–05, bạn tự xây được một ứng dụng RAG chạy được. Xong 06–09, bạn đo và cải thiện được nó. Xong 10–12, bạn vận hành được nó trên production.

## Cài đặt

Cần Python 3.10+ và [uv](https://docs.astral.sh/uv/) (hoặc pip).

```bash
git clone https://github.com/pxuanbach/RAG-From-Scratch.git
cd RAG-From-Scratch
uv venv && uv pip install -r requirements.txt
```

Lần chạy đầu sẽ tải model embedding `BAAI/bge-m3`, khoảng 2.3 GB.

## Chạy bằng model local miễn phí hoặc bằng Claude

Code gọi LLM qua SDK `anthropic`, nên cùng một script chạy được trên Claude hoặc trên [LM Studio](https://lmstudio.ai/docs/developer/anthropic-compat) (LM Studio có endpoint tương thích Anthropic).

```bash
# Local, miễn phí (LM Studio đang chạy model như qwen2.5-7b-instruct-1m)
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m

# Hoặc Claude: chỉ cần key thật, RAG_MODEL mặc định là claude-haiku-5-5
export ANTHROPIC_API_KEY=sk-ant-...
```

Rồi chạy bài trong thư mục của bài đó:

```bash
cd 01-what-is-rag
uv run python rag_50_lines.py
```

Trên Windows PowerShell, dùng `$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:1234"` thay cho `export`.

## Dữ liệu

Series dùng [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024): 609 bài báo tiếng Anh cuối năm 2023 và 2,556 câu hỏi kèm đáp án và bằng chứng. Script tự tải dataset từ Hugging Face ở lần chạy đầu, repo này không phân phối lại dữ liệu. Dataset dùng giấy phép ODC-BY 1.0, nội dung bài báo vẫn thuộc về các nhà xuất bản.
