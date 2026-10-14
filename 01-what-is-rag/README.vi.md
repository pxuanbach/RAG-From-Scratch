# RAG là gì? Tự xây RAG đầu tiên với khoảng 50 dòng Python

Đọc trên blog: [https://www.util-kits.io.vn/vi/blog/2026-10-15-what-is-rag-build-first-rag-python](https://www.util-kits.io.vn/vi/blog/2026-10-15-what-is-rag-build-first-rag-python) · [English](README.md)

*Phần 1 của series RAG From Scratch. Bài này: RAG là gì, và một pipeline chạy được mà bạn đọc hiểu từng dòng.*

> **Level:** Beginner (bài 1/5 của chặng Beginner)
> **Dành cho:** lập trình viên biết Python cơ bản, đã dùng ChatGPT hoặc Claude nhưng chưa từng tự xây ứng dụng LLM. PM và founder có thể đọc phần khái niệm, bỏ qua phần code.
> **Cần chuẩn bị:** Python 3.10+, `uv` hoặc `pip`, và LM Studio (miễn phí, chạy local) hoặc một API key của Anthropic.
> **Sau bài này bạn sẽ:** giải thích được ba bước retrieve, augment, generate bằng lời của mình, chạy được một script RAG không cần framework, và đo được chi phí token của một câu hỏi.

Mình hỏi một model 7B chạy local một câu đơn giản: theo chính sách công tác của Acme Robotics, mức trần tiền khách sạn mỗi đêm ở Tokyo là bao nhiêu. Acme Robotics là công ty mình tự bịa ra cho bài test này, nên chưa model nào từng thấy chính sách của nó. Model trả lời rằng nó không có thông tin về chính sách nội bộ và khuyên mình liên hệ phòng nhân sự.

Sau đó mình dán một tài liệu chính sách dài bốn câu vào prompt rồi hỏi lại. Câu trả lời về sau một phần ba giây: "240 USD [1]".

Lần hỏi thứ hai chính là retrieval-augmented generation (RAG) ở dạng thu nhỏ. Series này sẽ xây phiên bản mà code tự tìm đúng tài liệu trong hàng nghìn tài liệu và dán vào giúp bạn.

## RAG làm gì

Một hệ thống [RAG](https://www.util-kits.io.vn/vi/glossary#rag) trả lời câu hỏi qua ba bước:

1. **Retrieve (truy xuất):** tìm trong kho tài liệu những đoạn liên quan nhất tới câu hỏi.
2. **Augment (bổ sung):** đặt các đoạn đó vào prompt, cạnh câu hỏi.
3. **Generate (sinh câu trả lời):** LLM viết câu trả lời dựa trên các đoạn được đưa vào.

Bước tìm kiếm dùng embedding: mỗi đoạn văn được biến thành một dãy số (vector), và các đoạn có nghĩa giống nhau sẽ có vector gần nhau. Bài 2 sẽ giải thích cơ chế này. Tạm thời bạn cứ coi `embed()` là một hàm biến văn bản thành vector để so sánh.

## RAG, fine-tuning hay prompt to hơn

Có ba cách phổ biến để cung cấp cho LLM kiến thức nó chưa có:

| Cách làm | Hợp với | Yếu ở |
|---|---|---|
| Fine-tuning | Dạy văn phong, định dạng hoặc một tác vụ | Dữ kiện thay đổi thường xuyên, trích dẫn nguồn |
| Đưa toàn bộ tài liệu vào prompt | Vài tài liệu ngắn | Chi phí và tốc độ khi tài liệu nhiều lên, giới hạn context |
| RAG | Kho tài liệu lớn hoặc hay thay đổi, câu trả lời cần trích nguồn | Câu trả lời phụ thuộc vào tài liệu mà bước tìm kiếm bỏ sót |

Fine-tuning định hình *cách* model trả lời. RAG cung cấp *nội dung* model cần biết ngay lúc được hỏi. Thí nghiệm bên dưới cho thấy cách "prompt to hơn" gãy ở đâu.

## Dữ liệu: 609 bài báo thật

Series dùng [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG), một dataset công khai được xây để kiểm tra hệ thống RAG. Dataset gồm 609 bài báo tiếng Anh đăng từ tháng 9 đến tháng 12 năm 2023, tổng cộng khoảng 1,25 triệu token, kèm 2.556 câu hỏi có sẵn đáp án và các bài báo làm bằng chứng. Các bài sau sẽ dùng bộ câu hỏi này để đo pipeline. Dataset dùng license ODC-BY 1.0.

Bài này giữ quy mô nhỏ: 20 bài báo đầu tiên, mỗi bài lấy ba đoạn, tổng cộng 60 đoạn.

Cài bốn package:

```bash
uv venv && uv pip install sentence-transformers huggingface_hub anthropic numpy
```

Lần chạy đầu sẽ tải model embedding `BAAI/bge-m3`, khoảng 2,3 GB. Có GPU thì nhanh hơn, nhưng CPU vẫn xử lý 60 đoạn văn thoải mái.

## Code

```python
# rag_50_lines.py
import json
import os
import sys

import anthropic
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# 1. Load 20 news articles and keep the first 3 real paragraphs of each
path = hf_hub_download("yixuantt/MultiHopRAG", "corpus.json", repo_type="dataset")
articles = json.load(open(path, encoding="utf-8"))[:20]
docs = []
for a in articles:
    paragraphs = [p.strip() for p in a["body"].split("\n\n") if len(p.strip()) > 80]
    for p in paragraphs[:3]:
        docs.append({"title": a["title"], "source": a["source"], "text": p})

# 2. Embed every paragraph once (normalized, so dot product = cosine similarity)
model = SentenceTransformer("BAAI/bge-m3")
doc_vecs = model.encode([d["text"] for d in docs], normalize_embeddings=True)


def retrieve(question, k=3):
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vecs @ q_vec
    top = np.argsort(-scores)[:k]
    return [(docs[i], float(scores[i])) for i in top]


SYSTEM = (
    "Answer using only the documents provided. Cite documents as [1], [2]. "
    "If the documents do not contain the answer, say you don't know."
)


def ask(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{n}] {d['title']} ({d['source']})\n{d['text']}" for n, (d, _) in enumerate(hits, 1)
    )
    # Reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL, so the same code runs on Claude or LM Studio
    response = anthropic.Anthropic().messages.create(
        model=os.environ.get("RAG_MODEL", "claude-haiku-5-5"),
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<documents>\n{context}\n</documents>\n\n{question}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text"), hits


if __name__ == "__main__":
    answer, hits = ask(" ".join(sys.argv[1:]) or "When was FTX founded and what was it worth at its peak?")
    for d, score in hits:
        print(f"{score:.3f}  {d['source']}: {d['title']}")
    print("\n" + answer)
```

Script ứng với ba bước như sau:

- **Retrieve** là hàm `retrieve()`. Hàm này embed câu hỏi, chấm điểm nó với cả 60 vector đoạn văn bằng một phép nhân ma trận, rồi giữ lại 3 đoạn cao điểm nhất. Vì vector đã được chuẩn hóa nên tích vô hướng chính là cosine similarity.
- **Augment** là chuỗi `context` trong hàm `ask()`. Mỗi đoạn được đánh số để model trích dẫn.
- **Generate** là lời gọi `messages.create`. System prompt yêu cầu model chỉ dùng thông tin trong tài liệu và nói rõ khi tài liệu không có câu trả lời.

Bộ lọc `len(p) > 80` là cách thô để bỏ qua các khung đăng ký newsletter và chú thích ảnh ngắn trong bài báo gốc. Bài 3 sẽ thay nó bằng bước làm sạch và chia chunk đúng nghĩa.

## Chạy miễn phí trên model local

[LM Studio](https://www.util-kits.io.vn/vi/blog/2026-09-01-local-ai-lm-studio-guide) có endpoint `/v1/messages` tương thích với Anthropic, nên cùng một đoạn code chạy được ở local chỉ bằng cách đổi ba biến môi trường:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m
uv run python rag_50_lines.py
```

Bỏ ba biến này và đặt `ANTHROPIC_API_KEY` thật là script chạy trên Claude Haiku 5.5. Kết quả của mình với model local:

```text
0.649  TechCrunch: SBF's trial starts soon, but how did he — and FTX — get here?
0.556  TechCrunch: SBF's trial starts soon, but how did he — and FTX — get here?
0.454  TechCrunch: SBF's trial starts soon, but how did he — and FTX — get here?

FTX was founded in 2019 by Sam Bankman-Fried, and at its peak, it was valued at $32 billion [1].
```

Cả ba đoạn đều lấy từ đúng bài báo chứa câu trả lời, và câu trả lời khớp với bài báo đó.

## Ba cách trả lời cùng một câu hỏi

Mình hỏi "When was FTX founded and what was it worth at its peak?" theo ba cách, dùng Qwen2.5-7B-Instruct-1M trong LM Studio trên RTX 5060 Ti. Cột chi phí ước tính số tiền nếu cùng lượng token đó chạy trên Claude Haiku 5.5 ($0.10 cho mỗi triệu token đầu vào, $0.50 cho mỗi triệu token đầu ra). Số token đếm bằng tokenizer của Qwen, nên chi phí chỉ là ước tính.

| Cách làm | Token đầu vào | Độ trễ | Câu trả lời | Chi phí ước tính trên Haiku 5.5 |
|---|---|---|---|---|
| Không có tài liệu | 40 | 0.74 s | Sai năm thành lập (2017), kèm tên một nhà đồng sáng lập không xuất hiện trong bài báo nào | $0.000025 |
| Đưa cả 20 bài báo vào prompt | 39,965 | thất bại | Vượt context 13,056 token mà model local được nạp | n/a |
| Đưa 5 bài báo vào prompt | 10,535 | 4.0 s | Đúng (2019, $32 tỷ) | $0.0011 |
| RAG, 3 đoạn liên quan nhất | 311 | 0.74 s | Đúng, có trích dẫn | $0.000051 |

Từ bảng trên:

1. **Model không có tài liệu trả lời rất chắc chắn nhưng sai.** Nó nhầm năm 2017 Alameda Research ra đời với FTX, trong khi bài của TechCrunch ghi FTX thành lập năm 2019. Con số $32 tỷ thì đúng, nên năm sai càng khó phát hiện.
2. **Nhồi tài liệu nhanh chóng hết chỗ.** Hai mươi bài báo tạo ra prompt 39,965 token, gấp ba context mà LM Studio cấp. Bạn có thể nâng giới hạn này, nhưng context window lớn hơn cần nhiều bộ nhớ GPU hơn.
3. **Khi nhồi vẫn vừa, RAG vẫn dùng ít hơn khoảng 34 lần token** cho cùng một câu trả lời đúng (311 so với 10,535), và trả kết quả trong chưa tới một phần năm thời gian.
4. **Toàn bộ corpus thì không vừa.** 609 bài báo cộng lại khoảng 1,25 triệu token, vượt context 1 triệu token của Claude Haiku 5.5. Nếu vừa, mỗi câu hỏi sẽ tốn khoảng $0.62 theo giá prompt dài. RAG vẫn giữ quanh mức 300 token mỗi câu hỏi dù corpus lớn tới đâu.

## Script này còn sai ở đâu

Script chạy đúng với câu hỏi này nhưng sẽ dẫn bạn đi sai ở các câu khó hơn. RAG giảm hallucination (bịa thông tin) nhưng sinh ra một kiểu lỗi mới: khi bước retrieval trả về sai đoạn văn, model dựng câu trả lời sai trên chính các đoạn đó và trích chúng làm bằng chứng. Các bài sau sẽ lấp những chỗ hổng này:

- Đoạn văn được cắt bằng bộ lọc thô thay vì một chiến lược chia chunk (bài 3).
- Vector nằm trong RAM và bị tính lại mỗi lần chạy (bài 4).
- Chưa có gì đo xem retrieval có tìm đúng bài báo không (bài 6).
- Tìm kiếm theo tên riêng và con số chính xác có thể trượt (bài 7).

Bài test local còn cho thấy một giới hạn thực tế: [context window](https://www.util-kits.io.vn/vi/glossary#context-window) mà phần cứng của bạn kham được nhỏ hơn con số model quảng cáo. Prompt dài còn làm giảm chất lượng câu trả lời, như bài [nén context cho coding agent](https://www.util-kits.io.vn/vi/blog/2026-10-12-context-compression-coding-agents) đã phân tích.

Bài 2 sẽ mở hàm `embed()` ra xem: embedding là gì, chọn model thế nào, và câu hỏi tiếng Việt có tìm ra được bài báo tiếng Anh không.

## Tự kiểm tra

1. Vì sao RAG giảm được hallucination nhưng không loại bỏ hết?
2. Tài liệu của bạn thay đổi mỗi ngày. Bạn chọn RAG hay fine-tuning, và vì sao?
3. Trong `rag_50_lines.py`, dòng nào làm bước retrieve, dòng nào làm bước augment?

## Tài Liệu Tham Khảo

1. [MultiHop-RAG dataset card](https://huggingface.co/datasets/yixuantt/MultiHopRAG) trên Hugging Face: corpus, bộ câu hỏi, license ODC-BY.
2. [Tang & Yang, MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391): số bài báo, khoảng thời gian và độ dài trung bình mỗi bài.
3. [LM Studio docs: Anthropic-compatible endpoints](https://lmstudio.ai/docs/developer/anthropic-compat): endpoint `/v1/messages`.
4. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): giá Claude Haiku 5.5 dùng cho phần ước tính chi phí.
