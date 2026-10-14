"""One-off: pick the 10 smoke questions from MultiHop-RAG and attach hand-written Vietnamese translations.

Rule (no cherry-picking by score): walk the question file in order and take the first
4 inference, 3 comparison and 3 temporal questions that have exactly 2 evidence articles.
Run from the lesson folder: uv run python eval/build_smoke.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.corpus import load_questions

VI = {
    1: "Ai bị cho là đã thổi phồng giá trị một căn hộ ở Manhattan lên mức chưa từng có trong lịch sử bất động sản New York theo 'Fortune', và cũng bị cáo buộc điều chỉnh giá trị căn hộ này để bù cho khoản thua lỗ ở một tài sản khác theo 'The Age'?",
    2: "Nhân vật gắn với công nghệ AI tạo sinh nào có việc rời OpenAI bị Fortune coi là gây sốc, và cũng là đối tượng của giả thuyết phổ biến rằng ông không hoàn toàn trung thực với hội đồng quản trị theo TechCrunch?",
    3: "Bài của TechCrunch về các công ty phần mềm và bài của Hacker News về The Epoch Times có cùng đưa tin về mức tăng doanh thu liên quan đến mô hình thanh toán và mô hình đăng ký hay không?",
    5: "Ai bị cáo buộc đã xây dựng một sàn tiền mã hóa phát đạt trên những lời dối trá, và bị bên công tố cáo buộc gian lận vì lợi ích cá nhân, theo cả Fortune và TechCrunch?",
    6: "Bài của TechCrunch về chính sách chia sẻ doanh thu từ lượt đăng ký của Twitch có cho thấy chiến lược kiếm tiền khác với bài của TechCrunch về kế hoạch bán gói đăng ký Beeper Mini của Beeper hay không?",
    7: "Bài của 'The New York Times' có cho rằng hàng phòng ngự của Buffalo Bills thành công nhờ đóng góp của Jordan Poyer, trong khi bài của 'Sporting News' cho rằng hàng phòng ngự của Baltimore Ravens cần cải thiện trước trận gặp Cincinnati Bengals hay không?",
    12: "Sau bài của TechCrunch ngày 7 tháng 10 năm 2023 về phát biểu của Dave Clark liên quan đến Flexport, và bài tiếp theo của TechCrunch ngày 30 tháng 10 năm 2023 về hành động của Ryan Petersen tại Flexport, bản chất của các sự kiện được đưa tin có thay đổi không?",
    29: "Cách 'The Independent - Travel' đưa tin về các khu trượt tuyết có thay đổi không, sau bài về 'những sườn dốc Thụy Sĩ ở Zermatt và các đường trượt của Vail ở Colorado' ngày 13 tháng 10 năm 2023, so với bài tiếp theo về 'khu trượt tuyết Tremblant' ngày 25 tháng 10 năm 2023?",
    41: "Có phải The Independent - Sports đã đưa tin về các chiến thắng trên sân nhà của All Blacks trước Ireland, Nam Phi và Argentina vào mùa hè năm ngoái, ngày 14 tháng 10 năm 2023, và The Roar | Sports Writers Blog đã đưa tin về các chiến thắng của Argentina trước All Blacks ở Christchurch năm ngoái và chiến thắng đầu tiên của họ năm 2020 ở Sydney, ngày 18 tháng 10 năm 2023, khiến việc đưa tin về các trận thua của All Blacks trước Argentina nhất quán hay không?",
    67: "Nhóm người nào có thể tận dụng cơn sốt quanh các sự kiện cụ thể để kiếm tiền bằng cách đặt cược vào những tình huống như đội nào sẽ dẫn trước khi kết thúc một hiệp nhất định, theo Sporting News?",
}

want = {"inference_query": 4, "comparison_query": 3, "temporal_query": 3}
out = []
for i, x in enumerate(load_questions()):
    t = x["question_type"]
    if want.get(t, 0) > 0 and len(x["evidence_list"]) == 2:
        want[t] -= 1
        out.append({"id": f"s{len(out) + 1:02d}", "type": t, "src_index": i, "q_en": x["query"],
                    "q_vi": VI[i], "expected_urls": [e["url"] for e in x["evidence_list"]]})
assert len(out) == 10 and {o["src_index"] for o in out} == set(VI)
path = Path(__file__).with_name("smoke.jsonl")
path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
print(f"wrote {len(out)} questions to {path}")
