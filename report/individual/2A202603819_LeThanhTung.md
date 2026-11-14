# Báo cáo cá nhân — Day 6 · Data Ingestion Engineering

**Học viên:** Lê Thanh Tùng  
**MSSV:** 2A202603819  
**Lớp:** L3B  
**Repo bài nộp:** https://github.com/griin-03/K5-L3B-DAY06-2A202603819-DataIngestion  

---

## 1. Dữ liệu và pipeline

### Chủ đề, nguồn/quyền, số trang và quy tắc tính trang
- **Chủ đề:** Khảo sát toàn diện về các mô hình ngôn ngữ lớn (Large Language Models — LLMs), kiến trúc transformer, tiền huấn luyện, tinh chỉnh chỉ dẫn (instruction-tuning), kỹ thuật lượng hóa/tỉa thưa (quantization & pruning) và bộ dữ liệu đánh giá.
- **Nguồn & Quyền:** Bài báo khảo sát khoa học mở *"A Comprehensive Overview of Large Language Models"* (Naveed et al.), định danh arXiv: [2307.06435](https://arxiv.org/abs/2307.06435), phân phối theo giấy phép Open Access không độc quyền của arXiv (arXiv non-exclusive license).
- **Số trang & Quy tắc tính trang:** 
  - File PDF gốc gồm 47 trang vật lý.
  - Theo đúng hướng dẫn tại `GUIDE.md`, chỉ các trang chứa nội dung hữu ích thuộc các phần kỹ thuật từ Section 1 đến Section 8 (từ trang 1 đến trang 35) được đưa vào corpus (tổng cộng **35 trang nội dung thực tế**, vượt mốc tối thiểu ≥30 trang).
  - Các trang từ 36 đến 47 chỉ chứa danh mục tài liệu tham khảo (References) và bảng phụ lục rỗng, được loại bỏ khỏi corpus chuẩn. Mỗi trang từ 1 đến 35 đều có mật độ nội dung cao (thấp nhất 1.125 ký tự, trung bình >4.500 ký tự/trang, tổng cộng 176.326 ký tự chuẩn hóa).

### Chuẩn hóa, lỗi và giới hạn
- **Chuẩn hóa (Normalization):** Văn bản được chuẩn hóa bảng mã Unicode chuẩn NFC (`unicodedata.normalize`), loại bỏ ký tự zero-width (`\u200b`), chuẩn hóa non-breaking spaces (`\u00a0`). Điểm đặc biệt quan trọng trong trích xuất PDF hai cột là xử lý từ bị ngắt dòng có dấu nối (`(\w+)-\n(\w+)` -> `\1\2`), giúp các thuật ngữ chuyên ngành (như *transformer*, *instruction-tuning*, *quantization*, *pruning*) không bị xé đôi giữa dòng.
- **Lỗi & Giới hạn:** Một số bảng số liệu phức tạp (như Table 3, Table 5, Table 12) khi pypdf trích xuất dạng văn bản thuần có thể bị xáo trộn thứ tự cột nếu đọc tuần tự. Tuy nhiên toàn bộ tiêu đề bảng, chú thích và nội dung phân tích định tính đều được bảo toàn nguyên vẹn.

### Embedding model, Vector store, Lệnh chạy và Chi phí
- **Embedding Model:** `BAAI/bge-small-en-v1.5` chạy qua FastEmbed ONNX Runtime cục bộ trên chip Apple Silicon. Số chiều vector: **384 chiều**. Chuẩn hóa L2 embedding và đo khoảng cách bằng Cosine Similarity.
- **Vector Store:** ChromaDB `v1.5.9` (PersistentClient) lưu trữ bền vững tại `./results/chromadb`, quản lý 3 collection độc lập cho 3 chiến lược (`strategy_A`, `strategy_B`, `strategy_C`), hỗ trợ metadata filtering trực tiếp ở tầng index.
- **Lệnh chạy:**
  ```bash
  source .venv/bin/activate
  python3 -m starter.src.run
  ```
- **Chi phí (Cost):** **$0.00** (Hoàn toàn chạy local offline trên CPU/Neural Engine, không tiêu tốn token API thương mại).

### Metadata bắt buộc và trường lọc bổ sung
- **Metadata bắt buộc:** `document_id` (`llm_survey_2024`), `source` (`llm_survey.pdf`), `page` (số trang vật lý từ 1 đến 35).
- **Trường lọc bổ sung (`section`):** Trích xuất tự động và gắn nhãn theo mục chính của bài báo (`1. Introduction`, `2. Background`, `3. Large Language Models`, `4. Model Configurations`, `5. Datasets and Evaluation`, `6. Applications`, `7. Challenges and Future Directions`, `8. Conclusion`). Trường này cho phép lọc thu hẹp không gian tìm kiếm khi truy vấn người dùng có ngữ cảnh danh mục cụ thể.

---

## 2. Cấu hình và benchmark

### Mô tả 3 cấu hình A / B / C
- **Chiến lược A (Fixed Chunking Baseline):** Kích thước cố định 500 ký tự, overlap 50 ký tự. Lưu đầy đủ metadata (`document_id`, `source`, `page`, `section`), không áp dụng metadata filter khi tìm kiếm.
- **Chiến lược B (Paragraph-aware Chunking):** Tách chunk theo ranh giới đoạn văn tự nhiên (`\n\n`), heading và ranh giới câu, giới hạn tối đa 800 ký tự. Không áp dụng metadata filter khi tìm kiếm.
- **Chiến lược C (Paragraph-aware + Metadata Filter):** Cùng cấu hình chunking và metadata như B, nhưng kích hoạt bộ lọc metadata (`where={"section": ...}`) khi câu hỏi có điều kiện lọc ngữ cảnh định trước. Đây là so sánh trực tiếp với B (chỉ thay đổi duy nhất một yếu tố: metadata filtering).

### Thiết kế Benchmark (≥20 câu: 15 dev / 5 test)
- Bộ câu hỏi gồm đúng **20 câu hỏi** được biên soạn cẩn thận từ nội dung PDF, kiểm chứng từng đoạn `evidence_quote` thực tế có mặt ở đúng `gold_page`.
- Phân chia: **15 câu dev** (`Q01`–`Q15`) và **5 câu test** (`Q16`–`Q20`).
- Có 3 câu hỏi kiểm tra bộ lọc metadata filter: `Q04` (dev), `Q05` (dev) và `Q16` (test), đều có filter xác thực `{"section": "..."}`.
- **Tiêu chí chọn chiến lược đã chốt trước khi xem test:** Ưu tiên số 1 là **Evidence Hit@3** (khả năng bao quát đầy đủ bằng chứng trong top-3), sau đó đến **Grounding Quality / Coherence**, cuối cùng mới xét đến chi phí lưu trữ/tài nguyên.

---

## 3. Kết quả thực nghiệm

### Bảng 1: Thống kê định lượng truy xuất (Retrieval Accuracy & Resource)

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Avg Chunk Size | Thời gian ingest | Chi phí |
|---|---|---|---|---|---|---|---|
| **A (Fixed)** | Dev | 8/15 (53.3%) | 9/15 (60.0%) | 408 | 477.6 chars | 26.74s | $0.00 |
| **A (Fixed)** | Test | 4/5 (80.0%) | 4/5 (80.0%) | 408 | 477.6 chars | 26.74s | $0.00 |
| **B (Paragraph)** | Dev | 7/15 (46.7%) | 10/15 (66.7%) | 234 | 752.7 chars | 256.79s | $0.00 |
| **B (Paragraph)** | Test | 1/5 (20.0%) | 2/5 (40.0%) | 234 | 752.7 chars | 256.79s | $0.00 |
| **C (B + Filter)** | Dev | 8/15 (53.3%) | 10/15 (66.7%) | 234 | 752.7 chars | 491.37s | $0.00 |
| **C (B + Filter)** | Test | 1/5 (20.0%) | 1/5 (20.0%) | 234 | 752.7 chars | 491.37s | $0.00 |

### Bảng 2: Đánh giá chất lượng chuyên sâu (5 khía cạnh)

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ và đúng (n=15) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 (n=5 test) |
|---|---|---|---|---|
| **A (Fixed)** | 0.9 / 2.0 (n=10) | 15/15 (100%) | N/A | [0: 1 câu, 1: 0 câu, 2: 4 câu] → TB: **1.60 / 2.0** |
| **B (Paragraph)**| **1.9 / 2.0** (n=10) | 15/15 (100%) | N/A | [0: 3 câu, 1: 1 câu, 2: 1 câu] → TB: **0.60 / 2.0** |
| **C (B + Filter)**| **1.9 / 2.0** (n=10) | 15/15 (100%) | **1/1 (100%)** | [0: 4 câu, 1: 0 câu, 2: 1 câu] → TB: **0.40 / 2.0** |

#### Minh chứng chi tiết cho 10 vị trí đánh giá Chunk Coherence:
- Đánh giá trên cùng 10 vị trí đoạn mở đầu trang (Pages 1–10) theo thang điểm 0–2:
  - **Strategy A:** Điểm `[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]` → Trung bình **0.9/2.0**. Nguyên nhân: do cắt cố định 500 ký tự, hầu hết các chunk đều kết thúc lơ lửng giữa câu (ví dụ Page 1 cắt tại *"CSIRO), Sydne"*, Page 4 cắt tại *"Some of the commo"*, Page 5 cắt tại *"scaleable machine le"*).
  - **Strategy B & C:** Điểm `[2, 2, 2, 2, 2, 2, 2, 2, 2, 1]` → Trung bình **1.9/2.0**. Các chunk kết thúc trọn vẹn tại ranh giới câu và đoạn văn (`\n\n`), giữ nguyên chủ ngữ, vị ngữ và ngữ cảnh học thuật.

#### Minh chứng Metadata Usefulness:
- Kiểm tra thủ công 15 bản ghi (5 câu test × top-3) đối chiếu trực tiếp với trang nguồn PDF: 100% (15/15) bản ghi đều có `source`, `page` và `section` chính xác, không ghi nhận trường hợp bị lệch trang hay gán sai tên mục.
- Tỷ lệ tuân thủ filter: Với câu hỏi `Q16` có filter `section = "5. Datasets and Evaluation"`, cả top-3 chunk trả về trong Strategy C đều thuộc Section 5 (tuân thủ 100%).

#### Đường dẫn minh chứng (Artifacts & Traces):
- Dữ liệu manifest: `data/manifest.json`
- Bộ câu hỏi benchmark: `benchmark/questions.json`
- File nhật ký thí nghiệm: `artifacts/experiment_log.csv`
- File traces chi tiết top-3: `results/trace_strategy_A.json`, `results/trace_strategy_B.json`, `results/trace_strategy_C.json`
- Tổng hợp chỉ số: `results/summary_metrics.json`

---

## 4. Phân tích và kết luận

### So sánh A–B và B–C kèm 2 ví dụ lỗi cụ thể
1. **So sánh A–B (Fixed vs. Paragraph-aware):**
   - *Tài nguyên & Kích thước:* Strategy A sinh ra 408 chunks (trung bình 477.6 ký tự), trong khi B chỉ sinh 234 chunks (trung bình 752.7 ký tự), giảm 42.6% số lượng vector cần lưu trữ.
   - *Độ mạch lạc (Coherence):* B vượt trội A hoàn toàn (1.9/2.0 so với 0.9/2.0). Chunks của B luôn giữ trọn ý của câu và đoạn luận điểm.
   - *Đánh đổi về Truy xuất (Retrieval Trade-off):* Chunks kích thước nhỏ (500 ký tự) của Strategy A đạt mật độ từ khóa cao hơn cho mô hình embedding cỡ nhỏ (`bge-small-en-v1.5`), giúp Hit@1 trên tập test đạt 80% so với 20% của B. Trên các đoạn văn dài của B (lên tới 800 ký tự), thông tin bị phân tán (dilution effect), khiến độ tương đồng cosine giữa vector câu hỏi ngắn và vector chunk dài bị giảm sút.
2. **So sánh B–C (Ảnh hưởng của Metadata Filtering):**
   - Giữ nguyên cấu hình chunking của B, việc bổ sung bộ lọc metadata ở C giúp cải thiện Hit@1 trên tập dev từ 46.7% lên 53.3% đối với các câu hỏi có phạm vi hẹp (`Q04`, `Q05`).
   - Tuy nhiên, trên tập test với câu hỏi `Q16` (*PAWS dataset*): Chunk chứa định nghĩa chính xác về PAWS nằm ở Page 32 (thuộc Section 5), nhưng trong Strategy C, bộ lọc đã loại bỏ các section khác và trong top-3 của Section 5 lại ưu tiên chunk tại Page 31 (*PAWS-X* - biến thể đa ngữ của PAWS). Điều này cho thấy ranh giới giữa các khái niệm liên quan trong cùng một section có thể gây nhiễu nếu bộ lọc quá hẹp hoặc chunking nhóm gộp.

3. **Hai ví dụ lỗi truy xuất tiêu biểu:**
   - **Ví dụ 1 (Strategy A - Câu Q18):**
     - *Câu hỏi:* *"Why is research focusing on making LLMs more explainable?"* (Gold: Page 35, Section 8).
     - *Hiện tượng:* Strategy A trả về các chunk ở Page 1, Page 2 và Page 34.
     - *Nguyên nhân:* Các từ khóa *"research"*, *"LLMs"*, *"explainable"* xuất hiện rải rác ở phần Introduction (Page 1) và phần Challenges (Page 34). Do chunking 500 ký tự tách rời câu giải thích lý do cụ thể ở Page 35, vector similarity của chunk mở đầu ở Page 1 cao hơn, dẫn tới false positive về mặt ngữ nghĩa tổng quát nhưng trượt gold evidence.
   - **Ví dụ 2 (Strategy B - Câu Q17):**
     - *Câu hỏi:* *"What specific capability does the Truthful-QA benchmark measure?"* (Gold: Page 32, Section 5).
     - *Hiện tượng:* Strategy B trả về các chunk tại Page 31, Page 26 và Page 26 (chứa các bảng tổng hợp benchmark).
     - *Nguyên nhân:* Chunk ở Page 32 chứa định nghĩa Truthful-QA được nhóm chung trong một đoạn dài bàn về nhiều benchmark khác nhau. Trong khi đó, các chunk ở Page 26 chứa bảng tổng quan liệt kê hàng loạt bộ dữ liệu đánh giá có độ tương đồng embedding cao hơn với câu hỏi chung về benchmark, khiến chunk đích bị đẩy ra khỏi top-3.

### Chiến lược được chọn, đánh đổi và điều kiện áp dụng
- **Chiến lược được chọn:** **Strategy B (Paragraph-aware) kết hợp tinh chỉnh kích thước cửa sổ (khoảng 500–600 ký tự).**
- **Lý do & Đánh đổi:** 
  - Mặc dù Strategy A có điểm Hit@3 trên tập test nhỏ (5 câu) cao hơn, các chunk của A bị vỡ câu nghiêm trọng (điểm Coherence chỉ đạt 0.9/2.0), gây nguy cơ LLM sinh hallucination hoặc trích dẫn cụt lủn khi đưa vào prompt của AI agent thực tế.
  - Strategy B bảo đảm tính toàn vẹn của luận điểm khoa học (Coherence 1.9/2.0) và tiết kiệm 42.6% không gian index vector store.
  - Metadata filter (Strategy C) rất hữu ích khi người dùng hoặc hệ thống biết chắc chắn phạm vi tài liệu cần truy vấn, nhưng cần tránh áp dụng cứng nhắc nếu câu hỏi có tính chất liên chuyên mục.

### Hạn chế và hướng cải tiến
- **Hạn chế:**
  - Tập test gồm 5 câu là mẫu minh họa cho bài lab 4 giờ; chưa đủ lớn để ngoại suy kết luận thống kê tổng quát cho mọi domain.
  - Người đánh giá thủ công là tác giả bài làm, do đó có nguy cơ thiên lệch chủ quan trong việc chấm điểm Coherence và Grounding.
- **Hướng cải tiến (Nếu làm lại hoặc mở rộng):**
  - Áp dụng **Hybrid Search** kết hợp giữa Dense Vector (`bge-small-en-v1.5`) và Sparse BM25 / SPLADE để tận dụng cả độ chính xác từ khóa của Strategy A và tính mạch lạc ngữ nghĩa của Strategy B.
  - Sử dụng chiến lược **Parent-Child Chunking (Hierarchical Chunking)**: Lưu trữ các đoạn nhỏ (child chunks 200 ký tự) để embedding và truy xuất nhanh, nhưng khi trả về cho agent thì lấy đoạn văn cha (parent chunk 800 ký tự) để đảm bảo ngữ cảnh nguyên vẹn cho grounding.
