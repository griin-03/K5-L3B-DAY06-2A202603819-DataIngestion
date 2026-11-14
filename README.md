# Day 06 · Data Ingestion Engineering

**Học viên:** Lê Thanh Tùng  
**MSSV:** 2A202603819  
**Lớp:** L3B  

Dự án biến tài liệu khoa học kỹ thuật thành kho tri thức vector cho AI Agent, xây dựng pipeline trích xuất/chuẩn hóa PDF, triển khai 3 chiến lược chunking & metadata (A/B/C), và thực hiện benchmark đánh giá đa chiều (5 khía cạnh) trên 20 câu hỏi có gold evidence.

---

## 1. Cấu trúc thư mục

```text
K5-L3-Day06-DataIngestionEngineering/
├── STUDENT.md                              # Thông tin học viên và repo
├── README.md                               # Hướng dẫn setup, chạy và kiến trúc
├── data/
│   ├── llm_survey.pdf                     # Tài liệu nghiên cứu (35 trang nội dung, arXiv:2307.06435)
│   └── manifest.json                      # Nguồn, quyền, số trang, quy tắc tính
├── benchmark/
│   └── questions.json                     # 20 câu hỏi benchmark (15 dev / 5 test) kèm gold evidence
├── starter/
│   └── src/                               # Toàn bộ mã nguồn pipeline thực hiện
│       ├── ingestion.py                   # Đọc và chuẩn hóa văn bản PDF, nhận diện section
│       ├── chunking.py                    # Fixed-size (A) và Paragraph-aware (B/C)
│       ├── embeddings.py                  # Semantic Embedder thực (BAAI/bge-small-en-v1.5)
│       ├── store.py                       # Persistent ChromaDB store, cosine distance, metadata filter
│       ├── evaluation.py                  # Đo lường Hit@1, Evidence Hit@3, aggregation
│       └── run.py                         # Điều phối thực thi pipeline tự động
├── artifacts/
│   └── experiment_log.csv                 # Nhật ký thí nghiệm cấu hình A / B / C
├── results/
│   ├── trace_strategy_A.json              # Trace truy xuất top-3 chiến lược A
│   ├── trace_strategy_B.json              # Trace truy xuất top-3 chiến lược B
│   ├── trace_strategy_C.json              # Trace truy xuất top-3 chiến lược C
│   └── summary_metrics.json               # Tổng hợp số liệu định lượng dev/test
└── report/
    └── individual/
        └── 2A202603819_LeThanhTung.md     # Báo cáo cá nhân chi tiết theo mẫu
```

---

## 2. Hướng dẫn thiết lập và chạy pipeline

### Bước 1: Chuẩn bị môi trường Python
Yêu cầu Python 3.10 trở lên:
```bash
# Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install fastembed chromadb pypdf python-dotenv
```

### Bước 2: Chạy toàn bộ Ingestion & Benchmark Pipeline
```bash
python3 -m starter.src.run
```
Lệnh trên sẽ tự động:
1. Nạp và chuẩn hóa 35 trang tài liệu khoa học từ `data/llm_survey.pdf`.
2. Tạo 3 collection vector tương ứng 3 chiến lược trong ChromaDB (`./results/chromadb`).
3. Thực hiện embedding bằng model cục bộ `BAAI/bge-small-en-v1.5` (chi phí $0.00).
4. Đánh giá tự động 20 câu hỏi benchmark (15 dev / 5 test), lưu kết quả chi tiết top-3 traces và bảng thống kê vào thư mục `results/`.

---

## 3. Tóm tắt kết quả Benchmark

| Chiến lược | Phân chia | Hit@1 | Evidence Hit@3 | Số Chunks | Kích thước TB | Thời gian Ingest | Chi phí |
|---|---|---|---|---|---|---|---|
| **A (Fixed)** | Dev (15 câu) | 8/15 (53.3%) | 9/15 (60.0%) | 408 | 477.6 chars | 26.74s | $0.00 |
| **A (Fixed)** | Test (5 câu) | 4/5 (80.0%) | 4/5 (80.0%) | 408 | 477.6 chars | 26.74s | $0.00 |
| **B (Paragraph)** | Dev (15 câu) | 7/15 (46.7%) | 10/15 (66.7%) | 234 | 752.7 chars | 256.79s | $0.00 |
| **B (Paragraph)** | Test (5 câu) | 1/5 (20.0%) | 2/5 (40.0%) | 234 | 752.7 chars | 256.79s | $0.00 |
| **C (B + Filter)** | Dev (15 câu) | 8/15 (53.3%) | 10/15 (66.7%) | 234 | 752.7 chars | 491.37s | $0.00 |
| **C (B + Filter)** | Test (5 câu) | 1/5 (20.0%) | 1/5 (20.0%) | 234 | 752.7 chars | 491.37s | $0.00 |

Chi tiết phân tích sâu về 5 khía cạnh chất lượng (Coherence, Metadata Usefulness, Grounding Quality, Data Strategy Impact) xem tại:
👉 [Báo cáo cá nhân](report/individual/2A202603819_LeThanhTung.md)
