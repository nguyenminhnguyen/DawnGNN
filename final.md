# 1. Basic Info
- **Tên đề tài:** DawnGNN – Phát hiện mã độc Windows tăng cường bằng tài liệu (documentation-augmented) với GNN
- **Miền bài toán:** Phát hiện mã độc Windows dựa trên hành vi động (API calls)
- **Thành phần chính:** API Graph Constructor, API2Vec (BERT) Embedding Layer, GNN Classifier (GAT/GCN/GIN)
- **Đóng góp chính:** Khai thác ngữ nghĩa từ tài liệu API chính thức của Microsoft để tăng cường đặc trưng cho đồ thị API; so sánh nhiều kỹ thuật embedding và GNN; đạt kết quả tốt trên 3 bộ dữ liệu công khai.

# 2. Problem Definition
- **Vấn đề:** Phát hiện mã độc Windows dựa trên chuỗi gọi API động thường chỉ dùng tên API/tần suất, thiếu ngữ nghĩa dẫn đến dễ bị né tránh.
- **Mục tiêu:** Khai thác thêm chiều ngữ nghĩa mới từ tài liệu API chính thức (mô tả chức năng) để tạo embedding cho nút API, kết hợp cấu trúc chuỗi thành đồ thị, từ đó nâng hiệu quả phân loại mã độc/benign.
- **Đầu ra:** Nhãn `malware` hoặc `benign` cho mỗi chương trình Windows dựa trên đồ thị API có thuộc tính ngữ nghĩa.

# 3. Key Idea
- **Tài liệu API → Ngữ nghĩa API:** Mô tả chức năng trong tài liệu API chính thức chứa thông tin ngữ nghĩa giàu hơn tên API.
- **BERT cho embedding:** Dùng BERT (MLM pretrain) để mã hoá mô tả chức năng thành vector ngữ nghĩa cho từng API (API2Vec).
- **Đồ thị API:** Biến chuỗi gọi API thành đồ thị có hướng, cạnh biểu diễn quan hệ tuần tự; học ngữ cảnh qua message passing.
- **GAT ưu thế:** Cơ chế attention giúp trọng số hoá láng giềng quan trọng, phù hợp đồ thị có hướng và ngữ cảnh gọi API.

# 4. System Architecture
- **Pipeline tổng quát:**
  - Thu thập chuỗi API runtime bằng Cuckoo Sandbox.
  - Xây dựng đồ thị API từ thứ tự gọi (nút: API; cạnh: quan hệ kế tiếp).
  - Thu thập/crawl tài liệu API chính thức, làm sạch nội dung mô tả chức năng.
  - Mã hoá mô tả bằng BERT để tạo thuộc tính nút (API embeddings).
  - Huấn luyện bộ phân loại GNN (ưu tiên GAT) trên đồ thị gán thuộc tính để phát hiện mã độc.
  
  Hình minh họa kiến trúc tổng thể:
  
  ![Kiến trúc DawnGNN](paper/images/Fig3.png)
  
  - Đầu vào là chuỗi API động từ sandbox, được chuyển thành đồ thị có hướng.
  - Mỗi nút API được gắn embedding sinh từ BERT dựa trên mô tả chức năng chính thức.
  - GAT lan truyền thông điệp theo cạnh, sử dụng attention để tập trung vào láng giềng quan trọng, tạo graph embedding dùng cho phân loại.

# 5. Method
## 5.1. Data / Input
- **Nguồn dữ liệu:** 3 bộ dataset API calls công khai (MalBehavD-V1, PE_APICALLS, APIMDS).
- **Dạng đầu vào:** Chuỗi gọi API động (từ exe chạy trong môi trường cô lập); tài liệu mô tả chức năng API từ trang chính thức Microsoft.
- **Tiền xử lý:**
  - Chuyển chuỗi API thành đồ thị có hướng; xây ma trận kề `A` và ma trận thuộc tính `X`.
  - Làm sạch mô tả API: bỏ tên API, viết tắt/annotation không mang nghĩa chức năng (Unicode, ARP...).
  
  Ví dụ website tài liệu API (vùng mô tả chức năng dùng để embedding):
  
  ![Ví dụ trang tài liệu API](paper/images/Fig2.png)

## 5.2. Model / Algorithm
- **Embedding:**
  - BERT (base/small) với nhiệm vụ MLM trên tập mô tả API; lấy hidden state lớp cuối làm embedding API (kích thước 512/768 tuỳ phiên bản).
  - So sánh với one-hot và Word2Vec để chứng minh vai trò ngữ nghĩa.
- **GNN:**
  - GAT: attention-based message passing, đa đầu attention; phù hợp đồ thị có hướng và chọn lọc láng giềng quan trọng.
  - GCN, GIN: làm baseline so sánh. GIN dùng MLP aggregation mạnh hơn GCN.
- **Phân loại:** Graph embedding qua tổng hợp node representations → MLP dự đoán `malware/benign`.
  
  Minh họa cơ chế GAT và truyền thông điệp qua láng giềng:
  
  ![API Graph Structure Modeling với GAT](paper/images/Fig8.png)

## 5.3. Training pipeline
- **Bước 1:** Crawl và chuẩn hoá mô tả chức năng cho tất cả API; pretrain/fine-tune BERT với MLM trên corpus mô tả API.
- **Bước 2:** Tạo đồ thị API từ chuỗi gọi cho từng mẫu; gán embedding nút từ BERT.
- **Bước 3:** Huấn luyện GNN (GAT/GCN/GIN) với tham số tối ưu (epoch≈100, batch≈128, lr≈1e-4; hidden dims tuỳ mô hình: GCN=32, GIN=16, GAT=12).
- **Bước 4:** Tổng hợp graph embedding; huấn luyện MLP phân loại.
- **Chia tập:** Shuffle; train 80%, val 10%, test 10%.
  
  Thống kê corpus tài liệu API (quy mô từ vựng/độ dài mô tả):
  
  ![Phân bố số từ và word cloud tài liệu API](paper/images/Fig5.png)
  
  ![Word cloud](paper/images/Fig6.png)

# 6. Experiments
- **Thiết lập:** Ubuntu 20.04, Python 3.8.10, PyTorch 2.0.0, Transformers 4.28.1, Scikit-learn/Numpy/Pandas/Requests; GPU RTX 3060.
- **Đánh giá:** Precision, Recall, TNR, Accuracy, F1-score; dựa trên TP/TN/FP/FN.
- **Kết quả chính:**
  - BERT-based embedding vượt trội one-hot/Word2Vec trên mọi mô hình học.
  - GNN trên đồ thị (đặc biệt GAT) vượt thống kê/chuỗi (RF/LSTM).
  - Trên MalBehavD-V1: DawnGNN đạt Accuracy≈0.9638; cải thiện so với MalDy và MalDetConv.
  - Trên PE_APICALLS và APIMDS: BERT cải thiện hiệu năng trong tập mất cân bằng; TNR thấp hơn do số mẫu benign ít.
  
  Bảng và hình minh hoạ kết quả:
  
  ![Hiệu năng theo mã hoá PE_APICALLS/APIMDS](paper/images/Table4.png)
  
  ![So sánh tham số/hyper-parameters tối ưu](paper/images/Table5.png)
  
  ![So sánh GNN theo epochs](paper/images/Fig9.png)
  
  ![So sánh GCN/GIN/GAT](paper/images/Table6.png)
  
  ![So sánh với phương pháp hiện có](paper/images/Table7.png)

# 7. Strengths & Weaknesses
- **Strengths:**
  - Khai thác nguồn ngữ nghĩa chưa được dùng rộng rãi (tài liệu API chính thức), bổ sung mạnh cho đặc trưng hành vi.
  - Kết hợp cấu trúc đồ thị + ngữ nghĩa BERT; GAT phù hợp đồ thị có hướng, chọn lọc láng giềng.
  - Tính tổng quát: cơ chế có thể mở rộng sang Android/Linux.
- **Weaknesses / Limitations:**
  - Xây đồ thị dựa vào thứ tự gọi còn thô; cần ghép tham số API chính xác (DMalNet) để tinh hơn.
  - Phụ thuộc chống-crawl/thu thập tài liệu; dữ liệu mô tả có thể thiếu/không đồng nhất.
  - Mất cân bằng dữ liệu làm TNR giảm; khó học mẫu benign ít.
  - Dễ bị tấn công đối kháng vào GNN; cần cơ chế robust.
  - Hạn chế động phân tích: mã độc dùng kỹ thuật che giấu (delayed/conditional exec) có thể né; cần tăng phủ thực thi (X-Force).

# 8. For Reproduction
- **Môi trường:** Python 3.8+, PyTorch 2.0+, Transformers 4.28+, scikit-learn, numpy, pandas, requests; GPU khuyến nghị.
- **Dữ liệu:**
  - API call sequences từ sandbox (ví dụ Cuckoo) cho các mẫu exe.
  - Tập mô tả chức năng API (crawl Microsoft; lưu HTML/Markdown; làm sạch).
- **Các bước:**
  1. Crawl tài liệu API, trích câu mô tả chức năng; làm sạch.
  2. Huấn luyện hoặc tinh chỉnh BERT với MLM trên corpus mô tả.
  3. Biến chuỗi API thành đồ thị có hướng; xây `A`, `X` (embedding từ BERT).
  4. Huấn luyện GNN (ưu tiên GAT) với tham số tối ưu; tổng hợp graph embedding.
  5. Huấn luyện MLP; đánh giá bằng Precision/Recall/TNR/Accuracy/F1.
- **Mẹo tái hiện:**
  - Nếu thiếu tài liệu API: dùng bản cache trang, hoặc tìm mô tả thay thế từ nguồn chính thức.
  - Bắt đầu với BERT-base để có embedding giàu ngữ nghĩa; sau tối ưu hoá kích thước/phiên bản.
  
  Gợi ý thiết lập tham số thực nghiệm (tham khảo bài báo):
  - Epoch: 100
  - Batch size: 128
  - Learning rate: 0.0001
  - Hidden dims: GCN=32, GIN=16, GAT=12
  - Tách tập: train 80% / val 10% / test 10%

# 9. Notes / Questions
- Làm thế nào chuẩn hoá mô tả để tránh nhiễu (tên API, viết tắt, annotation) tối đa?
- Có thể kết hợp tham số API, vị trí trong chuỗi, ngữ nghĩa code vào thuộc tính nút để tăng phân biệt?
- Khả năng chống tấn công đối kháng cho GNN trong bài toán này nên triển khai theo hướng nào?
- Mở rộng sang đa tiến trình (process graphs) và hoạt động mạng để tăng độ bao phủ?

# 10. Possible Extensions
- **Nguồn ngữ nghĩa bổ sung:** Tham số, giá trị trả về, remarks, requirements từ trang tài liệu.
- **Mô hình ngôn ngữ:** Thử RoBERTa/ALBERT/DistilBERT/Sentence-BERT; hoặc LLM tinh chỉnh riêng cho tài liệu API.
- **Học đồ thị:** Thử Jumping Knowledge, tự giám sát, hoặc GNN mạnh hơn; thêm cơ chế robustness chống đối kháng.
- **Phân tích động:** Áp dụng forced execution để tăng coverage; kết hợp provenance graphs cho APT.
