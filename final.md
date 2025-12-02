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

# 11. Giải thích dễ hiểu cho người mới bắt đầu
Hãy tưởng tượng bạn xem nhật ký hành động của một chương trình. Nhật ký đó liệt kê tuần tự các API Windows mà chương trình gọi (mở file, sửa registry, gửi mạng...).  

DawnGNN làm 3 việc quan trọng:

1. **Đổi danh sách thành đồ thị**  
  Mỗi API là một nút; cạnh có hướng nối API A sang API B nếu B xuất hiện ngay sau A. Như vậy ta thấy được “dòng chảy” thay vì chỉ chuỗi dài.
2. **Thêm ý nghĩa (ngữ nghĩa) cho mỗi nút**  
  Thay vì chỉ tên API (ví dụ `CreateFile`), DawnGNN lấy mô tả chức năng chính thức từ trang Microsoft (ví dụ: “Creates or opens a file or I/O device...”). Đoạn mô tả này được đưa vào BERT để biến thành một vector số (embedding) đại diện ý nghĩa.
3. **Học mẫu hành vi với GNN (GAT)**  
  GAT nhìn vào một nút và các láng giềng của nó trong đồ thị, tính mức độ “quan trọng” (attention). Những chuỗi chuyển tiếp đáng ngờ (tạo file → sửa registry → kết nối mạng) sẽ nổi bật hơn. Sau đó gộp toàn bộ nút thành một vector chung và phân loại: mã độc hay lành.

**Quy trình khi áp dụng:**
- Chạy file trong sandbox để thu chuỗi API.
- Xây đồ thị + gắn embedding đã chuẩn bị.
- Cho vào GAT → nhận kết quả phân loại.

**Tại sao hiệu quả hơn dùng chuỗi thô?**
- Tên API ngắn, dễ mơ hồ; mô tả dài chứa ngữ cảnh thật sự.
- Đồ thị giữ được quan hệ lân cận (ai trước ai sau) rõ ràng hơn chuỗi phẳng.
- Attention giúp tập trung vào các mẫu hành vi nguy hiểm thay vì coi mọi bước như nhau.

**Ví dụ trực quan (ASCII):**
```
CreateFile -> WriteFile -> RegSetValue -> InternetConnect
|________ nhóm thao tác file/ghi -> thay đổi hệ thống -> kết nối ra ngoài
```

Nếu thêm ngữ nghĩa, mô hình hiểu “CreateFile + WriteFile” = thao tác I/O, “RegSetValue” = thay đổi hệ thống, “InternetConnect” = truyền thông; kết hợp lại thành mẫu lan truyền/thiết lập kênh.

**Tóm lại:**
- Dữ liệu: chuỗi API + mô tả chức năng.
- Chuyển đổi: chuỗi → đồ thị + embedding ngữ nghĩa.
- Mô hình: GAT học quan hệ + ý nghĩa.
- Đầu ra: Nhãn malware / benign.

**Điểm nhớ nhanh:**
- Thêm ngữ nghĩa = tăng phân biệt.
- Đồ thị = biểu diễn cấu trúc.
- Attention = lọc hành vi quan trọng.

# 12. Giải thích thuật ngữ (Glossary)
Mục này giải thích ngắn gọn các thuật ngữ quan trọng cho người mới bắt đầu, không cần nền tảng AI sâu.

- **API (Application Programming Interface):** Tập hợp hàm/hệ thống mà chương trình gọi để tương tác với hệ điều hành (mở file, đọc registry...). Mỗi lần gọi là một "dấu vết" hành vi.
- **API Call Sequence (Chuỗi gọi API):** Danh sách tuần tự các API được chương trình gọi trong quá trình chạy. Ví dụ: `CreateFile → WriteFile → RegSetValue → InternetConnect`.
- **Sandbox (Hộp cát):** Môi trường cách ly an toàn để chạy mã nghi ngờ, ghi lại hành vi mà không ảnh hưởng hệ thống thật.
- **Embedding:** Biểu diễn một thực thể (API, từ ngữ) thành vector số để mô hình xử lý. Giống như gán toạ độ cho khái niệm.
- **BERT:** Mô hình ngôn ngữ học ngữ cảnh hai chiều của văn bản. Ở đây dùng để suy ra ý nghĩa từ mô tả chức năng API.
- **MLM (Masked Language Model):** Nhiệm vụ trong đó một số từ bị che `[MASK]` và mô hình đoán lại; giúp học quan hệ ngữ cảnh.
- **Graph (Đồ thị):** Tập nút (nodes) và cạnh (edges). Nút = API; cạnh = quan hệ kế tiếp giữa hai API trong chuỗi.
- **Directed Edge (Cạnh có hướng):** Cạnh có chiều A → B (A xảy ra trước B). Giúp giữ thứ tự logic.
- **GNN (Graph Neural Network):** Mô hình học trên đồ thị bằng cách truyền và tổng hợp thông tin giữa các nút.
- **Message Passing:** Quá trình mỗi nút nhận thông tin từ láng giềng và cập nhật biểu diễn mình.
- **GAT (Graph Attention Network):** Loại GNN dùng cơ chế attention để gán trọng số khác nhau cho từng láng giềng quan trọng.
- **Attention:** Cách mô hình tập trung mạnh hơn vào phần "quan trọng" thay vì đối xử đều.
- **GCN / GIN:** Hai biến thể GNN khác; GCN dùng chuẩn hoá ma trận kề, GIN dùng MLP mạnh để tổng hợp thông tin.
- **One-hot Encoding:** Mã hoá một thực thể thành vector với đúng một vị trí = 1, còn lại = 0. Mất ngữ nghĩa liên hệ.
- **Word2Vec:** Kỹ thuật học embedding dựa trên tần suất/đồng xuất hiện; không bắt được ngữ cảnh sâu như BERT.
- **MLP (Multilayer Perceptron):** Mạng thần kinh nhiều lớp fully-connected dùng ở cuối để phân loại.
- **Classification (Phân loại):** Dự đoán nhãn đầu ra (malware hay benign).
- **Precision / Recall / TNR / Accuracy / F1:** Các thước đo đánh giá. Precision = đúng trong số dự đoán malware; Recall = bắt được bao nhiêu malware thật; TNR = đúng trong số benign; Accuracy = tỷ lệ đúng tổng thể; F1 = hài hoà giữa Precision & Recall.
- **Imbalanced Dataset (Tập dữ liệu mất cân bằng):** Khi một lớp (ví dụ malware) nhiều hơn lớp kia rất nhiều → khó học lớp hiếm.
- **Adversarial Attack (Tấn công đối kháng):** Chiến lược tinh chỉnh đầu vào để đánh lừa mô hình phân loại sai.
- **Forced Execution:** Kỹ thuật ép mã độc đi vào các nhánh thực thi tiềm năng để lộ hành vi.
- **Provenance Graph:** Đồ thị nguồn gốc luồng dữ liệu/hành động giúp phân tích APT phức tạp.

# 13. Flow hoạt động (Chi tiết tuần tự)
Sơ đồ chữ (ASCII) mô tả pipeline từ dữ liệu thô đến dự đoán:
```
[Chạy exe trong Sandbox]
     |
     v
    (Thu chuỗi API)
     |
     v
 [Xây Đồ Thị API]
  Nodes: API unique
  Edges: hướng theo thứ tự
     |
     +-----------------------------+
     |                             |
     v                             v
 (Crawl tài liệu API)           (Làm sạch mô tả)
     |                             |
     +-------------[Corpus mô tả]--+
         |
         v
       [Huấn luyện / Fine-tune BERT (MLM)]
         |
         v
     (Sinh API Embeddings từ mô tả)
         |
     +-------------+--------------+
     |                            |
     v                            v
 (Gán embedding vào nút)      (Chuẩn bị batch đồ thị)
     |
     v
    [GAT Message Passing nhiều lớp]
     |
     v
      (Graph Embedding)
     |
     v
     [MLP Phân loại]
     |
     v
   => Kết quả: Malware / Benign
```

Chuỗi hoạt động rút gọn:
1. Thu thập hành vi runtime.
2. Chuyển chuỗi thành đồ thị.
3. Lấy và mã hoá mô tả API bằng BERT.
4. Gán embedding vào nút đồ thị.
5. GAT học quan hệ + ngữ nghĩa.
6. MLP phân loại nhãn.

Lý do mỗi bước tồn tại:
- Đồ thị: giữ cấu trúc tuần tự tốt hơn chuỗi phẳng.
- BERT: thêm chiều ngữ nghĩa, hạn chế mù context.
- GAT: làm nổi bật quan hệ hành vi quan trọng (các chuỗi nghi vấn).
- MLP cuối: đơn giản, đủ cho quyết định nhị phân.

Điểm mở rộng dễ áp dụng:
- Thêm thông tin tham số API vào embedding.
- Dùng self-supervised pretext tasks trên đồ thị (mask node / edge prediction).
- Áp dụng cơ chế cảnh báo sớm (early exit) với ngưỡng độ tin cậy.
