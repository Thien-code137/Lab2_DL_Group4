# Lab 2 — Plan & Problem Understanding

> **Status:** Planning — chưa triển khai code, chưa chạy thí nghiệm và chưa có kết quả thực nghiệm.  
> **Mục đích của file:** Chốt cách hiểu đề, xác định việc cần làm, đầu ra cần nộp và trình tự triển khai trước khi bắt đầu viết code.

## 1. Nguồn yêu cầu

Tài liệu chính là slide **Chapter 3 — Training Deep Neural Networks**, phần **Practice 2 — Hands-on practice with pre-trained neural network architectures**, trang 23–24.

### Các yêu cầu được nêu trực tiếp trong đề

1. Thử nghiệm với nhiều mô hình đã được huấn luyện trước (*pre-trained models*), ví dụ VGG, DenseNet, v.v.
2. Điều chỉnh siêu tham số (*hyperparameters*) như learning rate, batch size, v.v.
3. Dùng TensorBoard để theo dõi quá trình huấn luyện.
4. In/xem cấu trúc mô hình và xác định các lớp, tham số.
5. Thay lớp phân loại cuối để phù hợp với số lớp của bộ dữ liệu.
6. Chọn cách huấn luyện: đóng băng các lớp trước (*transfer learning*) hoặc mở một số lớp cuối để fine-tune.
7. Chuẩn bị dữ liệu bằng `torchvision.transforms` và `torch.utils.data.DataLoader`.
8. Định nghĩa loss function, optimizer và vòng lặp huấn luyện.
9. Đánh giá mô hình trên tập test.

**Lưu ý:** Đề chưa quy định rõ số mô hình, số epoch, giá trị learning rate/batch size, ngưỡng accuracy, định dạng báo cáo hay cách nộp checkpoint. Các phần này cần được thống nhất trong nhóm hoặc đối chiếu với yêu cầu bổ sung của giảng viên.

## 2. Bài toán cần giải quyết

Bài lab thực hành **transfer learning và fine-tuning cho bài toán phân loại ảnh**.

Thay vì huấn luyện toàn bộ mạng từ trọng số ngẫu nhiên, ta tải một mô hình đã học trên ImageNet, thay lớp phân loại cuối cho phù hợp với nhãn của dữ liệu đang dùng, rồi huấn luyện một phần hoặc toàn bộ những lớp được chọn.

### Bộ dữ liệu dự kiến

**Đề xuất ban đầu: CIFAR-10**, vì slide trang 24 dùng bộ dữ liệu này làm ví dụ. Đây là lựa chọn triển khai dự kiến, không phải khẳng định rằng đề bắt buộc dùng CIFAR-10. Nếu giảng viên hoặc nhóm đã được chỉ định một bộ dữ liệu khác, cần thay đổi kế hoạch trước khi code.

CIFAR-10 có 10 lớp ảnh. Mục tiêu là dự đoán đúng lớp của mỗi ảnh.

### Đầu vào và đầu ra của bài toán

- **Đầu vào:** ảnh và nhãn tương ứng từ bộ dữ liệu.
- **Đầu ra khi dự đoán một ảnh:** điểm số cho từng lớp; lớp có điểm số cao nhất là lớp dự đoán.
- **Đầu ra khi đánh giá:** test loss, test accuracy và bảng tổng hợp kết quả của các thí nghiệm.
- **Đầu ra của dự án:** mã nguồn có thể chạy lại, cấu hình thí nghiệm, TensorBoard logs/ảnh chụp biểu đồ, kết quả đánh giá, checkpoint mô hình được chọn và báo cáo ngắn.

Không được điền trước accuracy hay kết luận mô hình nào tốt hơn khi chưa chạy thí nghiệm.

## 3. Kiến thức và công cụ cần dùng

| Thành phần | Vai trò |
|---|---|
| Python | Ngôn ngữ triển khai |
| PyTorch | Xây dựng mô hình, loss, optimizer và training loop |
| torchvision | Tải mô hình/weights, dataset và biến đổi ảnh |
| Dataset / DataLoader | Quản lý dữ liệu và tạo batch |
| Transfer learning | Đóng băng backbone và học lớp phân loại mới |
| Fine-tuning | Mở một phần mô hình đã pre-train để tiếp tục cập nhật trọng số |
| TensorBoard | Theo dõi và so sánh loss/accuracy theo epoch |
| Git / GitHub | Quản lý phiên bản, review thay đổi và lưu tài liệu/mã nguồn |

Khi triển khai, ưu tiên API hiện hành của phiên bản `torchvision` được cài. Slide dùng cách viết `pretrained=True`; các phiên bản mới thường dùng tham số `weights=...` để chọn trọng số pre-trained.

## 4. Trình tự thực hiện từ đầu đến cuối

### Phase 0 — Chốt kế hoạch

- [x] Đọc và diễn giải yêu cầu Lab 2.
- [x] Xác định các đầu ra cần có.
- [ ] Xác nhận bộ dữ liệu và phạm vi thí nghiệm với nhóm/giảng viên nếu cần.
- [ ] Chốt môi trường chạy (máy cá nhân, GPU hoặc môi trường notebook).

**Điều kiện qua phase:** nhóm thống nhất dữ liệu, mô hình dự kiến, cách chia việc và cách lưu kết quả.

### Phase 1 — Thiết lập môi trường và kiểm tra dữ liệu

1. Kiểm tra Python, PyTorch, torchvision và TensorBoard.
2. Kiểm tra thiết bị chạy: GPU CUDA nếu khả dụng, nếu không thì CPU.
3. Tải dataset.
4. Tách dữ liệu thành train, validation và test:
   - Train: cập nhật trọng số.
   - Validation: lựa chọn cấu hình và checkpoint.
   - Test: đánh giá cuối cùng sau khi chọn cấu hình.
5. Xây dựng transforms và DataLoader.
6. Kiểm tra hình ảnh, nhãn, kích thước tensor và số lượng mẫu.

**Nguyên tắc:** không dùng tập test để điều chỉnh hyperparameter hoặc chọn mô hình. Với mô hình ImageNet pre-trained, cần xử lý kích thước và chuẩn hóa ảnh phù hợp với weights của mô hình; không mặc định mọi kiến trúc dùng cùng một transform.

### Phase 2 — Tải và khảo sát kiến trúc pre-trained

- Thử tối thiểu các họ mô hình khác nhau, dự kiến **VGG16** và **DenseNet121**.
- Tải weights pre-trained.
- In cấu trúc mô hình.
- Xác định lớp phân loại cuối, số lớp đầu ra và số tham số.
- Đếm số tham số được huấn luyện và số tham số bị đóng băng.
- Kiểm tra forward pass bằng một batch nhỏ trước khi train.

**Đầu ra phase:** ghi chú kiến trúc và xác nhận mô hình nhận được đầu vào, trả ra đúng số logits bằng số lớp của dataset.

### Phase 3 — Xây dựng baseline transfer learning

1. Thay lớp phân loại cuối bằng lớp có số đầu ra bằng số lớp dữ liệu.
2. Đóng băng backbone, chỉ huấn luyện lớp phân loại mới.
3. Chọn loss phù hợp; với phân loại đa lớp một nhãn, dự kiến dùng `CrossEntropyLoss`.
4. Chọn optimizer và learning rate ban đầu.
5. Viết training loop: forward pass → tính loss → backward pass → optimizer step.
6. Sau mỗi epoch, tính validation loss và validation accuracy.
7. Lưu checkpoint tốt nhất dựa trên tiêu chí validation đã thống nhất.

**Đầu ra phase:** baseline chạy hoàn chỉnh, có metric train/validation và checkpoint.

### Phase 4 — Theo dõi bằng TensorBoard

Dùng `SummaryWriter` để ghi ít nhất:

- Training loss theo epoch.
- Validation loss theo epoch.
- Training accuracy theo epoch.
- Validation accuracy theo epoch.
- Tên run thể hiện mô hình và cấu hình.

Mỗi thí nghiệm phải có tên/log directory riêng để so sánh. Xác minh rằng TensorBoard mở được và đọc đúng log.

**Đầu ra phase:** biểu đồ thể hiện tiến trình học và giúp nhận biết dấu hiệu overfitting.

### Phase 5 — Thực nghiệm mô hình và hyperparameter

Kế hoạch thử nghiệm ban đầu dưới đây là **đề xuất để triển khai**, không phải giá trị bắt buộc từ slide. Giá trị cuối cùng cần phù hợp với tài nguyên máy và thời gian.

| Run | Mục đích | Thay đổi chính | Các yếu tố nên giữ ổn định |
|---|---|---|---|
| A | Baseline VGG16 | Kiến trúc VGG16, chỉ train classifier | Dataset split, tiêu chí đánh giá và quy trình log |
| B | So sánh kiến trúc | DenseNet121, chỉ train classifier | Cùng split và quy trình đánh giá; dùng preprocessing đúng với weights |
| C | Thử learning rate | Thay learning rate của mô hình baseline được chọn | Giữ nguyên kiến trúc, batch size, cách freeze và số epoch |
| D | Thử batch size | Thay batch size | Giữ nguyên kiến trúc, learning rate và cách freeze |
| E | Fine-tuning | Mở một phần các lớp cuối của mô hình được chọn | So sánh với baseline cùng kiến trúc; ghi rõ số lớp được mở và learning rate |

Để giải thích kết quả có cơ sở, nên thay đổi từng yếu tố chính một lần trong các thí nghiệm C và D. Nếu thay nhiều yếu tố cùng lúc, sẽ khó xác định yếu tố nào tạo ra khác biệt.

Không cần chạy toàn bộ thí nghiệm dài ngay từ đầu: trước tiên chạy một smoke test ngắn để phát hiện lỗi, sau đó mới chạy đủ epoch đã thống nhất.

### Phase 6 — Đánh giá cuối cùng

- Chọn cấu hình dựa trên validation, không dựa vào test.
- Nạp checkpoint tốt nhất của cấu hình được chọn.
- Chạy đánh giá trên test đúng một quy trình nhất quán.
- Ghi test loss, test accuracy và thời gian chạy nếu đo được.
- Tổng hợp kết quả các run thành bảng.
- Phân tích ảnh hưởng của kiến trúc, learning rate, batch size và fine-tuning.
- Chỉ kết luận dựa trên số liệu thực tế; không khẳng định mô hình lớn hơn luôn tốt hơn.

### Phase 7 — Hoàn thiện repository và báo cáo

- Viết README hướng dẫn cài đặt và chạy lại.
- Lưu cấu hình từng run và bảng kết quả.
- Thêm ảnh chụp TensorBoard/biểu đồ cần thiết vào báo cáo.
- Kiểm tra code có thể chạy lại từ hướng dẫn.
- Kiểm tra Git diff, không commit dữ liệu tải về, môi trường ảo, cache hoặc checkpoint lớn ngoài kế hoạch.
- Hoàn tất review qua pull request nếu nhóm dùng quy trình nhánh.

## 5. Thiết kế thí nghiệm và tiêu chí so sánh

Mỗi run cần ghi lại tối thiểu:

- Run ID và kiến trúc.
- Tên/phiên bản weights pre-trained.
- Dataset và cách chia train/validation/test.
- Preprocessing/transforms.
- Batch size, learning rate, optimizer, epoch.
- Chiến lược freeze/fine-tune.
- Train loss, validation loss, train accuracy, validation accuracy.
- Test loss và test accuracy của cấu hình cuối được chọn.
- Thiết bị chạy và thời gian chạy nếu đo được.

### Bảng kết quả cuối dự kiến

| Run | Model | Learning rate | Batch size | Strategy | Best val accuracy | Test accuracy | Thời gian |
|---|---|---:|---:|---|---:|---:|---:|
| A | VGG16 | Chưa chạy | Chưa chạy | Frozen backbone | Chưa có | Chưa có | Chưa có |
| B | DenseNet121 | Chưa chạy | Chưa chạy | Frozen backbone | Chưa có | Chưa có | Chưa có |
| C | Mô hình được chọn | Chưa chạy | Cố định | Frozen backbone | Chưa có | Chưa có | Chưa có |
| D | Mô hình được chọn | Cố định | Chưa chạy | Frozen backbone | Chưa có | Chưa có | Chưa có |
| E | Mô hình được chọn | Chưa chạy | Cố định | Fine-tuning | Chưa có | Chưa có | Chưa có |

Các ô “Chưa có” sẽ chỉ được thay bằng số liệu sau khi chạy thật. Có thể điều chỉnh số run sau khi xác nhận yêu cầu của giảng viên và tài nguyên tính toán.

## 6. Đầu ra cuối cùng cần bàn giao

### A. Mã nguồn

- Script/notebook chuẩn bị dữ liệu.
- Script/notebook tải, sửa và khảo sát mô hình.
- Training/evaluation loop.
- TensorBoard logging.
- Cấu hình thí nghiệm và cách tái chạy.

### B. Bằng chứng thực nghiệm

- Bảng so sánh các run.
- Biểu đồ train/validation loss và accuracy.
- Kết quả đánh giá trên test.
- Một số dự đoán mẫu, nếu cần minh họa.
- Nhận xét về overfitting, ảnh hưởng hyperparameter và trade-off giữa chất lượng với thời gian chạy.

### C. Mô hình và tài liệu

- Checkpoint của cấu hình được chọn, nếu kích thước và chính sách lưu trữ cho phép.
- Báo cáo ngắn mô tả phương pháp, thí nghiệm, kết quả và kết luận.
- README hướng dẫn cài đặt, chạy và xem TensorBoard.

**Lưu ý về GitHub:** không nên commit trực tiếp các file weights/checkpoint hoặc TensorBoard event logs rất lớn nếu chúng làm repository nặng. Có thể lưu checkpoint ở nơi lưu trữ phù hợp và ghi hướng dẫn tải; trong Git nên giữ code, cấu hình, bảng kết quả và các ảnh minh họa cần thiết. Cần kiểm tra chính sách nộp bài trước khi quyết định.

## 7. Cấu trúc repository dự kiến

Đây là cấu trúc mục tiêu đề xuất, chưa phải các file đã được tạo:

```text
Lab2_DL_Group4/
├── README.md
├── LAB2_PLAN.md
├── requirements.txt
├── configs/
│   └── experiments/
├── notebooks/
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── reports/
│   ├── results.csv
│   └── figures/
├── runs/                 # TensorBoard logs; cân nhắc không commit
├── checkpoints/          # Model weights; cân nhắc không commit
└── .gitignore
```

Có thể bắt đầu bằng notebook để khám phá, sau đó tách thành các script nếu phù hợp với cách tổ chức nhóm. Không cần tạo toàn bộ cấu trúc ngay trong giai đoạn hoạch định.

## 8. GitHub team workflow đề xuất

1. Giữ nhánh `main` ở trạng thái ổn định.
2. Tạo nhánh riêng cho từng phần việc, ví dụ `docs/...`, `feature/data-pipeline`, `feature/model-training`.
3. Commit theo nội dung cụ thể, tránh gom mọi thứ vào một commit lớn.
4. Mở pull request, tự kiểm tra diff và nhờ thành viên khác review.
5. Chỉ merge khi code/tài liệu liên quan đã được kiểm tra.
6. Không đưa dữ liệu lớn, secrets, token, môi trường ảo hay file cache vào Git.

## 9. Checklist hoàn thành Lab 2

- [ ] Đã xác nhận dataset và môi trường.
- [ ] Đã tải ít nhất hai kiến trúc pre-trained khác nhau.
- [ ] Đã in và giải thích cấu trúc, lớp phân loại và số tham số.
- [ ] Đã thay lớp cuối theo số lớp dữ liệu.
- [ ] Đã thực hiện transfer learning.
- [ ] Đã thực hiện thử nghiệm thay hyperparameter.
- [ ] Đã thực hiện hoặc đánh giá fine-tuning theo kế hoạch.
- [ ] Đã log và xem TensorBoard.
- [ ] Đã chọn mô hình bằng validation.
- [ ] Đã đánh giá cấu hình cuối trên test.
- [ ] Đã tổng hợp bảng kết quả và nhận xét.
- [ ] Đã lưu mã nguồn, hướng dẫn chạy và đầu ra cần bàn giao.
- [ ] Đã kiểm tra repository và quy trình pull request.

## 10. Những điểm cần chốt trước khi bắt đầu code

1. Có dùng CIFAR-10 như ví dụ trong slide hay giảng viên chỉ định dataset khác?
2. Nhóm chạy trên GPU máy cá nhân, Google Colab hay môi trường khác?
3. Giảng viên có yêu cầu cụ thể về số mô hình, số epoch, báo cáo hoặc định dạng nộp không?
4. Nhóm muốn ưu tiên notebook dễ trình bày hay scripts có cấu trúc để tái sử dụng?

Sau khi chốt các điểm này, bước tiếp theo là thiết lập môi trường và hoàn thiện **data pipeline** trước khi triển khai mô hình.
