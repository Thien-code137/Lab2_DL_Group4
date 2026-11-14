# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) VÀ PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA) TRÊN BỘ DỮ LIỆU CIFAR-10
## Dự án: Deep Learning Lab 2 — Pre-trained Neural Network Architectures (Group 4)
*Căn cứ thực nghiệm: Trích xuất từ quá trình thực thi mã nguồn notebook `notebooks/03_eda_pca.ipynb` và các tệp trực quan hóa trong `reports/figures/`.*

---

## 1. TỔNG QUAN VÀ MỤC TIÊU PHÂN TÍCH

Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA) và Phân tích Thành phần Chính (Principal Component Analysis - PCA) đóng vai trò là bước thẩm định cơ sở, cung cấp cái nhìn định lượng sâu sắc về cấu trúc phân phối, đặc tính không gian màu, tính cân bằng dữ liệu và giới hạn biểu diễn của không gian pixel thô trước khi đưa vào các mô hình học sâu tiền huấn luyện (Pre-trained Deep Neural Networks).

### Mục tiêu cụ thể:
1. **Kiểm chứng dữ liệu (Data Validation):** Xác nhận số lượng mẫu, kích thước, phân bố lớp, tính cân bằng giữa tập Train/Validation/Test, và kiểm tra triệt để nguy cơ rò rỉ dữ liệu (Zero Data Leakage).
2. **Khảo sát đặc trưng phổ màu (Color & Intensity Analysis):** Đánh giá phân bố cường độ pixel, độ sáng trung bình (Luma), màu sắc đặc trưng của từng lớp và so sánh với bộ thông số chuẩn hóa ImageNet.
3. **Đánh giá tác động tiền xử lý (Preprocessing Overhead):** Phân tích chi phí tính toán và mức tăng dung lượng bộ nhớ khi phóng to ảnh từ $32 \times 32$ lên $224 \times 224$ phục vụ mạng VGG16 và DenseNet121.
4. **Phân tích cấu trúc thấp chiều với PCA:** Khảo sát phương sai giải thích tích lũy, các hướng trực giao chính (Eigen-images), chất lượng nén/tái tạo ảnh và mức độ tách cụm trong không gian 2D.
5. **Xây dựng đường cơ sở phi học sâu (Non-Deep Baseline):** Huấn luyện thuật toán $k$-NN trên không gian vector PCA để đối chiếu hiệu năng với các mạng nơ-ron tích chập sâu (CNN).

---

## 2. PHẦN 1: KHÁM PHÁ DỮ LIỆU (EDA — EXPLORATORY DATA ANALYSIS)

### 2.1. Tổng quan kích thước và kiểu dữ liệu

Bộ dữ liệu CIFAR-10 gồm 60.000 ảnh màu RGB thuộc 10 lớp đối tượng riêng biệt: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

| Tập dữ liệu | Số lượng ảnh | Shape tensor gốc | Kiểu dữ liệu (dtype) | Giá trị pixel Min | Giá trị pixel Max |
|---|---:|:---:|:---:|---:|---:|
| **Train gốc** | 50.000 | `(50000, 32, 32, 3)` | `uint8` | 0 | 255 |
| **Test** | 10.000 | `(10000, 32, 32, 3)` | `uint8` | 0 | 255 |

*Nhận xét:* Dữ liệu thô có khoảng giá trị nguyên chuẩn tắc $[0, 255]$, không có giá trị bất thường (out-of-range) hoặc giá trị khuyết thiếu (NaN/Null).

---

### 2.2. Phân bố số ảnh theo lớp (Class Distribution)

Phân bố số lượng ảnh giữa các lớp được thống kê chính xác:

| Tên lớp (Class Name) | Số ảnh tập Train | Số ảnh tập Test | Tỷ lệ trong tập (%) |
|---|---:|---:|:---:|
| `airplane` (máy bay) | 5.000 | 1.000 | 10.0% |
| `automobile` (ô tô) | 5.000 | 1.000 | 10.0% |
| `bird` (chim) | 5.000 | 1.000 | 10.0% |
| `cat` (mèo) | 5.000 | 1.000 | 10.0% |
| `deer` (hươu) | 5.000 | 1.000 | 10.0% |
| `dog` (chó) | 5.000 | 1.000 | 10.0% |
| `frog` (ếch) | 5.000 | 1.000 | 10.0% |
| `horse` (ngựa) | 5.000 | 1.000 | 10.0% |
| `ship` (tàu thủy) | 5.000 | 1.000 | 10.0% |
| `truck` (xe tải) | 5.000 | 1.000 | 10.0% |
| **Tổng cộng** | **50.000** | **10.000** | **100.0%** |

![Phân bố số ảnh mỗi lớp](reports/figures/eda_class_distribution.png)

*Nhận xét:* CIFAR-10 có tính cân bằng lớp hoàn hảo (Perfect Balance). Mô hình huấn luyện trên tập dữ liệu này sẽ không bị thiên kiến (bias) do chênh lệch tần suất xuất hiện giữa các lớp.

---

### 2.3. Kiểm tra phép chia Train / Validation (Validation Split Verification)

Theo quy ước dự án (`src/data.py`), tập 50.000 ảnh train được chia thành **45.000 ảnh Train** và **5.000 ảnh Validation** ($10\%$) bằng hàm `torch.utils.data.random_split` với `seed = 42`. Do đây là phép chia ngẫu nhiên thuần túy (không phân tầng - non-stratified), số lượng mẫu mỗi lớp trong tập validation có dao động nhẹ quanh mức kỳ vọng đều 500 ảnh/lớp ($10.0\%$):

| Tên lớp | Số ảnh Train (45k) | Số ảnh Val (5k) | Tỷ lệ Val trong lớp (%) | Chênh lệch so với kỳ vọng (500) |
|---|---:|---:|---:|:---:|
| `airplane` | 4.512 | 488 | 9.76% | -12 ảnh (-2.4%) |
| `automobile` | 4.488 | 512 | 10.24% | +12 ảnh (+2.4%) |
| `bird` | 4.468 | 532 | 10.64% | +32 ảnh (+6.4%) |
| `cat` | 4.529 | 471 | 9.42% | -29 ảnh (-5.8%) |
| `deer` | 4.529 | 471 | 9.42% | -29 ảnh (-5.8%) |
| `dog` | 4.486 | 514 | 10.28% | +14 ảnh (+2.8%) |
| `frog` | 4.493 | 507 | 10.14% | +7 ảnh (+1.4%) |
| `horse` | 4.500 | 500 | 10.00% | 0 ảnh (0.0%) |
| `ship` | 4.496 | 504 | 10.08% | +4 ảnh (+0.8%) |
| `truck` | 4.499 | 501 | 10.02% | +1 ảnh (+0.2%) |

![Phân bố lớp trong tập validation](reports/figures/eda_val_split_distribution.png)

*Nhận xét:*
- Lớp có số lượng ảnh validation lớn nhất là `bird` (532 ảnh), nhỏ nhất là `cat` và `deer` (471 ảnh).
- Độ lệch cực đại chỉ nằm trong khoảng $[-29, +32]$ mẫu (tương đương biên độ $\pm 6.4\%$ so với kỳ vọng 500). Mức chênh lệch này rất nhỏ và không làm sai lệch tính khách quan khi sử dụng tập validation để theo dõi hàm mất mát và lựa chọn checkpoint tối ưu.
- Kiểm tra tính giao thoa tập chỉ số: $|Train \cap Val| = 0$ (hoàn toàn tách biệt).

---

### 2.4. Khảo sát trực quan ảnh mẫu theo từng lớp

Hình ảnh bên dưới thể hiện 8 mẫu ngẫu nhiên từ mỗi lớp trong tập huấn luyện:

![Ảnh mẫu từng lớp](reports/figures/eda_samples_per_class.png)

*Nhận xét đặc điểm thị giác:*
- Độ phân giải gốc $32 \times 32$ pixel rất thấp, khiến mắt thường khó nhận diện các đặc trưng vi mô chi tiết.
- Độ biến thiên nội lớp (Intra-class variance) rất lớn: góc chụp, hướng quay, kích thước chủ thể thay đổi mạnh.
- Bối cảnh nền (Background clutter) phức tạp: máy bay có thể bay trên trời xanh hoặc đỗ trên đường băng màu xám; tàu thuyền nổi trên biển xanh hoặc neo đậu ở cảng; động vật thường hòa lẫn vào đồng cỏ, cành cây hoặc thảm trong nhà.

---

### 2.5. Thống kê kênh màu RGB và so sánh chuẩn hóa ImageNet

Thống kê cường độ trung bình (Mean) và độ lệch chuẩn (Std) trên thang $[0, 1]$ được tính toán chính xác trên toàn bộ tập dữ liệu:

| Kênh màu | CIFAR-10 Train Mean | CIFAR-10 Test Mean | ImageNet Mean chuẩn | CIFAR-10 Train Std | CIFAR-10 Test Std | ImageNet Std chuẩn |
|:---:|---:|---:|---:|---:|---:|---:|
| **Đỏ (R)** | **0.4914** | 0.4942 | 0.4850 | **0.2470** | 0.2467 | 0.2290 |
| **Lục (G)** | **0.4822** | 0.4851 | 0.4560 | **0.2435** | 0.2429 | 0.2240 |
| **Lam (B)** | **0.4465** | 0.4504 | 0.4060 | **0.2616** | 0.2616 | 0.2250 |

![Phân bố cường độ pixel theo kênh](reports/figures/eda_pixel_histogram.png)

*Nhận xét:*
1. **Tính tương đồng Train - Test:** Giá trị trung bình và độ lệch chuẩn của tập Train và Test gần như trùng khớp hoàn toàn (độ lệch $< 0.004$). Điều này đảm bảo phân phối mẫu giữa hai tập là đồng nhất (i.i.d).
2. **So sánh với ImageNet:** Kênh G và B của CIFAR-10 sáng hơn ImageNet từ $0.026$ đến $0.040$. Độ phân tán màu (Std) của CIFAR-10 rộng hơn ImageNet khoảng $0.018 - 0.036$.
3. **Ý nghĩa Transfer Learning:** Dù CIFAR-10 có sự chênh lệch phân phối nhẹ so với ImageNet, pipeline vẫn bắt buộc phải áp dụng phép chuẩn hóa ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Việc này giúp căn chỉnh không gian giá trị đầu vào phù hợp tuyệt đối với các bộ lọc tích chập (conv filters) đã học trước của VGG16 và DenseNet121.

---

### 2.6. Độ sáng trung bình (Luma) và Ảnh trung bình theo lớp

Độ sáng luma của mỗi ảnh được tính theo công thức chuẩn truyền hình:
$$\text{Luma} = 0.299 \times R + 0.587 \times G + 0.114 \times B$$

| Tên lớp (Class) | Cường độ R trung bình | Cường độ G trung bình | Cường độ B trung bình | Độ sáng Luma trung bình |
|---|---:|---:|---:|---:|
| **`airplane`** | 134.04 | 142.88 | 150.17 | **141.07** (Sáng nhất) |
| **`ship`** | 125.01 | 133.98 | 141.44 | **132.15** (Sáng nhì) |
| `truck` | 127.16 | 123.76 | 121.91 | 124.57 |
| `bird` | 124.76 | 125.33 | 108.13 | 123.20 |
| `horse` | 128.00 | 122.37 | 106.31 | 122.22 |
| `dog` | 127.48 | 118.48 | 106.22 | 119.77 |
| `cat` | 126.35 | 116.39 | 105.96 | 118.18 |
| `automobile` | 120.15 | 115.91 | 114.04 | 116.96 |
| `deer` | 120.26 | 118.63 | 96.44 | 116.59 |
| **`frog`** | 119.86 | 111.79 | 88.03 | **111.50** (Tối nhất) |

![Phân bố độ sáng và ảnh trung bình](reports/figures/eda_brightness_mean_images.png)

*Nhận xét:*
- **Các lớp sáng nổi bật:** `airplane` (141.07) và `ship` (132.15). Cả hai lớp này đều có giá trị kênh Blue cao vượt trội ($>140$) do ảnh chụp ngoài trời với nền trời xanh hoặc mặt nước biển bao quanh.
- **Các lớp tối đặc trưng:** `frog` (111.50) và `deer` (116.59) có kênh Blue thấp nhất ($<97$), do ảnh chủ yếu được chụp trong môi trường rừng cây, đầm lầy hoặc thảm thực vật màu xanh lục và nâu sẫm.
- **Ảnh trung bình (Mean Images):** Khi lấy trung bình cộng hàng ngàn ảnh trong cùng một lớp, ta thấy rõ cấu trúc phông nền: lớp `ship` có dải xanh dương ngang đại diện cho đường chân trời biển; lớp `airplane` sáng đều khắp khung hình; các lớp động vật có vùng tối mờ hội tụ ở trung tâm đại diện cho vị trí chủ thể.

---

### 2.7. Thẩm định ảnh trùng lặp và kiểm toán rò rỉ dữ liệu (Zero Leakage Audit)

Nhóm đã thực hiện kiểm toán tính độc lập dữ liệu bằng thuật toán băm mã hóa MD5 hash trên từng chuỗi byte của 60.000 ảnh:

- Số lượng ảnh trùng lặp nội bộ trong tập Train: **0 ảnh** ($0\%$).
- Số lượng ảnh trùng lặp nội bộ trong tập Test: **0 ảnh** ($0\%$).
- Số lượng ảnh xuất hiện đồng thời ở cả tập Train và tập Test: **0 ảnh** ($0\%$).

*Kết luận kiểm toán:* Bộ dữ liệu CIFAR-10 được cách ly hoàn hảo, **đáp ứng 100% nguyên tắc Zero Data Leakage** ở cấp độ hình ảnh.

---

### 2.8. Tác động của phép biến đổi phóng ảnh ($32 \times 32 \rightarrow 224 \times 224$)

Để tương thích với các kiến trúc mô hình pre-trained (VGG16, DenseNet121) vốn nhận kích thước ảnh chuẩn ImageNet ($224 \times 224$), pipeline thực hiện nội suy phóng đại ảnh:

![Minh họa phóng đại ảnh](reports/figures/eda_resize_effect.png)

*Phân tích chi phí tính toán:*
- **Tỷ lệ tăng điểm ảnh:**
  $$\text{Tỷ lệ} = \frac{224 \times 224}{32 \times 32} = 49\text{ lần}$$
- **Dung lượng bộ nhớ tensor `float32` cho 1 ảnh:**
  - Kích thước gốc ($32 \times 32 \times 3$): $3 \times 32 \times 32 \times 4\text{ bytes} \approx \mathbf{12.0\text{ KB}}$
  - Kích thước sau resize ($224 \times 224 \times 3$): $3 \times 224 \times 224 \times 4\text{ bytes} \approx \mathbf{588.0\text{ KB}}$
- **Hệ quả đối với tài nguyên thực nghiệm:**
  - Một epoch huấn luyện 45.000 ảnh kích thước $224 \times 224$ đòi hỏi xử lý lưu lượng tensor đầu vào lên tới $\approx \mathbf{26.46\text{ GB}}$ dữ liệu.
  - Phép biến đổi nội suy Bilinear làm tăng số phép toán dấu phẩy động (FLOPs) của các tầng tích chập đầu mạng lên gấp hàng chục lần.
  - Đây chính là nguyên nhân khách quan trực tiếp giải thích vì sao thời gian huấn luyện 1 epoch trên CPU kéo dài từ **1.5 giờ (DenseNet121)** đến **hơn 6 giờ (VGG16)** trong các thí nghiệm thực tế của nhóm.

---

## 3. PHẦN 2: PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA — PRINCIPAL COMPONENT ANALYSIS)

PCA được thực thi trên tập con 10.000 ảnh (lấy mẫu phân tầng theo lớp từ tập train gốc), sau đó chia thành 8.000 ảnh fit PCA và 2.000 ảnh kiểm định độc lập. Mỗi ảnh được làm phẳng (flatten) thành một vector đặc trưng thô $32 \times 32 \times 3 = \mathbf{3.072\text{ chiều}}$.

### 3.1. Phương sai giải thích tích lũy (Explained Variance Ratio)

Khảo sát tỷ lệ phương sai tích lũy của các thành phần chính:

| Ngưỡng phương sai giải thích | Số thành phần chính cần thiết ($k$) | Tỷ lệ số chiều so với không gian gốc ($3.072$) |
|:---:|---:|:---:|
| **80%** | **39** | **1.3%** |
| **90%** | **99** | **3.2%** |
| **95%** | **206** | **6.7%** |
| **99%** | **663** | **21.6%** |

![Phương sai giải thích tích lũy](reports/figures/pca_explained_variance.png)

*Nhận xét:*
- Chỉ cần **39 thành phần** (tương đương $1.3\%$ số chiều) đã nén giữ được $80\%$ lượng thông tin phương sai của ảnh.
- Để giữ lại $90\%$ thông tin, ta chỉ cần **99 thành phần** (khoảng $3.2\%$ số chiều).
- Biểu đồ Scree plot cho thấy phương sai giảm dốc đứng ở 10 thành phần đầu tiên, sau đó thoải dần. Điều này chứng minh các pixel lân cận trong ảnh tự nhiên có tính phụ thuộc và tương quan không gian cực kỳ mạnh mẽ.

---

### 3.2. Trực quan hóa các thành phần chính (Eigen-images)

10 thành phần chính đầu tiên được định hình lại (reshape) về kích thước $(32, 32, 3)$ và chuẩn hóa hiển thị:

![10 thành phần chính đầu tiên](reports/figures/pca_eigen_images.png)

*Phân tích ngữ nghĩa không gian:*
- **PC1 (chiếm 28.5% phương sai):** Đại diện cho mức độ chiếu sáng đồng nhất toàn cục (Global Illumination) và cân bằng trắng. Vector này phân biệt giữa ảnh sáng (trời, biển) và ảnh tối (rừng cây, nội thất).
- **PC2 (chiếm 12.0% phương sai):** Nắm bắt gradient chuyển sắc theo chiều dọc (nửa trên sáng tương ứng bầu trời, nửa dưới tối tương ứng mặt đất) cùng độ lệch giữa gam màu ấm (đỏ/vàng) và gam màu lạnh (xanh dương).
- **PC3 đến PC10:** Chuyển dịch dần từ biến thiên màu diện rộng sang phát hiện các đường biên ngang, biên dọc, vi phân tần số cao và đối xứng tâm.

---

### 3.3. Biểu diễn không gian chiếu 2D (2D Scatter Plot)

Hình ảnh phân tán dữ liệu khi chiếu các ảnh lên mặt phẳng 2D (PC1 vs PC2, và PC3 vs PC4):

![Chiếu dữ liệu lên không gian 2D](reports/figures/pca_scatter_2d.png)

*Nhận xét giới hạn của phép chiếu tuyến tính:*
- Các điểm dữ liệu thuộc 10 lớp bị **chồng lấn rất nặng nề (severe overlap)** tạo thành một cụm mây điểm dày đặc ở gốc tọa độ.
- Chỉ có sự phân tách yếu dựa theo màu sắc môi trường:
  - Lớp `airplane` và `ship` lệch về phía dải giá trị dương do màu nền trời/biển sáng.
  - Lớp `frog` và `deer` phân tán về phía dải giá trị âm do nền đất/rừng tối.
- Các lớp có hình thái khác nhau nhưng chia sẻ gam màu (ví dụ: `cat` vs `dog` vs `horse`, hoặc `automobile` vs `truck`) hoàn toàn đan xen lẫn nhau.
- *Kết luận:* Không gian tuyến tính 2D của pixel thô **hoàn toàn bất lực trong việc phân tách ranh giới quyết định (decision boundary)** cho bài toán phân loại 10 lớp.

---

### 3.4. Đánh giá chất lượng tái tạo ảnh (Image Reconstruction)

Ảnh từ tập kiểm định được tái tạo ngược từ $k$ thành phần chính:
$$\hat{X} = Z_k \cdot W_k^T + \mu$$

| Số thành phần ($k$) | Sai số bình phương trung bình (MSE trên tập Val) | Tỷ lệ phương sai tích lũy | Chất lượng cảm nhận thị giác |
|:---:|---:|:---:|---|
| **$k = 5$** | 0.02830 | 54.45% | Ảnh nhòe mờ hoàn toàn, chỉ nhận diện được mảng màu nền đại thể. |
| **$k = 20$** | 0.01592 | 74.59% | Xuất hiện bóng mờ phác thảo hình dạng khối của đối tượng trung tâm. |
| **$k = 50$** | 0.00984 | 84.50% | Nhận diện được đường nét cơ bản của xe cộ, thân hình con vật. |
| **$k = 100$** | 0.00627 | 90.33% | Chi tiết vật thể hiện rõ, biên cạnh và màu sắc tương đối sắc nét. |
| **$k = 200$** | 0.00353 | 94.82% | Phục hồi gần như đầy đủ các chi tiết quan trọng của ảnh gốc. |
| **$k = 500$** | **0.00115** | **98.51%** | Tái tạo hoàn hảo, mắt thường không phân biệt được với ảnh gốc $32 \times 32$. |

![Tái tạo ảnh validation bằng k thành phần chính](reports/figures/pca_reconstruction.png)

---

### 3.5. Đường cơ sở phân loại k-NN trên không gian PCA

Nhóm xây dựng mô hình phân loại cổ điển $k$-NN ($k = 5$ láng giềng gần nhất) trên các không gian vector PCA rút gọn để thiết lập đường cơ sở đối chuẩn:

| Số thành phần PCA ($k$) | Độ chính xác trên tập Held-out (Accuracy) | Nhận xét xu hướng |
|:---:|---:|---|
| $k = 2$ | 0.1540 (15.40%) | Thấp, chỉ nhỉnh hơn đoán ngẫu nhiên (10%) một chút do mất quá nhiều thông tin. |
| $k = 5$ | 0.2290 (22.90%) | Tăng nhanh khi có thêm thông tin màu sắc nền. |
| $k = 10$ | 0.3185 (31.85%) | Cải thiện đáng kể khi bổ sung các biên cạnh tổng quát. |
| **$k = 20$** | **0.3505 (35.05%)** | **ĐẠT ĐỈNH CỰC ĐẠI TOÀN CỤC.** |
| $k = 50$ | 0.3260 (32.60%) | Bắt đầu suy giảm do nhiễu chiều cao. |
| $k = 100$ | 0.3150 (31.50%) | Suy giảm tiếp tục. |
| $k = 200$ | 0.3035 (30.35%) | Tốc độ suy luận chậm đi đáng kể. |
| $k = 500$ | 0.2970 (29.70%) | Bị ảnh hưởng mạnh bởi lời nguyền số chiều. |
| $k = 1.000$ | 0.2955 (29.55%) | Khoảng cách Euclidean bị bão hòa. |
| **$k = 3.072$ (Pixel gốc)** | **0.2930 (29.30%)** | Kém hơn $k=20$ tới 5.75 điểm phần trăm do nhiễu pixel cục bộ. |

![Độ chính xác kNN theo số chiều PCA](reports/figures/pca_knn_accuracy.png)

*Hiện tượng kỹ thuật quan trọng:*
Độ chính xác $k$-NN đạt đỉnh tại **$k = 20$ thành phần** ($35.05\%$), sau đó giảm dần khi tăng số chiều. Điều này minh họa rõ nét **"Lời nguyền số chiều" (Curse of Dimensionality)** trong không gian vector: khi số chiều tăng lên, khoảng cách Euclidean giữa các điểm dữ liệu có xu hướng trở nên đồng đều, khiến khái niệm "láng giềng gần nhất" mất dần ý nghĩa phân loại và dễ bị chi phối bởi các nhiễu hạt ngẫu nhiên ở tần số cao.

---

### 3.6. Bảng so sánh đối chuẩn: k-NN trên PCA vs Mô hình Pre-trained Deep Learning

Đặt kết quả tốt nhất của phương pháp phân loại truyền thống bên cạnh kết quả thực nghiệm huấn luyện của nhóm trong Lab 2:

| Mô hình / Phương pháp | Không gian biểu diễn | Phương pháp trích xuất đặc trưng | Độ chính xác (Accuracy) | Khoảng cách so với Deep Learning |
|---|---|---|---:|:---:|
| **Ngẫu nhiên (Random Guess)** | Không gian nhãn rời rạc | Không có | 10.00% | - |
| **k-NN trên Pixel gốc ($3.072$ chiều)** | Pixel thô | Căn tâm toàn cục | 29.30% | Cơ sở |
| **k-NN trên PCA ($k = 20$)** | Không gian vector trực giao | Phép chiếu tuyến tính (Linear SVD) | **35.05%** | +5.75% so với pixel gốc |
| **E1: VGG16 (`classifier_only`)** | Feature Maps tích chập sâu | Deep Convolutional Backbone (ImageNet) | **82.88%** | **+47.83% so với PCA** |
| **E3: DenseNet121 (`classifier_only`, LR=1e-4)** | Feature Maps kết nối dày đặc | Dense Connectivity (ImageNet) | **83.44%** | **+48.39% so với PCA** |
| **E4: DenseNet121 (`classifier_only`, Batch=64)** | Feature Maps kết nối dày đặc | Dense Connectivity (ImageNet) | **87.16%** | **+52.11% so với PCA** |
| **E2: DenseNet121 (`classifier_only`, LR=1e-3)** | Feature Maps kết nối dày đặc | Dense Connectivity (ImageNet) | **87.84%** | **+52.79% so với PCA** |
| **E5: DenseNet121 (`finetune_last_layers`)** | Deep Features + Fine-tuned Blocks | DenseBlock4 + Norm5 + Classifier | **92.10%** | **+57.05% so với PCA** |

---

## 4. BÀI HỌC VÀ Ý NGHĨA KỸ THUẬT (KEY TAKEAWAYS)

1. **Giới hạn cố hữu của phương pháp tuyến tính:**
   PCA trên pixel thô chỉ tìm kiếm các trục có phương sai cực đại dựa trên ma trận hiệp phương sai tuyến tính. Tuy nhiên, trong thị giác máy tính, hai ảnh chụp cùng một vật thể (ví dụ con mèo ở góc nghiêng khác nhau hoặc dưới điều kiện ánh sáng khác nhau) có thể có khoảng cách Euclidean trong không gian pixel thô rất xa nhau, trong khi hai ảnh khác lớp có cùng phông nền lại có khoảng cách rất gần. Do đó, PCA không thể tạo ra các đặc trưng mang tính ngữ nghĩa trừu tượng.

2. **Sức mạnh vượt trội của Convolutional Neural Networks (CNN):**
   Các mô hình Deep Learning (VGG16 đạt $82.88\%$, DenseNet121 đạt tới $92.10\%$) vượt trội hoàn toàn so với k-NN PCA ($35.05\%$) nhờ:
   - Khả năng bất biến cục bộ đối với tịnh tiến (Translation Invariance) thông qua các lớp tích chập và pooling.
   - Cơ chế học biểu diễn phân cấp phi tuyến (Hierarchical Non-linear Feature Learning): từ cạnh biên, kết cấu bề mặt, hình khối bộ phận cho đến toàn thể đối tượng.

3. **Giá trị của Transfer Learning và Fine-tuning:**
   Mô hình tiền huấn luyện trên ImageNet đã tích lũy hàng triệu bộ lọc thị giác tổng quát chất lượng cao. Khi áp dụng vào CIFAR-10, việc mở fine-tuning các tầng sâu nhất (`denseblock4` và `norm5` trong E5) giúp mô hình điều chỉnh lại các đặc trưng bậc cao cho phù hợp với kích thước và miền dữ liệu mới, mang lại độ chính xác vượt bậc đạt **$92.10\%$ validation accuracy** và **$91.81\%$ test accuracy**.

4. **Tính nhất quán của quy trình kỹ thuật:**
   Báo cáo EDA và PCA này khẳng định toàn bộ quy trình tiền xử lý, kiểm soát rò rỉ dữ liệu, chuẩn hóa ImageNet và cấu hình mô hình của nhóm 4 là chuẩn mực, minh bạch và hoàn toàn nhất quán với kết quả thực nghiệm đã công bố.
