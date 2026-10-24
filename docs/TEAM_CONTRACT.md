# Quy ước giao diện module — đọc trước khi code

Đây là giao diện logic dự kiến. Các thành viên cần thống nhất chữ ký hàm chính thức trong PR đầu tiên trước khi các module phụ thuộc vào nhau.

## TV1 — `src/data.py`
Dự kiến:
- `build_dataloaders(config)` → dict có `train`, `val`, `test` loaders và metadata lớp.
- Ghi rõ split seed, transforms, normalization, batch size và num_workers.
- Không dùng test để tune model.

## TV2 — `src/models.py`
Dự kiến:
- `build_model(config)` → model.
- Output logits shape `[batch_size, 10]`.
- Cho phép chọn kiến trúc, pretrained weights, strategy freeze/fine-tune.
- Ghi total/trainable parameters.
- Không đặt Softmax trước `CrossEntropyLoss`.

## TV3 — `src/train.py`
Dự kiến:
- `train_model(model, dataloaders, config)` → history và metadata checkpoint.
- Ghi train/validation loss, accuracy theo epoch.
- Chọn checkpoint theo validation metric đã thống nhất.
- Ghi log TensorBoard phối hợp TV4.

## TV4 — `src/evaluate.py`
Dự kiến:
- `evaluate_model(model, test_loader, config)` → dict metrics.
- Chỉ đánh giá test sau khi chọn model.
- Không dùng test metrics để điều chỉnh cấu hình.

## TV5 — `configs/` và CSV kết quả
- Mỗi run có tên duy nhất và cấu hình cụ thể.
- Kết quả phải gắn với đúng config.
- Chỉ điền giá trị đã đo.

## TV6 — `main.py`
- Điều phối data → model → train/validation → chọn checkpoint → test → ghi kết quả.
- Không sao chép logic từ các module khác.
- README commands phải khớp với giao diện đã triển khai.
