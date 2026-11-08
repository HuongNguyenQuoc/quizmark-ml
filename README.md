# quizmark-ml

Model phân loại ô trắc nghiệm (**trống / tô / tẩy**) cho app [QuizMark](https://github.com/leduchuy11/QuizMark).

Trong app, OpenCV vẫn làm phần hình học: tìm 4 góc, nắn phẳng phiếu, cắt từng ô. Model này chỉ thay **một việc**:
quyết định ô có được tô hay không. Trước đây việc đó dùng ngưỡng cố định 50% điểm ảnh tối, nên dễ sai khi bút chì nhạt, tẩy không sạch hoặc ảnh bị bóng.

Model là một CNN rất nhỏ (khoảng 27 nghìn tham số, file ONNX khoảng 100 KB), chạy offline trong app bằng OpenCV DNN.

## Quy trình

```
App (bản debug)                    Google Drive                      Colab (notebook)                 App
Tài khoản → bật "Thu dữ liệu ô tô"
quét phiếu mẫu 1–5          →  quizmark-ml/data/*.zip     →  sửa nhãn → train → export   →  assets/models/*.onnx
"Xuất dữ liệu (zip) lên Drive"  quizmark-ml/labels_override.csv   kiểm tra bằng OpenCV 4.5.3
                                quizmark-ml/models/*.onnx  ←─────────────┘
```

1. **Thu dữ liệu**: trong app bản debug, vào Tài khoản → "Công cụ dữ liệu (chỉ bản debug)".
2. **Train**: mở [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) trên Google Colab rồi chạy lần lượt từng ô.
3. **Đưa vào app**: chép `bubble_classifier_v{N}.onnx` vào `QuizMark/app/src/main/assets/models/` và chép một bản vào thư mục `models/` của repo này.

Chưa có dữ liệu thật thì đặt `USE_SYNTHETIC = True` trong notebook để chạy thử bằng dữ liệu giả.

## Quy ước giữa repo này và app

**Đổi bất kỳ điều nào dưới đây thì phải sửa cả code Kotlin và tăng phiên bản model.** Các giá trị nằm trong [`src/config.py`](src/config.py).

| | Giá trị |
|---|---|
| Input | Ô cắt từ **ảnh xám** (`grayMat`, chưa vẽ vòng màu) → resize 32×32 bằng `INTER_AREA` → float32 chia 255. Shape `[1, 1, 32, 32]` (NCHW). |
| Output | 3 logit theo thứ tự `[empty, filled, erased]`. Softmax ra xác suất. `erased` = **không tô**. |
| Định dạng | ONNX opset 11, chỉ gồm `Conv, Relu, MaxPool, Flatten, Gemm` (BatchNorm đã gộp vào Conv). |
| Tên file | `bubble_classifier_v{N}.onnx` |

Phía Kotlin xử lý tương ứng:
```kotlin
Imgproc.resize(grayCell, resized, Size(32.0, 32.0), 0.0, 0.0, Imgproc.INTER_AREA)
val blob = Dnn.blobFromImage(resized, 1.0 / 255.0)
net.setInput(blob); val logits = net.forward()
```

## Dữ liệu

App xuất file zip có cấu trúc:
```
bubble_dataset/<empty|filled>/<sheetId>_t<template>_<vùng>_<vị trí>_<lựa chọn>.png
```
- Thư mục `empty|filled` là **nhãn tạm** theo ngưỡng 50% cũ. Nhãn đúng được sửa trong notebook và lưu ở `labels_override.csv` trên Drive.
- `sheetId` (ví dụ `261108-142233-k3fz`) bắt đầu bằng thời gian quét, mỗi lần quét một mã khác nhau.
- Tập train / val / test được **chia theo lần quét** để độ chính xác đo được sát với thực tế.
- Ảnh và nhãn chỉ nằm trên Google Drive, **không commit vào git**.

## Cấu trúc

| File | Vai trò |
|---|---|
| `src/config.py` | Quy ước chung (lớp, kích thước input, opset) |
| `src/data.py` | Giải nén zip, đọc mẫu, áp nhãn sửa tay, chia tập, tiền xử lý, augmentation |
| `src/model.py` | `BubbleNet` |
| `src/train.py` | Train, đánh giá, so sánh với ngưỡng 50% cũ |
| `src/review.py` | Xem & sửa nhãn trong Colab (lưới ảnh có đánh số, mẫu đáng ngờ) |
| `src/export_onnx.py` | Export ONNX (exporter của PyTorch, hoặc tự dựng nếu exporter đó bị gỡ), test vectors, model card |
| `src/verify_onnx.py` | Kiểm tra file ONNX bằng **đúng OpenCV 4.5.3** (Python 3.9, chạy độc lập) |
| `src/synthetic.py` | Dữ liệu giả để chạy thử |
| `notebooks/train_colab.ipynb` | Notebook chạy trên Colab |

## Chạy trên máy (không bắt buộc)

```bash
pip install -r requirements.txt
# Kiểm tra bằng OpenCV 4.5.3 (cần Python 3.9; uv tự tải về):
uv venv .venv-cv453 --python 3.9
uv pip install --python .venv-cv453 "opencv-python-headless==4.5.3.56" "numpy<2"
.venv-cv453/Scripts/python src/verify_onnx.py --model out/bubble_classifier_v1.onnx --vectors out/testvectors.npz
```
