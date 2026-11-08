"""Kiểm tra file ONNX bằng ĐÚNG OpenCV 4.5.3 — cùng phiên bản thư viện trong app Android.

Chạy trong môi trường Python 3.9 riêng (opencv-python 4.5.3 chỉ có bản cho Python ≤ 3.9):
    pip install opencv-python-headless==4.5.3.56 "numpy<2"
    python src/verify_onnx.py --model bubble_classifier_v1.onnx --vectors testvectors.npz

Kiểm tra 2 việc:
  [1] Model chạy trên OpenCV 4.5.3 cho kết quả giống PyTorch.
  [2] Tiền xử lý kiểu Kotlin (resize INTER_AREA + blobFromImage) trên OpenCV 4.5.3 khớp với tiền xử lý lúc train.
File này cố ý KHÔNG import torch hay code khác trong repo, để chạy độc lập.
"""
import argparse
import sys

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    print("OpenCV", cv2.__version__)
    if not cv2.__version__.startswith("4.5.3"):
        print("CẢNH BÁO: đây không phải OpenCV 4.5.3 như trong app Android — kết quả chỉ mang tính tham khảo.")

    net = cv2.dnn.readNetFromONNX(args.model)
    data = np.load(args.vectors)
    inputs, logits = data["inputs"], data["logits"]

    # [1] Model
    max_diff, wrong_class = 0.0, 0
    for i in range(len(inputs)):
        net.setInput(inputs[i:i + 1])
        out = net.forward()
        max_diff = max(max_diff, float(np.abs(out - logits[i:i + 1]).max()))
        wrong_class += int(out.argmax() != logits[i].argmax())
    ok_model = max_diff < args.atol and wrong_class == 0
    print("[1] Model: %d mẫu, sai khác lớn nhất = %.2e, khác kết quả phân loại = %d  →  %s"
          % (len(inputs), max_diff, wrong_class, "ĐẠT" if ok_model else "KHÔNG ĐẠT"))

    # [2] Tiền xử lý giống code Kotlin: Imgproc.resize(INTER_AREA) rồi Dnn.blobFromImage(scale 1/255)
    raw, sizes = data["raw"], data["raw_sizes"]
    max_pre = 0.0
    for i in range(len(raw)):
        h, w = int(sizes[i][0]), int(sizes[i][1])
        resized = cv2.resize(np.ascontiguousarray(raw[i, :h, :w]), (32, 32), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0 / 255.0)  # (1, 1, 32, 32)
        max_pre = max(max_pre, float(np.abs(blob - inputs[i:i + 1]).max()))
    # Khác nhau tối đa 1 mức xám (1/255) giữa các phiên bản OpenCV là chấp nhận được
    ok_pre = max_pre <= 1.5 / 255
    print("[2] Tiền xử lý: sai khác lớn nhất = %.4f (≈ %.1f mức xám)  →  %s"
          % (max_pre, max_pre * 255, "ĐẠT" if ok_pre else "KHÔNG ĐẠT"))

    if ok_model and ok_pre:
        print("OK — model dùng được trong app Android (OpenCV 4.5.3).")
        return 0
    print("LỖI — không đưa model này vào app. Gửi toàn bộ output này cho Claude để xử lý.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
