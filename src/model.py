"""BubbleNet: CNN rất nhỏ phân loại một ô tròn 32x32 thành empty / filled / erased."""
import torch.nn as nn

from .config import CLASSES


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BubbleNet(nn.Module):
    """Khoảng 27 nghìn tham số, file ONNX chỉ khoảng 100 KB.

    CHỈ dùng các phép toán mà OpenCV 4.5.3 DNN (trong app Android) đọc được:
    Conv, BatchNorm, ReLU, MaxPool, Flatten, Linear (Gemm). Không dùng shape động hay adaptive pooling.
    """

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64))  # 32 → 16 → 8 → 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
