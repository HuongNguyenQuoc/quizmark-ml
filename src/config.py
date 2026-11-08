"""Quy ước chung giữa repo này và app QuizMark.

ĐỔI BẤT KỲ GIÁ TRỊ NÀO Ở ĐÂY → phải sửa cả code Kotlin trong app và tăng MODEL_VERSION trong notebook.
"""

# Thứ tự 3 output (logit) của model. "erased" (đã tẩy / gạch) được app tính là KHÔNG tô.
CLASSES = ("empty", "filled", "erased")
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

# Nhãn sửa tay đặc biệt: loại mẫu khỏi dữ liệu (ví dụ lần quét bị nhận sai góc phiếu).
IGNORE = "ignore"

# Input: ô cắt từ ảnh xám (grayMat) → resize INPUT_SIZE x INPUT_SIZE bằng INTER_AREA → float32 / 255.
# Shape [1, 1, 32, 32] (NCHW), kích thước cố định.
INPUT_SIZE = 32

# App dùng OpenCV 4.5.3 (com.quickbirdstudios:opencv:4.5.3.0); bộ đọc ONNX của bản này hỗ trợ tốt opset 11.
ONNX_OPSET = 11

TEMPLATE_NAMES = {1: "THPT", 2: "20 câu", 3: "40 câu", 4: "50 câu", 5: "120 câu"}
