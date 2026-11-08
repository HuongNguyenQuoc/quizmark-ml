"""Export BubbleNet → ONNX opset 11 cho OpenCV 4.5.3 DNN (app Android), kèm test vectors và model card.

Hai cách export:
  - "torch": exporter TorchScript (dynamo=False) của PyTorch. Vẫn chạy trên torch 2.15 nhưng đã bị đánh dấu sẽ bỏ.
  - "manual": tự dựng đồ thị ONNX từ trọng số (gộp BatchNorm vào Conv). Không phụ thuộc exporter của PyTorch.
"auto" thử "torch" trước, lỗi (hoặc sinh phép toán lạ) thì dùng "manual". Cả hai đều được verify_onnx.py kiểm tra.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

from .config import CLASSES, INPUT_SIZE, ONNX_OPSET
from .data import Sample, preprocess, read_gray
from .model import BubbleNet

# Các phép toán OpenCV 4.5.3 DNN đọc chắc chắn được
SAFE_OPS = {"Conv", "Relu", "MaxPool", "Flatten", "Gemm", "BatchNormalization"}


def _export_with_torch(model: nn.Module, path: Path) -> None:
    dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=ONNX_OPSET, do_constant_folding=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # cảnh báo "legacy exporter" là bình thường
        try:
            torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
        except TypeError:
            # PyTorch cũ chưa có tham số dynamo (mặc định đã là exporter cũ)
            torch.onnx.export(model, dummy, str(path), **kwargs)


def _build_manually(model: BubbleNet) -> onnx.ModelProto:
    """Dựng đồ thị ONNX opset 11: (Conv+BN gộp → Relu → MaxPool) x3 → Flatten → Gemm."""
    nodes, initializers = [], []

    def add_init(array: np.ndarray, name: str) -> str:
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), name))
        return name

    current = "input"
    for i, block in enumerate(model.features):
        conv, bn, relu, pool = block
        assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
        assert isinstance(relu, nn.ReLU) and isinstance(pool, nn.MaxPool2d), "Kiến trúc model đã đổi — sửa _build_manually"
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = (bn.weight / std).detach().cpu().numpy()
        weight = conv.weight.detach().cpu().numpy() * scale[:, None, None, None]
        bias = (bn.bias - bn.running_mean * bn.weight / std).detach().cpu().numpy()
        if conv.bias is not None:
            bias = bias + conv.bias.detach().cpu().numpy() * scale
        w_name, b_name = add_init(weight, f"conv{i}.weight"), add_init(bias, f"conv{i}.bias")
        k, p = conv.kernel_size, conv.padding
        nodes.append(helper.make_node("Conv", [current, w_name, b_name], [f"conv{i}"],
                                      kernel_shape=list(k), pads=[p[0], p[1], p[0], p[1]], strides=list(conv.stride)))
        nodes.append(helper.make_node("Relu", [f"conv{i}"], [f"relu{i}"]))
        ks = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
        nodes.append(helper.make_node("MaxPool", [f"relu{i}"], [f"pool{i}"], kernel_shape=[ks, ks], strides=[ks, ks]))
        current = f"pool{i}"

    flatten, _dropout, linear = model.classifier
    assert isinstance(flatten, nn.Flatten) and isinstance(linear, nn.Linear), "Kiến trúc model đã đổi — sửa _build_manually"
    nodes.append(helper.make_node("Flatten", [current], ["flat"], axis=1))
    nodes.append(helper.make_node("Gemm", ["flat",
                                           add_init(linear.weight.detach().cpu().numpy(), "fc.weight"),
                                           add_init(linear.bias.detach().cpu().numpy(), "fc.bias")],
                                  ["logits"], alpha=1.0, beta=1.0, transB=1))

    graph = helper.make_graph(
        nodes, "BubbleNet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, INPUT_SIZE, INPUT_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, len(CLASSES)])],
        initializer=initializers)
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)], producer_name="quizmark-ml")
    proto.ir_version = 6  # IR của thời opset 11, để bộ đọc ONNX cũ trong OpenCV 4.5.3 chắc chắn hiểu
    return proto


def export_onnx(model: nn.Module, out_path, method: str = "auto") -> dict:
    """Export và kiểm tra. Trả về {"method": ..., "ops": [...], "size_kb": ...}."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    used = None

    if method in ("auto", "torch"):
        try:
            _export_with_torch(model, out_path)
            ops = {n.op_type for n in onnx.load(str(out_path)).graph.node}
            if ops <= SAFE_OPS:
                used = "torch"
            elif method == "torch":
                raise RuntimeError(f"Exporter của PyTorch sinh phép toán OpenCV 4.5.3 có thể không đọc được: {ops - SAFE_OPS}")
            else:
                print(f"Exporter của PyTorch sinh phép toán lạ {ops - SAFE_OPS} → chuyển sang tự dựng ONNX")
        except Exception as e:
            if method == "torch":
                raise
            print(f"Exporter của PyTorch không dùng được ({type(e).__name__}: {e}) → chuyển sang tự dựng ONNX")

    if used is None:
        onnx.save(_build_manually(model), str(out_path))
        used = "manual"

    proto = onnx.load(str(out_path))
    onnx.checker.check_model(proto)
    ops = sorted({n.op_type for n in proto.graph.node})
    return {"method": used, "ops": ops, "opset": proto.opset_import[0].version,
            "size_kb": round(out_path.stat().st_size / 1024, 1)}


def save_test_vectors(model: nn.Module, samples: list[Sample], out_path, n: int = 64, seed: int = 0) -> int:
    """Lưu input đã tiền xử lý + logits của PyTorch + ảnh ô gốc, để verify_onnx.py so khớp trên OpenCV 4.5.3.
    Chỉ dùng mảng số thuần (không pickle) để đọc được bằng numpy 1.x trong môi trường Python 3.9."""
    model = model.eval().cpu()
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    raws = [read_gray(samples[i].path) for i in picks]
    inputs = np.stack([preprocess(g) for g in raws])[:, None].astype(np.float32)  # (N, 1, 32, 32)
    with torch.no_grad():
        logits = model(torch.from_numpy(inputs)).numpy().astype(np.float32)
    h_max, w_max = max(g.shape[0] for g in raws), max(g.shape[1] for g in raws)
    raw = np.zeros((len(raws), h_max, w_max), dtype=np.uint8)
    for i, g in enumerate(raws):
        raw[i, :g.shape[0], :g.shape[1]] = g
    sizes = np.array([g.shape for g in raws], dtype=np.int32)
    np.savez(str(out_path), inputs=inputs, logits=logits, raw=raw, raw_sizes=sizes)
    return len(raws)


def quick_check_with_current_opencv(onnx_path, vectors_path) -> float:
    """Kiểm tra nhanh bằng OpenCV đang cài (Colab/máy tính). Bước kiểm tra chuẩn vẫn là verify_onnx.py với 4.5.3."""
    import cv2
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    data = np.load(str(vectors_path))
    max_diff = 0.0
    for i in range(len(data["inputs"])):
        net.setInput(data["inputs"][i:i + 1])
        max_diff = max(max_diff, float(np.abs(net.forward() - data["logits"][i:i + 1]).max()))
    print(f"OpenCV {cv2.__version__}: sai khác lớn nhất so với PyTorch = {max_diff:.2e}")
    return max_diff


def write_model_card(path, version: int, test_report: dict, counts: dict,
                     export_info: dict, repo_dir: Optional[str] = None) -> dict:
    commit = None
    try:
        commit = subprocess.run(["git", "-C", repo_dir or ".", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    card = {
        "name": f"bubble_classifier_v{version}",
        "version": version,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "classes": list(CLASSES),
        "input": {"name": "input", "shape": [1, 1, INPUT_SIZE, INPUT_SIZE], "dtype": "float32", "layout": "NCHW",
                  "preprocessing": "ô cắt từ grayMat → resize 32x32 INTER_AREA → chia 255"},
        "output": {"name": "logits", "shape": [1, len(CLASSES)],
                   "note": "softmax ra xác suất; 'erased' = không tô"},
        "export": export_info,
        "torch_version": torch.__version__,
        "git_commit": commit,
        "data_counts": counts,
        "test_metrics": test_report,
    }
    Path(path).write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card
