"""Train BubbleNet và đánh giá — kèm so sánh trực tiếp với ngưỡng 50% cũ của app."""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import CLASSES, LABEL_TO_INDEX
from .data import Sample, augment, preprocess, read_gray
from .model import BubbleNet


class BubbleDataset(Dataset):
    """Nạp toàn bộ ảnh vào RAM (vài nghìn ảnh nhỏ, không đáng kể)."""

    def __init__(self, samples: list[Sample], train: bool):
        self.images = [read_gray(s.path) for s in samples]
        self.labels = np.array([LABEL_TO_INDEX[s.label] for s in samples], dtype=np.int64)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train:
            img = augment(img, np.random.default_rng())
        x = preprocess(img)[None, :, :]  # (1, 32, 32)
        return torch.from_numpy(x), int(self.labels[i])


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    """Ô trống chiếm đa số → tăng trọng số cho lớp ít mẫu (1/sqrt(số mẫu)). Lớp chưa có mẫu: trọng số 0."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    weights = np.zeros(len(CLASSES))
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def _predict_loader(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(x.to(device)), dim=1).cpu().numpy() for x, _ in loader]
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), dtype=np.float32)


def predict(model: nn.Module, samples: list[Sample], device: Optional[str] = None) -> np.ndarray:
    """Xác suất (N, 3) theo thứ tự CLASSES."""
    device = device or next(model.parameters()).device.type
    loader = DataLoader(BubbleDataset(samples, train=False), batch_size=512, shuffle=False)
    return _predict_loader(model, loader, device)


def train(
    train_samples: list[Sample],
    val_samples: list[Sample],
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> BubbleNet:
    """Train và trả về model có macro-F1 trên tập val tốt nhất."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or pick_device()
    log(f"Thiết bị: {device} | train={len(train_samples)} ô, val={len(val_samples)} ô")

    train_ds = BubbleDataset(train_samples, train=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          persistent_workers=num_workers > 0)
    val_dl = DataLoader(BubbleDataset(val_samples, train=False), batch_size=512, shuffle=False)

    model = BubbleNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_ds.labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(train_dl)))

    best_state, best_score = None, -math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        started, total_loss, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
            seen += len(y)

        report = evaluate(val_samples, _predict_loader(model, val_dl, device))
        score = report["macro_f1"]
        log(f"epoch {epoch:2d}/{epochs}  loss={total_loss / max(seen, 1):.4f}  "
            f"val_acc={report['accuracy']:.4f}  val_macroF1={score:.4f}  ({time.time() - started:.1f}s)")
        if not math.isnan(score) and score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"Giữ model tốt nhất: val macro-F1 = {best_score:.4f}")
    return model.eval()


def evaluate(samples: list[Sample], probs: np.ndarray) -> dict:
    """Độ chính xác 3 lớp + điều app thực sự cần: ô có được tính là 'tô' hay không (erased = không tô)."""
    n_classes = len(CLASSES)
    y_true = np.array([LABEL_TO_INDEX[s.label] for s in samples], dtype=np.int64)
    y_pred = probs.argmax(axis=1) if len(samples) else np.zeros(0, dtype=np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class, f1_scores = {}, []
    for i, name in enumerate(CLASSES):
        tp, support, predicted = confusion[i, i], confusion[i].sum(), confusion[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else float("nan")
        if support:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1_scores.append(f1)
        else:
            f1 = float("nan")
        per_class[name] = dict(precision=float(precision), recall=float(recall), f1=float(f1), support=int(support))

    filled = LABEL_TO_INDEX["filled"]
    true_filled = y_true == filled
    model_filled = y_pred == filled
    threshold_filled = np.array([s.auto_label == "filled" for s in samples], dtype=bool)
    n = len(samples)
    return dict(
        n=n,
        accuracy=float((y_pred == y_true).mean()) if n else float("nan"),
        macro_f1=float(np.mean(f1_scores)) if f1_scores else float("nan"),
        per_class=per_class,
        confusion=confusion.tolist(),
        filled_errors_model=int((model_filled != true_filled).sum()),
        filled_errors_threshold=int((threshold_filled != true_filled).sum()),
    )


def format_report(report: dict) -> str:
    n = report["n"]
    lines = [f"Số ô đánh giá: {n}",
             f"Độ chính xác 3 lớp: {report['accuracy']:.4f}   macro-F1: {report['macro_f1']:.4f}",
             "",
             "Ma trận nhầm lẫn (hàng = nhãn đúng, cột = model đoán):",
             " " * 10 + "".join(f"{c:>9}" for c in CLASSES)]
    for name, row in zip(CLASSES, report["confusion"]):
        lines.append(f"{name:>10}" + "".join(f"{v:>9}" for v in row))
    lines.append("")
    for name, m in report["per_class"].items():
        lines.append(f"{name:>8}: precision={m['precision']:.3f}  recall={m['recall']:.3f}  (số mẫu {m['support']})")
    em, et = report["filled_errors_model"], report["filled_errors_threshold"]
    lines += ["",
              "Quyết định 'ô có được tô không' (điều app dùng để chấm điểm):",
              f"  Model mới   : sai {em}/{n} ô",
              f"  Ngưỡng 50% cũ: sai {et}/{n} ô",
              "  (Chỉ chính xác khi nhãn đã được kiểm tra; ngưỡng cũ sai đúng bằng số ô bạn đã sửa nhãn tô↔trống/tẩy.)"]
    return "\n".join(lines)


def save_checkpoint(model: nn.Module, path) -> None:
    torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES)}, str(path))


def load_checkpoint(path, device: Optional[str] = None) -> BubbleNet:
    device = device or pick_device()
    ckpt = torch.load(str(path), map_location=device)
    model = BubbleNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
