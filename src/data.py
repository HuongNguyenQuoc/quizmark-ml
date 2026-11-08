"""Đọc dữ liệu ô tô do app QuizMark (bản debug) xuất ra, áp nhãn sửa tay, chia tập theo lần quét.

Định dạng file zip app xuất (Tài khoản → "Xuất dữ liệu (zip) lên Drive"):
    bubble_dataset/<empty|filled>/<sheetId>_t<template>_<vùng>_<vị trí>_<lựa chọn>.png
Thư mục <empty|filled> là NHÃN TẠM theo ngưỡng 50% cũ của app; nhãn đúng được sửa trong labels_override.csv.
"""
from __future__ import annotations

import csv
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .config import CLASSES, IGNORE, INPUT_SIZE, TEMPLATE_NAMES

AUTO_LABELS = ("empty", "filled")


@dataclass(frozen=True)
class Sample:
    path: Path
    name: str          # tên file — dùng làm khoá trong labels_override.csv
    sheet_id: str      # mỗi lần quét một mã, bắt đầu bằng thời gian quét yyMMdd-HHmmss
    template: int      # 1=THPT, 2=20 câu, 3=40 câu, 4=50 câu, 5=120 câu
    region: str        # sbd, code, ans, p1, p2, p3
    position: str      # q12 (câu 12), c3 (cột 3), q73c0 ...
    option: str        # A-D, chữ số, T/F, neg/dot
    auto_label: str    # nhãn tạm theo ngưỡng 50% (empty/filled)
    label: str         # nhãn cuối cùng sau khi áp nhãn sửa tay


def parse_name(name: str) -> Optional[dict]:
    """<sheetId>_t<template>_<vùng>_<vị trí>_<lựa chọn>.png → dict, hoặc None nếu sai định dạng."""
    if not name.lower().endswith(".png"):
        return None
    parts = name[:-4].split("_")
    if len(parts) != 5 or not parts[1].startswith("t"):
        return None
    try:
        template = int(parts[1][1:])
    except ValueError:
        return None
    return dict(sheet_id=parts[0], template=template, region=parts[2], position=parts[3], option=parts[4])


def extract_zips(zip_dir, out_dir) -> int:
    """Giải nén mọi bubble_dataset_*.zip trong zip_dir vào out_dir. File trùng tên được ghi đè (không nhân đôi)."""
    zip_dir, out_dir = Path(zip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(zip_dir.glob("bubble_dataset_*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
    return len(zips)


class LabelOverrides:
    """Nhãn sửa tay, lưu trong CSV (filename,label). Nên đặt file trên Google Drive để không mất khi Colab ngắt."""

    def __init__(self, csv_path):
        self.path = Path(csv_path)
        self.labels: dict[str, str] = {}
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.labels[row["filename"]] = row["label"]

    def set_many(self, names: Iterable[str], label: str) -> int:
        if label not in CLASSES and label != IGNORE:
            raise ValueError(f"Nhãn '{label}' không hợp lệ. Dùng một trong: {', '.join(CLASSES)}, {IGNORE}")
        names = list(names)
        for name in names:
            self.labels[name] = label
        self.save()
        return len(names)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            for name in sorted(self.labels):
                writer.writerow([name, self.labels[name]])
        tmp.replace(self.path)


def load_samples(data_dir, overrides: Optional[LabelOverrides] = None, include_ignored: bool = False) -> list[Sample]:
    """Đọc danh sách mẫu (chưa đọc ảnh). data_dir là thư mục đã giải nén (chứa bubble_dataset/)."""
    root = Path(data_dir)
    if (root / "bubble_dataset").is_dir():
        root = root / "bubble_dataset"
    samples = []
    for auto_label in AUTO_LABELS:
        for path in sorted((root / auto_label).glob("*.png")):
            info = parse_name(path.name)
            if info is None:
                continue
            label = overrides.labels.get(path.name, auto_label) if overrides else auto_label
            if label == IGNORE and not include_ignored:
                continue
            samples.append(Sample(path=path, name=path.name, auto_label=auto_label, label=label, **info))
    return samples


def summarize(samples: list[Sample]) -> str:
    lines = [f"Tổng: {len(samples)} ô từ {len({s.sheet_id for s in samples})} lần quét"]
    by_label = Counter(s.label for s in samples)
    lines.append("Theo nhãn: " + ", ".join(f"{c}={by_label.get(c, 0)}" for c in CLASSES))
    by_template = Counter(s.template for s in samples)
    lines.append("Theo mẫu phiếu: " + ", ".join(
        f"{TEMPLATE_NAMES.get(t, t)}={n}" for t, n in sorted(by_template.items())))
    changed = sum(1 for s in samples if s.label != s.auto_label)
    lines.append(f"Đã sửa nhãn: {changed} ô")
    if by_label.get("erased", 0) == 0:
        lines.append("Lưu ý: chưa có ô 'erased' nào — hãy đánh dấu các ô đã tẩy/gạch ở bước kiểm tra nhãn.")
    return "\n".join(lines)


def split_by_sheet(samples: list[Sample], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Chia train/val/test THEO LẦN QUÉT (không chia theo từng ô) để độ chính xác đo được sát thực tế."""
    sheets = sorted({s.sheet_id for s in samples})
    if len(sheets) < 3:
        raise ValueError(f"Cần ít nhất 3 lần quét để chia train/val/test (hiện có {len(sheets)}).")
    random.Random(seed).shuffle(sheets)
    n_test = max(1, round(len(sheets) * test_frac))
    n_val = max(1, round(len(sheets) * val_frac))
    test_ids = set(sheets[:n_test])
    val_ids = set(sheets[n_test:n_test + n_val])
    train = [s for s in samples if s.sheet_id not in test_ids and s.sheet_id not in val_ids]
    val = [s for s in samples if s.sheet_id in val_ids]
    test = [s for s in samples if s.sheet_id in test_ids]
    return train, val, test


# ---------------- Ảnh ----------------

def read_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Không đọc được ảnh: {path}")
    return img


def preprocess(gray_u8: np.ndarray) -> np.ndarray:
    """ĐÚNG NHƯ APP: resize 32x32 bằng INTER_AREA → float32 → chia 255. Trả về mảng (32, 32)."""
    resized = cv2.resize(gray_u8, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def augment(gray_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Biến đổi ngẫu nhiên khi train, mô phỏng khác biệt thực tế giữa các lần quét."""
    img = gray_u8.astype(np.float32)
    h, w = img.shape
    # Lưới cắt ô cố định nên ô hay lệch vài pixel và to/nhỏ khác nhau một chút
    scale = rng.uniform(0.92, 1.08)
    tx, ty = rng.uniform(-2, 2), rng.uniform(-2, 2)
    matrix = np.array([[scale, 0, (1 - scale) * w / 2 + tx],
                       [0, scale, (1 - scale) * h / 2 + ty]], dtype=np.float32)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Ánh sáng: độ tương phản + độ sáng
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.75, 1.25) + mean + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.5:
        img = img + rng.normal(0, rng.uniform(1, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
