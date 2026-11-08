"""Sinh dữ liệu GIẢ đúng định dạng app xuất ra.

Dùng để chạy thử notebook trước khi có dữ liệu thật và để kiểm thử code.
KHÔNG đưa model train từ dữ liệu giả vào app.
"""
from __future__ import annotations

import csv
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .data import LabelOverrides


def _line(img, p1, p2, tone, thickness=2):
    cv2.line(img, p1, p2, int(tone), thickness, cv2.LINE_AA)


def make_cell(kind: str, rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Một ô giả: vòng tròn in sẵn (+ chữ cái mờ), rồi tô / tẩy / gạch tuỳ `kind`."""
    bg = int(rng.integers(175, 245))
    img = np.full((h, w), bg, np.uint8)
    cx, cy = int(round(w / 2 + rng.uniform(-1.5, 1.5))), int(round(h / 2 + rng.uniform(-1.5, 1.5)))
    r = int(min(w, h) * 0.38)
    ring = int(max(0, bg - rng.uniform(70, 120)))
    cv2.circle(img, (cx, cy), r, ring, 1, lineType=cv2.LINE_AA)
    if rng.random() < 0.7:  # chữ A/B/C/D in mờ trong ô
        cv2.putText(img, "ABCD"[int(rng.integers(4))], (cx - r // 2, cy + r // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, r / 22, ring, 1, cv2.LINE_AA)

    if kind == "filled":
        tone = int(rng.uniform(15, 120))  # đậm → nhạt
        cv2.circle(img, (cx, cy), int(r * rng.uniform(0.7, 1.0)), tone, -1, lineType=cv2.LINE_AA)
    elif kind == "erased":
        if rng.random() < 0.5:  # tẩy: vết mờ loang + vệt cục tẩy
            cv2.circle(img, (cx, cy), int(r * 0.9), int(bg - rng.uniform(25, 70)), -1, lineType=cv2.LINE_AA)
            for _ in range(int(rng.integers(2, 5))):
                p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                _line(img, p1, p2, bg - rng.uniform(0, 15))
        else:  # gạch chéo
            d, tone = int(r * 0.9), rng.uniform(20, 90)
            _line(img, (cx - d, cy - d), (cx + d, cy + d), tone)
            _line(img, (cx - d, cy + d), (cx + d, cy - d), tone)

    out = img.astype(np.float32)
    out += np.linspace(-1, 1, w)[None, :] * rng.uniform(-20, 20)  # ánh sáng không đều
    out += rng.normal(0, rng.uniform(1, 6), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _simulated_threshold_label(kind: str, rng: np.random.Generator) -> str:
    """Giả lập lỗi của ngưỡng 50% cũ: bỏ sót ô tô nhạt, nhận nhầm ô tẩy/gạch là tô."""
    if kind == "filled":
        return "empty" if rng.random() < 0.08 else "filled"
    if kind == "erased":
        return "filled" if rng.random() < 0.35 else "empty"
    return "filled" if rng.random() < 0.003 else "empty"


def make_synthetic_zip(out_dir, n_sheets: int = 16, seed: int = 0) -> dict[str, str]:
    """Tạo out_dir/bubble_dataset_synthetic.zip (đúng định dạng app) + synthetic_truth.csv.
    Trả về {tên file: nhãn đúng}."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bubble_dataset"
        for label in ("empty", "filled"):
            (root / label).mkdir(parents=True)

        def save(sheet_id, template, region, position, option, kind, w, h):
            name = f"{sheet_id}_t{template}_{region}_{position}_{option}.png"
            auto = _simulated_threshold_label(kind, rng)
            cv2.imwrite(str(root / auto / name), make_cell(kind, rng, w, h))
            truth[name] = kind

        for s in range(n_sheets):
            sheet_id = f"991231-{s:06d}-syn{seed}"
            template = int(rng.integers(1, 6))
            w, h = int(rng.integers(28, 42)), int(rng.integers(28, 42))
            for col in range(6):  # SBD: 6 cột x 10 hàng, mỗi cột tô 1 ô
                digit = int(rng.integers(10))
                for row in range(10):
                    save(sheet_id, template, "sbd", f"c{col}", str(row), "filled" if row == digit else "empty", w, h)
            for q in range(1, int(rng.integers(20, 41)) + 1):
                chosen = int(rng.integers(4)) if rng.random() < 0.9 else -1  # 10% bỏ trống
                changed = int(rng.integers(4)) if rng.random() < 0.15 else -1  # đổi ý: tẩy/gạch 1 ô khác
                for c, opt in enumerate("ABCD"):
                    kind = "filled" if c == chosen else ("erased" if c == changed else "empty")
                    save(sheet_id, template, "ans", f"q{q}", opt, kind, w, h)

        zip_path = out_dir / "bubble_dataset_synthetic.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(root.rglob("*.png")):
                zf.write(f, f.relative_to(Path(tmp)).as_posix())

    with (out_dir / "synthetic_truth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(sorted(truth.items()))
    return truth


def apply_truth(overrides: LabelOverrides, truth: dict[str, str]) -> int:
    """Giả lập việc 'đã kiểm tra xong nhãn': ghi nhãn đúng của dữ liệu giả vào overrides."""
    by_label: dict[str, list[str]] = {}
    for name, label in truth.items():
        by_label.setdefault(label, []).append(name)
    return sum(overrides.set_many(names, label) for label, names in by_label.items())
