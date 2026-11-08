"""Xem và sửa nhãn ngay trong Colab: lưới ảnh có đánh số, sửa bằng một dòng lệnh.

Ví dụ:
    rv = Reviewer(all_samples, overrides)
    rv.show("filled")                      # lưới 100 ô đang mang nhãn 'filled', có số thứ tự
    rv.fix("filled", [3, 17], to="erased") # số thứ tự lấy từ lần show("filled") gần nhất
    rv.ignore_sheet("261108-142233")       # bỏ cả một lần quét lỗi
Mọi bản sửa được ghi ngay vào labels_override.csv (trên Drive).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import CLASSES, IGNORE, TEMPLATE_NAMES
from .data import LabelOverrides, Sample, read_gray

SHORT = {"empty": "trống", "filled": "tô", "erased": "tẩy", IGNORE: "bỏ"}


class Reviewer:
    def __init__(self, samples: list[Sample], overrides: LabelOverrides):
        """samples nên được nạp với include_ignored=True để có thể xem lại cả các ô đã bỏ."""
        self.samples = samples
        self.overrides = overrides
        self._views: dict[str, list[Sample]] = {}
        self._probs: dict[str, np.ndarray] = {}
        self._suspicious: list[tuple[float, Sample, str]] = []

    def label_of(self, s: Sample) -> str:
        return self.overrides.labels.get(s.name, s.auto_label)

    # ---------- Hiển thị ----------

    @staticmethod
    def _grid(items: Sequence[Sample], captions: Sequence[str], title: str, cols: int = 10) -> None:
        if not items:
            print("(không có ô nào)")
            return
        rows = math.ceil(len(items) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.3))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, s, cap in zip(axes, items, captions):
            ax.imshow(read_gray(s.path), cmap="gray", vmin=0, vmax=255)
            ax.set_title(cap, fontsize=7)
        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.show()

    def show(self, label: str, page: int = 0, per_page: int = 100,
             template: Optional[int] = None, region: Optional[str] = None) -> None:
        """Lưới các ô đang mang nhãn `label`. Lọc thêm theo mẫu phiếu (template) hoặc vùng (region) nếu muốn."""
        pool = sorted((s for s in self.samples
                       if self.label_of(s) == label
                       and (template is None or s.template == template)
                       and (region is None or s.region == region)),
                      key=lambda s: s.name)
        self._views[label] = pool
        pages = max(1, math.ceil(len(pool) / per_page))
        start = page * per_page
        items = pool[start:start + per_page]
        self._grid(items, [str(start + i) for i in range(len(items))],
                   f"Nhãn '{label}': {len(pool)} ô — trang {page + 1}/{pages}")

    def list_sheets(self) -> None:
        """Bảng các lần quét. Tỉ lệ ô 'tô' bất thường (quá cao/quá thấp) thường là lần quét bị nhận sai góc phiếu."""
        groups: dict[str, list[Sample]] = defaultdict(list)
        for s in self.samples:
            groups[s.sheet_id].append(s)
        print(f"{'sheetId':<22}{'mẫu phiếu':<11}{'số ô':>6}{'tô':>6}{'tẩy':>6}{'bỏ':>6}{'tỉ lệ tô':>10}")
        for sheet_id in sorted(groups):
            items = groups[sheet_id]
            labels = Counter(self.label_of(s) for s in items)
            ratio = labels["filled"] / max(1, len(items))
            print(f"{sheet_id:<22}{TEMPLATE_NAMES.get(items[0].template, items[0].template):<11}"
                  f"{len(items):>6}{labels['filled']:>6}{labels['erased']:>6}{labels[IGNORE]:>6}{ratio:>10.1%}")

    def show_sheet(self, sheet_prefix: str, label: str = "filled", max_items: int = 100) -> None:
        """Xem nhanh các ô của MỘT lần quét (mặc định các ô 'tô') để kiểm tra lần quét đó có bị lỗi không."""
        items = sorted((s for s in self.samples
                        if s.sheet_id.startswith(sheet_prefix) and self.label_of(s) == label),
                       key=lambda s: s.name)[:max_items]
        self._grid(items, [f"{s.region} {s.position} {s.option}" for s in items],
                   f"Lần quét {sheet_prefix}: các ô '{label}'")

    # ---------- Sửa nhãn ----------

    def fix(self, label: str, indices: Sequence[int], to: str) -> None:
        """Đổi nhãn các ô theo số thứ tự trong lần show(label) gần nhất."""
        view = self._views.get(label)
        if view is None:
            raise RuntimeError(f"Hãy chạy rv.show('{label}') trước rồi mới fix.")
        bad = [i for i in indices if not 0 <= i < len(view)]
        if bad:
            raise IndexError(f"Số thứ tự không hợp lệ: {bad} (lần show gần nhất có {len(view)} ô)")
        n = self.overrides.set_many((view[i].name for i in indices), to)
        print(f"Đã đổi {n} ô '{label}' → '{to}'. Chạy lại rv.show(...) để xem danh sách mới.")

    def ignore_sheet(self, sheet_prefix: str) -> None:
        """Bỏ toàn bộ các ô của những lần quét có sheetId bắt đầu bằng `sheet_prefix`."""
        names = [s.name for s in self.samples if s.sheet_id.startswith(sheet_prefix)]
        if not names:
            print(f"Không có lần quét nào bắt đầu bằng '{sheet_prefix}'")
            return
        sheets = {s.sheet_id for s in self.samples if s.sheet_id.startswith(sheet_prefix)}
        self.overrides.set_many(names, IGNORE)
        print(f"Đã bỏ {len(names)} ô của {len(sheets)} lần quét: {', '.join(sorted(sheets))}")

    # ---------- Mẫu đáng ngờ (sau khi train) ----------

    def set_predictions(self, samples: Sequence[Sample], probs: np.ndarray) -> None:
        self._probs = {s.name: p for s, p in zip(samples, probs)}

    def show_suspicious(self, page: int = 0, per_page: int = 100) -> None:
        """Các ô mà model đoán KHÁC nhãn hiện tại, model càng chắc chắn thì xếp càng trước.
        Chú thích: 'số: nhãn hiện tại→model đoán độ tin cậy'."""
        items = []
        for s in self.samples:
            label = self.label_of(s)
            probs = self._probs.get(s.name)
            if label == IGNORE or probs is None:
                continue
            predicted = CLASSES[int(probs.argmax())]
            if predicted != label:
                items.append((float(probs.max()), s, predicted))
        items.sort(key=lambda t: -t[0])
        self._suspicious = items
        pages = max(1, math.ceil(len(items) / per_page))
        start = page * per_page
        chunk = items[start:start + per_page]
        self._grid([s for _, s, _ in chunk],
                   [f"{start + i}: {SHORT[self.label_of(s)]}→{SHORT[p]} {conf:.2f}"
                    for i, (conf, s, p) in enumerate(chunk)],
                   f"Mẫu đáng ngờ: {len(items)} ô — trang {page + 1}/{pages}")

    def accept_model(self, indices: Sequence[int]) -> None:
        """Đồng ý với model cho các ô (số thứ tự trong show_suspicious gần nhất)."""
        by_label: dict[str, list[str]] = defaultdict(list)
        for i in indices:
            _, s, predicted = self._suspicious[i]
            by_label[predicted].append(s.name)
        for label, names in by_label.items():
            self.overrides.set_many(names, label)
        print(f"Đã chấp nhận nhãn của model cho {sum(map(len, by_label.values()))} ô.")

    def fix_suspicious(self, indices: Sequence[int], to: str) -> None:
        """Tự chọn nhãn đúng cho các ô (số thứ tự trong show_suspicious gần nhất)."""
        n = self.overrides.set_many((self._suspicious[i][1].name for i in indices), to)
        print(f"Đã đổi {n} ô → '{to}'.")
