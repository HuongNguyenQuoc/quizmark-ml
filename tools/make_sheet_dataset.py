"""Make a COMPUTER-MADE training dataset in exactly the format the QuizMark app exports.

Steps for every sheet:
  1. Render a template PDF from the app (QuizMark/app/src/main/res/raw/form_*.pdf).
  2. Draw marks exactly on the cells the app reads (ROI coordinates copied from the app's
     util/scanner/*Processor.kt, mapped through the same 4 corner markers ScannerCore uses).
     Mark styles: dark / light / partial / overflowing pencil, pen, erased, X, single slash.
  3. Add photo effects: desk border, perspective, uneven light, shadow, colour tint, blur,
     noise, lower resolution, JPEG compression.
  4. Run a Python copy of the app pipeline (corner detection → 800x1200 warp → adaptive
     threshold → 50% rule) and cut every cell from the warped gray image, like BubbleDatasetDumper.

Outputs (in --out):
  bubble_dataset_synth_<tag>.zip  → upload to  My Drive/quizmark-ml/data/
  labels_override.csv             → upload to  My Drive/quizmark-ml/   (TRUE labels incl. "erased")
  photos/                         → a few sample sheet photos to look at
  stats.txt                       → what the old 50% rule gets wrong on this data

This is practice data. A model trained only on it must not be shipped before it is trained on real sheets.

Usage (needs: numpy, opencv-python, pymupdf):
  python tools/make_sheet_dataset.py --out ../quizmark-synthetic --per-template 12 --seed 2026
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import fitz  # pymupdf
import numpy as np

DEFAULT_RAW = Path(__file__).resolve().parents[2] / "QuizMark" / "app" / "src" / "main" / "res" / "raw"
W, H = 800.0, 1200.0
FILLED_STYLES = ("dark", "light", "partial", "overflow", "pen")
ERASED_STYLES = ("erased", "x", "slash")


# ---------------- Layouts copied from the app's processors (800x1200 warped space) ----------------
def _basic(sbd, code, cols):
    specs = [dict(kind="grid", region="sbd", roi=sbd, rows=10, cols=6, r=0.38),
             dict(kind="grid", region="code", roi=code, rows=10, cols=3, r=0.38)]
    specs += [dict(kind="answers", region="ans", roi=(x, y, w, h), rows=e - s + 1, cols=4, start=s, r=0.36)
              for x, y, w, h, s, e in cols]
    return specs


LAYOUTS = {
    2: ("form_20_cau", _basic((186, 337, 201, 381), (417, 338, 104, 382),
                              [(221, 759, 131, 380, 1, 10), (418, 760, 131, 380, 11, 20)])),
    3: ("form_40_cau", _basic((90, 337, 198, 384), (320, 337, 100, 384),
                              [(520, 337, 130, 384, 1, 10), (123, 757, 130, 386, 11, 20),
                               (321, 757, 130, 386, 21, 30), (520, 757, 130, 386, 31, 40)])),
    4: ("form_50_cau", _basic((601, 80, 107, 254), (730, 80, 54, 254),
                              [(128, 445, 233, 717, 1, 25), (495, 445, 233, 717, 26, 50)])),
    5: ("form_120_cau", _basic((601, 78, 107, 254), (731, 78, 55, 254),
                               [(61, 389, 129, 779, 1, 30), (257, 389, 129, 779, 31, 60),
                                (448, 389, 129, 779, 61, 90), (642, 389, 129, 779, 91, 120)])),
    1: ("form_thpt_2025",
        [dict(kind="grid", region="sbd", roi=(549, 84, 138, 253), rows=10, cols=8, r=0.4),
         dict(kind="grid", region="code", roi=(713, 84, 70, 253), rows=10, cols=4, r=0.4)]
        + [dict(kind="answers", region="p1", roi=(x, 435, 147, 197), rows=10, cols=4, start=s, r=0.4)
           for x, s in ((53, 1), (241, 11), (428, 21), (615, 31))]
        + [dict(kind="tf", region="p2", roi=(x, 719, 68, 79), rows=4, cols=2, start=41 + 4 * i, r=0.36)
           for i, x in enumerate((55, 132, 244, 319, 430, 505, 618, 693))]
        + [dict(kind="number", region="p3", roi=(x, 894, 101, 244), rows=12, cols=4, start=73 + i, r=0.36)
           for i, x in enumerate((53, 173, 293, 412, 533, 653))]),
}


def p3_valid(r: int, c: int) -> bool:
    """Same rule as ExamThptProcessor.readPart3Box: row 0 only col 0 ('-'), row 1 cols 1-2 ('.')."""
    return c == 0 if r == 0 else (c in (1, 2) if r == 1 else True)


def cell_name_parts(spec, r, c):
    """(region, position, option) exactly as the app's BubbleDatasetDumper names files."""
    kind = spec["kind"]
    if kind == "grid":
        return spec["region"], f"c{c}", str(r)
    if kind == "answers":
        return spec["region"], f"q{spec['start'] + r}", "ABCD"[c]
    if kind == "tf":
        return "p2", f"q{spec['start'] + r}", "T" if c == 0 else "F"
    option = "neg" if r == 0 else ("dot" if r == 1 else str(r - 2))
    return "p3", f"q{spec['start']}c{c}", option


def cells_of(spec):
    for r in range(spec["rows"]):
        for c in range(spec["cols"]):
            if spec["kind"] != "number" or p3_valid(r, c):
                yield r, c


# ---------------- Same corner detection as ScannerCore.findExamCorners ----------------
def find_corners(gray):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt.astype(np.float32), 0.03 * cv2.arcLength(cnt, True), True)
        if 4 <= len(approx) <= 6:
            area = cv2.contourArea(approx)
            if 100 <= area <= 30000:
                x, y, w, h = cv2.boundingRect(cnt)
                if 0.5 <= w / h <= 2.0:
                    squares.append((approx, area))
    if len(squares) < 4:
        return None
    median = sorted(a for _, a in squares)[len(squares) // 2]
    centers = []
    for s, a in squares:
        if median * 0.1 <= a <= median * 15:
            x, y, w, h = cv2.boundingRect(s.astype(np.int32))
            centers.append((x + w / 2, y + h / 2))
    if len(centers) < 4:
        return None
    c = np.array(centers)
    tl, br = c[np.argmin(c.sum(1))], c[np.argmax(c.sum(1))]
    tr, bl = c[np.argmax(c[:, 0] - c[:, 1])], c[np.argmin(c[:, 0] - c[:, 1])]
    wt, hl = np.linalg.norm(tr - tl), np.linalg.norm(bl - tl)
    wb, hr = np.linalg.norm(br - bl), np.linalg.norm(br - tr)
    if wt < 80 or hl < 100 or hl < wt * 1.05 or abs(wt - wb) > wt * 0.25 or abs(hl - hr) > hl * 0.25:
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ---------------- Drawing marks ----------------
def _blob_mask(shape, center, radius, rng, roughness=0.12):
    """Irregular filled shape (hand-made fill), radius varies around the circle."""
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    radii = radius * (1 + rng.normal(0, roughness, angles.size))
    radii = cv2.GaussianBlur(radii.reshape(1, -1).astype(np.float32), (1, 5), 0).ravel()
    pts = np.stack([center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)], 1)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255, cv2.LINE_AA)
    return mask


def _paint(img, mask, tone, rng, grain):
    """Darken img towards `tone` where mask is set, with graphite grain texture."""
    m = mask.astype(np.float32) / 255.0
    texture = tone + rng.normal(0, grain, img.shape)
    texture = cv2.GaussianBlur(texture.astype(np.float32), (3, 3), 0)
    out = img.astype(np.float32) * (1 - m) + np.minimum(img.astype(np.float32), texture) * m
    img[:] = np.clip(out, 0, 255).astype(np.uint8)


def _strokes(img, center, radius, rng, tone, n, thickness):
    for _ in range(n):
        ang = rng.uniform(0, np.pi)
        off = rng.uniform(-0.6, 0.6) * radius
        dx, dy = np.cos(ang) * radius, np.sin(ang) * radius
        ox, oy = -np.sin(ang) * off, np.cos(ang) * off
        p1 = (int(center[0] - dx + ox), int(center[1] - dy + oy))
        p2 = (int(center[0] + dx + ox), int(center[1] + dy + oy))
        cv2.line(img, p1, p2, float(tone), thickness, cv2.LINE_AA)


def draw_mark(page, center, radius, style, rng):
    """Draws on a small patch around the mark only (fast), then writes the patch back."""
    half = int(radius * 1.8) + 4
    x0, y0 = max(0, int(center[0]) - half), max(0, int(center[1]) - half)
    x1, y1 = min(page.shape[1], int(center[0]) + half), min(page.shape[0], int(center[1]) + half)
    patch = page[y0:y1, x0:x1].copy()
    _draw_mark_local(patch, (center[0] - x0, center[1] - y0), radius, style, rng)
    page[y0:y1, x0:x1] = patch


def _draw_mark_local(img, center, radius, style, rng):
    cx, cy = center
    jitter = rng.normal(0, radius * 0.08, 2)
    c = (cx + jitter[0], cy + jitter[1])
    if style == "dark":
        _paint(img, _blob_mask(img.shape, c, radius * rng.uniform(0.82, 1.0), rng), rng.uniform(20, 75), rng, 18)
    elif style == "light":
        _paint(img, _blob_mask(img.shape, c, radius * rng.uniform(0.8, 1.0), rng), rng.uniform(105, 165), rng, 22)
    elif style == "partial":   # half-filled, crescent or small dot
        sub = rng.choice(["small", "crescent"])
        if sub == "small":
            mask = _blob_mask(img.shape, c, radius * rng.uniform(0.4, 0.65), rng, 0.15)
        else:
            mask = _blob_mask(img.shape, c, radius * 0.95, rng)
            off = rng.uniform(0.35, 0.7) * radius
            ang = rng.uniform(0, 2 * np.pi)
            cut = _blob_mask(img.shape, (c[0] + off * np.cos(ang), c[1] + off * np.sin(ang)), radius * 0.9, rng)
            mask = cv2.subtract(mask, cut)
        _paint(img, mask, rng.uniform(25, 90), rng, 18)
    elif style == "overflow":  # bigger than the bubble and off-centre
        c2 = (c[0] + rng.uniform(-0.25, 0.25) * radius, c[1] + rng.uniform(-0.25, 0.25) * radius)
        _paint(img, _blob_mask(img.shape, c2, radius * rng.uniform(1.05, 1.3), rng, 0.15), rng.uniform(25, 80), rng, 18)
    elif style == "pen":       # ballpoint: dark, smooth, scribble strokes
        _paint(img, _blob_mask(img.shape, c, radius * rng.uniform(0.8, 0.98), rng, 0.08), rng.uniform(10, 45), rng, 6)
        _strokes(img, c, radius * 0.9, rng, rng.uniform(5, 30), int(rng.integers(2, 5)), 1)
    elif style == "erased":    # graphite left after erasing + eraser streaks
        _paint(img, _blob_mask(img.shape, c, radius * rng.uniform(0.8, 1.05), rng, 0.2), rng.uniform(160, 232), rng, 15)
        _strokes(img, c, radius, rng, rng.uniform(140, 200), int(rng.integers(1, 4)), 1)
    elif style == "x":
        tone, d = rng.uniform(20, 90), radius * rng.uniform(0.7, 1.05)
        t = int(rng.integers(1, 3))
        cv2.line(img, (int(c[0] - d), int(c[1] - d)), (int(c[0] + d), int(c[1] + d)), float(tone), t, cv2.LINE_AA)
        cv2.line(img, (int(c[0] - d), int(c[1] + d)), (int(c[0] + d), int(c[1] - d)), float(tone), t, cv2.LINE_AA)
    elif style == "slash":     # crossed out with one line
        tone, d = rng.uniform(20, 90), radius * rng.uniform(0.8, 1.15)
        ang = rng.uniform(0.5, 1.1)
        cv2.line(img, (int(c[0] - d * np.cos(ang)), int(c[1] + d * np.sin(ang))),
                 (int(c[0] + d * np.cos(ang)), int(c[1] - d * np.sin(ang))), float(tone), int(rng.integers(1, 3)), cv2.LINE_AA)


def filled_style(rng):
    return rng.choice(FILLED_STYLES, p=[0.55, 0.13, 0.12, 0.08, 0.12])


def erased_style(rng):
    return rng.choice(ERASED_STYLES, p=[0.6, 0.25, 0.15])


def plan_marks(layout, template_id, rng):
    """{(spec_index, r, c): style}. Includes realistic mistakes: blanks, double fills, changed answers."""
    marks = {}
    for i, spec in enumerate(layout):
        kind, rows, cols = spec["kind"], spec["rows"], spec["cols"]
        if kind == "grid":  # one digit per column
            for c in range(cols):
                marks[(i, int(rng.integers(rows)), c)] = filled_style(rng) if rng.random() < 0.97 else erased_style(rng)
                if rng.random() < 0.06:  # changed a digit
                    marks.setdefault((i, int(rng.integers(rows)), c), erased_style(rng))
        elif kind in ("answers", "tf"):
            for r in range(rows):
                u = rng.random()
                chosen = [] if u < 0.07 else (list(rng.choice(cols, 2, replace=False)) if u < 0.11
                                              else [int(rng.integers(cols))])
                for c in chosen:
                    marks[(i, r, int(c))] = filled_style(rng)
                if rng.random() < 0.22:  # changed mind on another option
                    others = [c for c in range(cols) if c not in chosen]
                    if others:
                        marks[(i, r, int(rng.choice(others)))] = erased_style(rng)
        else:  # THPT part 3 number: optional sign/decimal point + digits from the left
            n_digits = int(rng.integers(1, 5))
            for c in range(n_digits):
                marks[(i, 2 + int(rng.integers(10)), c)] = filled_style(rng)
            if rng.random() < 0.2:
                marks[(i, 0, 0)] = filled_style(rng)
            if rng.random() < 0.25:
                marks[(i, 1, int(rng.choice([1, 2])))] = filled_style(rng)
            if rng.random() < 0.2:
                r, c = 2 + int(rng.integers(10)), int(rng.integers(4))
                marks.setdefault((i, r, c), erased_style(rng))
    return marks


# ---------------- Photo effects ----------------
def make_photo(page_gray, rng):
    """Returns (photo, M): M maps page pixel coordinates to photo pixel coordinates."""
    h, w = page_gray.shape
    pad = int(w * rng.uniform(0.05, 0.12))
    desk = np.array([rng.uniform(40, 120), rng.uniform(60, 140), rng.uniform(80, 170)], np.float32)
    canvas = np.empty((h + 2 * pad, w + 2 * pad, 3), np.float32)
    canvas[:] = desk + rng.normal(0, 6, canvas.shape)
    canvas[pad:pad + h, pad:pad + w] = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)
    H2, W2 = canvas.shape[:2]
    src = np.float32([[0, 0], [W2, 0], [W2, H2], [0, H2]])
    dst = (src + rng.uniform(-0.03, 0.03, src.shape) * [W2, H2]).astype(np.float32)
    persp = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(canvas, persp, (W2, H2), borderMode=cv2.BORDER_REPLICATE)
    # uneven light + colour tint (warm lamp / cool daylight)
    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.1), W2, dtype=np.float32)[None, :, None]
    gy = np.linspace(rng.uniform(0.85, 1.08), rng.uniform(0.75, 1.0), H2, dtype=np.float32)[:, None, None]
    tint = np.array([rng.uniform(0.85, 1.05), rng.uniform(0.95, 1.03), rng.uniform(0.97, 1.12)], np.float32)
    canvas = canvas * gx * gy * tint
    if rng.random() < 0.5:  # soft shadow (hand / phone)
        mask = np.zeros((H2, W2), np.float32)
        cv2.ellipse(mask, (int(rng.uniform(0, W2)), int(rng.uniform(0, H2))),
                    (int(W2 * rng.uniform(0.2, 0.5)), int(H2 * rng.uniform(0.15, 0.4))), rng.uniform(0, 180), 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), W2 * 0.06)
        canvas *= (1 - rng.uniform(0.15, 0.4) * mask)[:, :, None]
    canvas += rng.normal(0, rng.uniform(1.5, 5), canvas.shape)
    photo = np.clip(canvas, 0, 255).astype(np.uint8)
    blur = rng.choice(["none", "gauss", "motion"], p=[0.4, 0.4, 0.2])
    if blur == "gauss":
        photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.6, 1.4))
    elif blur == "motion":
        k = int(rng.integers(3, 6))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        photo = cv2.filter2D(photo, -1, cv2.warpAffine(kernel, rot, (k, k)))
    scale = rng.uniform(0.7, 1.0)  # lower camera resolution
    if scale < 0.98:
        photo = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
    to_photo = np.diag([scale, scale, 1.0]) @ persp @ np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], np.float64)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), to_photo


# ---------------- Python copy of the app pipeline ----------------
def app_gray(photo_bgr):
    """App: bitmapToMat gives RGBA, then COLOR_BGR2GRAY → R and B weights are swapped. Do the same."""
    return cv2.cvtColor(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB), cv2.COLOR_BGR2GRAY)


def run_app_pipeline(photo_bgr, layout, expected_corners=None, tolerance=0.01):
    """Returns (status, cells). status: "ok", "no_corners" (app rejects the photo) or
    "wrong_corners" (app would grade wrong positions — such sheets must not be used for training)."""
    gray = app_gray(photo_bgr)
    corners = find_corners(gray)
    if corners is None:
        return "no_corners", None
    if expected_corners is not None:
        error = np.linalg.norm(corners - expected_corners, axis=1).max()
        if error > tolerance * gray.shape[1]:
            return "wrong_corners", None
    m = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    warped = cv2.warpPerspective(gray, m, (int(W), int(H)))
    binary = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    cells = []
    for i, spec in enumerate(layout):
        x, y, w, h = spec["roi"]
        cw, ch = w / spec["cols"], h / spec["rows"]
        radius = int(min(cw, ch) * spec["r"])
        mask = np.zeros((int(ch), int(cw)), np.uint8)
        cv2.circle(mask, (int(round(cw / 2)), int(round(ch / 2))), radius, 255, -1)
        for r, c in cells_of(spec):
            rx, ry = x + int(c * cw), y + int(r * ch)
            cell_bin = binary[ry:ry + int(ch), rx:rx + int(cw)]
            filled = cv2.countNonZero(cv2.bitwise_and(cell_bin, mask[:cell_bin.shape[0], :cell_bin.shape[1]])) \
                > np.pi * radius * radius * 0.5
            cells.append((i, r, c, filled, warped[ry:ry + int(ch), rx:rx + int(cw)].copy()))
    return "ok", cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="folder with the app's form_*.pdf templates")
    ap.add_argument("--per-template", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--sample-photos", type=int, default=1, help="photos to keep per template for viewing")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "photos").mkdir(parents=True, exist_ok=True)
    zip_path = out / f"bubble_dataset_synth_{args.tag}.zip"
    labels, stats, skipped = {}, Counter(), Counter()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
        for template_id, (stem, layout) in sorted(LAYOUTS.items()):
            pix = fitz.open(Path(args.raw) / f"{stem}.pdf")[0].get_pixmap(dpi=200)
            base = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY if pix.n == 3 else cv2.COLOR_RGBA2GRAY)
            corners = find_corners(base)
            to_page = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), corners)
            scale = np.linalg.norm(corners[1] - corners[0]) / W

            for n in range(args.per_template):
                page = base.copy()
                marks = plan_marks(layout, template_id, rng)
                for (i, r, c), style in marks.items():
                    spec = layout[i]
                    x, y, w, h = spec["roi"]
                    cw, ch = w / spec["cols"], h / spec["rows"]
                    wx, wy = x + (c + 0.5) * cw, y + (r + 0.5) * ch
                    px, py = cv2.perspectiveTransform(np.float32([[[wx, wy]]]), to_page)[0, 0]
                    draw_mark(page, (px, py), min(cw, ch) * spec["r"] * scale, style, rng)
                photo, to_photo = make_photo(page, rng)
                expected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), to_photo).reshape(-1, 2)
                status, cells = run_app_pipeline(photo, layout, expected)
                if status != "ok":
                    skipped[status] += 1
                    continue
                if n < args.sample_photos:
                    cv2.imwrite(str(out / "photos" / f"synth_t{template_id}_{stem[5:]}_{n + 1}.jpg"), photo)
                sheet_id = f"990101-{template_id}{n:05d}-syn{args.seed % 1000:03d}"
                for i, r, c, filled, crop in cells:
                    region, position, option = cell_name_parts(layout[i], r, c)
                    name = f"{sheet_id}_t{template_id}_{region}_{position}_{option}.png"
                    auto = "filled" if filled else "empty"
                    style = marks.get((i, r, c))
                    truth = "empty" if style is None else ("erased" if style in ERASED_STYLES else "filled")
                    zf.writestr(f"bubble_dataset/{auto}/{name}", cv2.imencode(".png", crop)[1].tobytes())
                    labels[name] = truth
                    stats[(truth, auto)] += 1
                    if style:
                        stats[("style", style, auto)] += 1

    with (out / "labels_override.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(sorted(labels.items()))

    total = sum(v for k, v in stats.items() if k[0] != "style")
    made = args.per_template * len(LAYOUTS) - sum(skipped.values())
    lines = [f"Sheets: {made} used | skipped: {skipped['no_corners']} (app finds no corners), "
             f"{skipped['wrong_corners']} (app picks a WRONG corner → would grade wrong positions)",
             f"Cells: {total}"]
    for truth in ("empty", "filled", "erased"):
        n_all = stats[(truth, "empty")] + stats[(truth, "filled")]
        wrong = stats[(truth, "empty")] if truth == "filled" else stats[(truth, "filled")]
        lines.append(f"  {truth:>6}: {n_all:6d} cells — old 50% rule wrong on {wrong} ({wrong / max(n_all, 1):.1%})")
    lines.append("Old rule by mark style (cells read as 'filled' / total):")
    for style in FILLED_STYLES + ERASED_STYLES:
        f_, e_ = stats[("style", style, "filled")], stats[("style", style, "empty")]
        lines.append(f"  {style:>8}: {f_}/{f_ + e_}")
    report = "\n".join(lines)
    (out / "stats.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB) and {out / 'labels_override.csv'}")


if __name__ == "__main__":
    main()
