import csv
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract
    _PYTESSERACT_IMPORT_ERROR = None
except ImportError as _e:
    pytesseract = None
    _PYTESSERACT_IMPORT_ERROR = str(_e)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE_DIR = os.path.join(SCRIPT_DIR, "SEMANALYSIS result")
DEFAULT_OUTPUT_NAME = "gray_analysis_output"
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\SEMANALYSIS\li\tesseract.exe",
    r"D:\SEMANALYSIS\li\Tesseract-OCR\tesseract.exe",
    r"D:\SEMANALYSIS\tesseract.exe",
    r"D:\SEMANALYSIS\Tesseract-OCR\tesseract.exe",
    r"C:\Users\mio\Downloads\code\VASP\tesseract.exe",
    r"C:\Users\mio\Downloads\code\VASP\Tesseract-OCR\tesseract.exe",
]

_TESSDATA_CANDIDATES = [
    r"D:\SEMANALYSIS\tessdata",
    r"D:\SEMANALYSIS\li\tessdata",
    r"C:\Program Files\Tesseract-OCR\tessdata",
]


MANUAL_POINT_OVERRIDE = {}
YELLOWNESS_THRESHOLD = 50
MIN_CROSS_AREA = 100
DILATE_KERNEL_SIZE = 7
MARKER_ROI_HALF_SIZE = 3
COLORED_PIXEL_THRESHOLD = 30
BACKGROUND_SMOOTHING_WINDOW = 5
DEFAULT_BACKGROUND_THRESHOLD = 9
OCR_BAND_FRACTION = 0.10
OCR_BAND_MIN_PX = 40
OCR_UPSCALE = 4
OCR_FIXED_BINARIZE_THRESHOLD = 150
OCR_PSM_LIST = [6, 4, 11]
OCR_MIN_CONF_WITH_CONTEXT = 55
OCR_MIN_CONF_OVERRIDE = 85
OCR_CONTEXT_BG_FRACTION = 0.5
OCR_BOX_PADDING = 4
SCALE_BAR_SEARCH_FRACTION = 0.15
SCALE_BAR_BRIGHT_THRESHOLD = 180
SCALE_BAR_MIN_RUN_LENGTH = 60
SCALE_BAR_MAX_THICKNESS = 12
SCALE_BAR_MIN_ASPECT_RATIO = 4.0
DEFAULT_INFO_BAR_MARGIN = 0
LABEL_SEARCH_LEFT = 120
LABEL_SEARCH_RIGHT = 160
LABEL_SEARCH_UP = 55
LABEL_SEARCH_DOWN = 3
LABEL_OCR_UPSCALE = 3

CLASS_COLORS = [
    (30, 100, 255),
    (0, 190, 0),
    (230, 30, 30),
    (255, 210, 0),
    (255, 140, 0),
    (160, 32, 240),
    (0, 210, 210),
    (255, 105, 180),
]

_LABEL_OCR_FAILED = False 

def _has_eng_data(tessdata_dir):
    return bool(tessdata_dir) and os.path.isfile(os.path.join(tessdata_dir, "eng.traineddata"))


def configure_tesseract():
    """
    Point pytesseract at a tesseract executable AND its language data.

    Tesseract needs both the exe and tessdata/eng.traineddata. A bare exe
    copied on its own fails with "Error opening data file ...
    eng.traineddata"; setting TESSDATA_PREFIX to a folder that holds the
    file fixes that. Returns (ocr_probably_works, status_text).
    """
    if pytesseract is None:
        return False, f"pytesseract not installed ({_PYTESSERACT_IMPORT_ERROR}) -- OCR disabled"

    found = [c for c in _TESSERACT_CANDIDATES if os.path.isfile(c)]
    on_path = shutil.which("tesseract")
    if on_path and on_path not in found:
        found.append(on_path)
    if not found:
        return False, "tesseract executable not found -- OCR disabled"

    for exe in found:
        tessdata = os.path.join(os.path.dirname(exe), "tessdata")
        if _has_eng_data(tessdata):
            pytesseract.pytesseract.tesseract_cmd = exe
            os.environ["TESSDATA_PREFIX"] = tessdata
            return True, f"{exe} (language data: {tessdata})"

    exe = found[0]
    pytesseract.pytesseract.tesseract_cmd = exe
    search = [os.environ.get("TESSDATA_PREFIX", ""),
              os.path.join(os.path.dirname(os.path.dirname(exe)), "tessdata"),
              os.path.join(SCRIPT_DIR, "tessdata")] + _TESSDATA_CANDIDATES
    for tessdata in search:
        if _has_eng_data(tessdata):
            os.environ["TESSDATA_PREFIX"] = tessdata
            return True, f"{exe} (language data: {tessdata})"

    existing = [d for d in search if d and os.path.isdir(d)]
    detail = ""
    if existing:
        d = existing[0]
        detail = (f" The folder {d} exists but has no eng.traineddata directly inside it "
                  f"(contents: {sorted(os.listdir(d))[:10] or 'empty'}).")
    return True, (f"{exe} -- WARNING: no eng.traineddata found, so OCR will probably fail."
                  f"{detail} Put eng.traineddata directly in a 'tessdata' folder "
                  r"(e.g. D:\SEMANALYSIS\tessdata\eng.traineddata).")

def _strip_quotes(s):
    s = s.strip()
    if s.startswith("& "):       
        s = s[2:].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s


def ask_existing_file(prompt, allow_empty=False):
    while True:
        path = _strip_quotes(input(prompt))
        if not path:
            if allow_empty:
                return None
            print("Path cannot be empty. Please try again.\n")
            continue
        if not os.path.isfile(path):
            print(f"File not found: {path}. Please check the path and try again.\n")
            continue
        return path


def get_output_dir_name(default_name):
    name = _strip_quotes(input(f"Enter a name for the output folder [default: {default_name}]: "))
    if not name:
        return default_name
    if ":" in name or "\\" in name or "/" in name:
        base = os.path.splitext(os.path.basename(name.rstrip("\\/")))[0]
        print(f"[NOTE] That looks like a path, not a plain name -- using '{base}' instead.")
        name = base
    cleaned = "".join(c for c in name if c not in '<>:"/\\|?*').strip()
    if not cleaned:
        print(f"That name wasn't valid, using default '{default_name}' instead.")
        cleaned = default_name
    return cleaned


def ask_int(prompt, default, lo, hi):
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print("Please enter a whole number.\n")
            continue
        if v < lo or v > hi:
            print(f"Must be between {lo} and {hi}.\n")
            continue
        return v


def detect_yellow_crosses(img_bgr):
    """Yellow '+' markers via relative yellowness min(R,G)-B (robust to JPEG
    desaturation); fragments merged by dilation; tiny blobs dropped."""
    b, g, r = cv2.split(img_bgr.astype(int))
    mask = ((np.minimum(r, g) - b) > YELLOWNESS_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8))
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(centroids[i]) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= MIN_CROSS_AREA]


def compute_colored_pixel_mask(img_bgr, threshold=COLORED_PIXEL_THRESHOLD):
    """True where R,G,B differ by more than JPEG noise -> overlay, not SEM signal."""
    b, g, r = cv2.split(img_bgr.astype(int))
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return spread > threshold


def robust_gray_at(gray_array, cx, cy, img_bgr, half=MARKER_ROI_HALF_SIZE):
    """Median gray of a small ROI around a cross, ignoring colored pixels
    (the cross itself / label text)."""
    h, w = gray_array.shape
    xi = int(round(np.clip(cx, 0, w - 1)))
    yi = int(round(np.clip(cy, 0, h - 1)))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    roi = gray_array[y0:y1, x0:x1]
    clean = roi[~compute_colored_pixel_mask(img_bgr[y0:y1, x0:x1])]
    return int(np.median(clean if clean.size else roi))


def detect_background_threshold(gray_array):
    """First histogram peak (black background) -> next valley = cutoff."""
    counts = np.bincount(gray_array.ravel(), minlength=256).astype(float)
    w = BACKGROUND_SMOOTHING_WINDOW
    smoothed = np.convolve(counts, np.ones(w) / w, mode="same")
    peak_idx = 0
    for i in range(1, 255):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peak_idx = i
            break
    for i in range(peak_idx + 1, 255):
        if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            return i, peak_idx
    print(f"[WARNING] Could not find a background valley; using {DEFAULT_BACKGROUND_THRESHOLD}.")
    return DEFAULT_BACKGROUND_THRESHOLD, peak_idx


def compute_background_mask(gray_array, threshold):
    """Low-gray pixels count as background ONLY if their connected region
    touches the image border; enclosed dark pores/gaps stay foreground."""
    low = (gray_array <= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    border_ids = set()
    if n > 1:
        for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            border_ids.update(edge.tolist())
    border_ids.discard(0)
    mask = np.isin(labels, list(border_ids)) if border_ids else np.zeros_like(low, dtype=bool)
    return mask, labels, stats, border_ids


def find_background_sample_location(labels, stats, border_ids):
    """Deepest interior point of the largest background region (for the BK check)."""
    if not border_ids:
        return None
    largest = max(border_ids, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    region = np.pad((labels == largest).astype(np.uint8), 1)
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    yi, xi = np.unravel_index(np.argmax(dist), dist.shape)
    return int(xi - 1), int(yi - 1)

def detect_scale_bar_line(gray_array):
    """Long, thin, bright horizontal run near the bottom = scale bar line."""
    h, w = gray_array.shape
    boxes = []
    for y in range(int(h * (1 - SCALE_BAR_SEARCH_FRACTION)), h):
        row = gray_array[y] >= SCALE_BAR_BRIGHT_THRESHOLD
        x = 0
        while x < w:
            if row[x]:
                start = x
                while x < w and row[x]:
                    x += 1
                if x - start >= SCALE_BAR_MIN_RUN_LENGTH:
                    boxes.append((start, y, x - start))
            else:
                x += 1
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y0 = max(0, min(b[1] for b in boxes) - 6)
    y1 = max(b[1] for b in boxes) + 6
    bw, bh = x1 - x0, y1 - y0
    if bh > SCALE_BAR_MAX_THICKNESS or bh == 0 or bw / bh < SCALE_BAR_MIN_ASPECT_RATIO:
        return None
    return x0, y0, bw, bh


def detect_annotation_mask_ocr(gray_array, background_mask):
    """OCR the bottom band (several binarisations x page modes) for MAG/HV/WD
    text; accept a hit only if confident AND surrounded by background, or
    very confident. Adds the scale-bar line. Raises on Tesseract failure."""
    h, w = gray_array.shape
    band_h = max(OCR_BAND_MIN_PX, int(h * OCR_BAND_FRACTION))
    y_off = max(0, h - band_h)
    band_big = cv2.resize(gray_array[y_off:, :], None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                          interpolation=cv2.INTER_CUBIC)
    _, bin_fixed = cv2.threshold(band_big, OCR_FIXED_BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, bin_otsu = cv2.threshold(band_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pad = OCR_BOX_PADDING
    mask = np.zeros_like(gray_array, dtype=bool)
    confirmed = {}
    for band_bin in (bin_fixed, bin_otsu):
        for psm in OCR_PSM_LIST:
            data = pytesseract.image_to_data(band_bin, config=f"--psm {psm}",
                                             output_type=pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                try:
                    conf = int(float(data["conf"][i]))
                except (ValueError, TypeError):
                    conf = -1
                if not text or conf < 30:
                    continue
                ox = int(data["left"][i] / OCR_UPSCALE)
                oy = int(y_off + data["top"][i] / OCR_UPSCALE)
                ow = max(1, int(data["width"][i] / OCR_UPSCALE))
                oh = max(1, int(data["height"][i] / OCR_UPSCALE))
                outer = np.zeros_like(mask)
                outer[max(0, oy - pad):min(h, oy + oh + pad), max(0, ox - pad):min(w, ox + ow + pad)] = True
                inner = np.zeros_like(mask)
                inner[oy:oy + oh, ox:ox + ow] = True
                ring = outer & ~inner
                if ring.sum() == 0:
                    continue
                bg_frac = background_mask[ring].mean()
                if (conf >= OCR_MIN_CONF_WITH_CONTEXT and bg_frac > OCR_CONTEXT_BG_FRACTION) \
                        or conf >= OCR_MIN_CONF_OVERRIDE:
                    key = (ox // 5, oy // 5)
                    if key not in confirmed or conf > confirmed[key]:
                        confirmed[key] = conf
                        mask[max(0, oy - 3):oy + oh + 3, max(0, ox - 3):ox + ow + 3] = True

    bar = detect_scale_bar_line(gray_array)
    if bar is not None:
        sx, sy, sw, sh = bar
        mask[sy:sy + sh, sx:sx + sw] = True
    return mask, len(confirmed), bar is not None


def save_bottom_ruler_preview(image_path, output_dir, max_fraction=0.25, step=20):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for m in range(step, max(step, int(h * max_fraction)) + 1, step):
        y = h - m
        draw.line([(0, y), (w, y)], fill=(0, 255, 255), width=1)
        draw.text((6, max(0, y - 18)), f"{m}px", fill=(0, 255, 255), font=font)
    path = os.path.join(output_dir, "bottom_ruler_preview.png")
    img.save(path)
    return path


def get_annotation_exclusion_mask(gray_array, background_mask, image_path, output_dir, ocr_ok):
    """OCR detection first; if unavailable/failed, manual ruler-guided bottom margin.
    Returns (mask, used_ocr, manual_margin_px)."""
    h = gray_array.shape[0]
    if ocr_ok:
        try:
            mask, n_text, has_bar = detect_annotation_mask_ocr(gray_array, background_mask)
            px = int(mask.sum())
            print(f"\n[OCR] Detected {n_text} instrument-text region(s)"
                  + (" and the scale bar line" if has_bar else " (no scale bar line found)")
                  + f" -- excluding {px:,} px ({px / gray_array.size * 100:.2f}% of image).")
            return mask, True, 0
        except Exception as e:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            print(f"\n[NOTE] OCR-based info-bar detection failed ({msg[:200]}).")
            print("       Falling back to manual info-bar exclusion.")
    else:
        print("\n[NOTE] OCR is unavailable, so the info bar must be excluded manually.")

    ruler = save_bottom_ruler_preview(image_path, output_dir)
    print(f"[OK] Ruler preview saved: {ruler}")
    print("     Open it and read off the line just above the top of the text/scale bar.")
    margin = ask_int(f"Pixels from the BOTTOM to exclude as the info bar "
                     f"[default: {DEFAULT_INFO_BAR_MARGIN} = none]: ",
                     DEFAULT_INFO_BAR_MARGIN, 0, h - 1)
    mask = np.zeros_like(background_mask)
    if margin > 0:
        mask[h - margin:, :] = True
    return mask, False, margin


def ask_supplementary_bottom_margin(mask):
    print("\nIf detected_crosses.png shows text/scale-bar pixels that OCR missed, "
          "you can add an extra bottom margin (combined with the OCR result).")
    extra = ask_int("Extra pixels to exclude from the BOTTOM [default: 0 = skip]: ",
                    0, 0, mask.shape[0] - 1)
    if extra > 0:
        mask = mask.copy()
        mask[mask.shape[0] - extra:, :] = True
    return mask, extra

def load_eds_table(xlsx_path):
    """Per-point mean Ti/(Ti+Fe) (atomic %) from the EDS export. Summary rows
    (Mean/Sigma...) are dropped by requiring the Spectrum name to end in .spx;
    repeat spectra of the same point are averaged (SD and n kept)."""
    df = pd.read_excel(xlsx_path)
    missing = [c for c in ("Spectrum", "Ti", "Fe") if c not in df.columns]
    if missing:
        raise ValueError(f"EDS sheet is missing column(s) {missing}; found {list(df.columns)}")
    df = df[df["Spectrum"].astype(str).str.endswith(".spx")].copy()
    df["point"] = df["Spectrum"].astype(str).str.extract(r"(\d+)\.spx$")[0]
    df = df[df["point"].notna()].copy()
    df["point"] = df["point"].astype(int)
    df["Ti"] = pd.to_numeric(df["Ti"], errors="coerce")
    df["Fe"] = pd.to_numeric(df["Fe"], errors="coerce")

    valid = df["Ti"].notna() & df["Fe"].notna() & (df["Ti"] >= 0) & (df["Fe"] >= 0) \
        & ((df["Ti"] + df["Fe"]) > 0)
    if (~valid).any():
        print(f"\n[WARNING] Dropping {(~valid).sum()} EDS row(s) with invalid Ti/Fe values:")
        for _, row in df[~valid].iterrows():
            print(f"  {row['Spectrum']}: Ti={row['Ti']}, Fe={row['Fe']}")
    df = df[valid].copy()
    df["Ti_ratio"] = df["Ti"] / (df["Ti"] + df["Fe"])

    agg = df.groupby("point")["Ti_ratio"].agg(["mean", "std", "count"])
    agg = agg.rename(columns={"mean": "Ti_ratio", "std": "Ti_ratio_sd", "count": "n_replicates"})
    agg["Ti_ratio_sd"] = agg["Ti_ratio_sd"].fillna(0.0)
    rep = agg[agg["n_replicates"] > 1]
    if len(rep):
        print(f"\n[NOTE] {len(rep)} point(s) had multiple EDS measurements; averaged per "
              "point. Repeat-measurement scatter (mean ± SD, n):")
        for pt, row in rep.iterrows():
            print(f"  point {pt}: {row['Ti_ratio']*100:.2f}% ± {row['Ti_ratio_sd']*100:.2f}% "
                  f"(n={int(row['n_replicates'])})")
    return agg


def read_point_number_near_cross(img_bgr, cx, cy):
    """
    OCR the green label above-left of a cross (e.g. '0811-1_001 3') and return
    the trailing point number, or None.

    The whitelist has no space (a quoted space in the config crashes
    pytesseract on Windows), so '001 3' is often read as '0013': the last
    digit group is taken, and if it is longer than 2 digits only its last
    digit is used. Misreads are caught by the duplicate check and can be
    fixed with MANUAL_POINT_OVERRIDE. Any Tesseract error is caught and
    reported once.
    """
    global _LABEL_OCR_FAILED
    if pytesseract is None or _LABEL_OCR_FAILED:
        return None
    b, g, r = cv2.split(img_bgr.astype(int))
    green = ((g > 100) & (r < 130) & (b < 130) & (g - r > 30)).astype(np.uint8) * 255
    h, w = green.shape
    x0, x1 = max(0, int(cx - LABEL_SEARCH_LEFT)), min(w, int(cx + LABEL_SEARCH_RIGHT))
    y0, y1 = max(0, int(cy - LABEL_SEARCH_UP)), min(h, int(cy - LABEL_SEARCH_DOWN))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = green[y0:y1, x0:x1]
    if crop.sum() == 0:
        return None
    crop = 255 - cv2.resize(crop, None, fx=LABEL_OCR_UPSCALE, fy=LABEL_OCR_UPSCALE,
                            interpolation=cv2.INTER_CUBIC)
    try:
        txt = pytesseract.image_to_string(
            crop, config="--psm 7 -c tessedit_char_whitelist=0123456789-_").strip()
    except Exception as e:
        _LABEL_OCR_FAILED = True
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        print(f"\n[WARNING] Label OCR failed ({msg[:200]}).")
        print("          Crosses not listed in MANUAL_POINT_OVERRIDE will be unmatched.")
        return None
    groups = re.findall(r"\d+", txt)
    if not groups:
        return None
    last = groups[-1]
    return int(last[-1]) if len(last) > 2 else int(last)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_detected_crosses_image(image_path, output_dir, gray_array, img_bgr, crosses,
                                point_labels, bk_location, info_bar_mask):
    """Original image + red circle per cross (class number, gray value, EDS
    point if known), cyan BK background check, magenta info-bar tint."""
    img = Image.open(image_path).convert("RGB")
    h, w = gray_array.shape
    if info_bar_mask.any():
        arr = np.array(img).astype(float)
        arr[info_bar_mask] = arr[info_bar_mask] * 0.45 + np.array([255, 0, 255]) * 0.55
        img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    font = _font(20)
    r = max(6, int(min(w, h) * 0.012))
    for idx, (cx, cy) in enumerate(crosses, start=1):
        gv = robust_gray_at(gray_array, cx, cy, img_bgr)
        x, y = int(round(cx)), int(round(cy))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=3)
        extra = f", pt{point_labels[idx - 1]}" if point_labels and point_labels[idx - 1] else ""
        text = f"{idx} (g={gv}{extra})"
        tw = draw.textlength(text, font=font) if hasattr(draw, "textlength") else 12 * len(text)
        tx = x + r + 4 if x + r + 4 + tw <= w else x - r - 4 - tw   # keep label inside image
        draw.text((tx, y - r - 4), text, fill=(255, 0, 0), font=font)
    if bk_location is not None:
        bx, by = bk_location
        draw.rectangle([bx - r, by - r, bx + r, by + r], outline=(0, 255, 255), width=3)
        draw.text((bx + r + 4, by - r - 4), f"BK (g={int(gray_array[by, bx])})",
                  fill=(0, 255, 255), font=font)
    path = os.path.join(output_dir, "detected_crosses.png")
    try:
        img.save(path)
    except OSError as e:
        path = os.path.join(output_dir, "detected_crosses_2.png")
        print(f"[WARNING] Could not overwrite detected_crosses.png ({e}); is it open "
              f"in another program? Saved to {path} instead.")
        img.save(path)
    return path


def save_analysis_mask_image(output_dir, foreground, background, info_bar, marker):
    rgb = np.zeros(foreground.shape + (3,), dtype=np.uint8)
    rgb[foreground] = (255, 255, 255)
    rgb[background] = (40, 40, 40)
    rgb[info_bar] = (255, 0, 255)
    rgb[marker & ~info_bar] = (0, 200, 255)
    path = os.path.join(output_dir, "analysis_mask.png")
    Image.fromarray(rgb).save(path)
    return path


def save_debug_crop_grid(img_bgr, entries, output_dir):
    """Zoomed crop of every cross with OCR reading and exact override key."""
    if not entries:
        return
    n = len(entries)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, (cx, cy, pt) in enumerate(entries):
        x0, x1 = max(0, int(cx - 160)), min(img_bgr.shape[1], int(cx + 160))
        y0, y1 = max(0, int(cy - 90)), min(img_bgr.shape[0], int(cy + 90))
        axes[i].imshow(cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"read: point={'unmatched' if pt is None else pt}\n"
                          f"key=({round(cx)}, {round(cy)})", fontsize=10)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    path = os.path.join(output_dir, "detected_crosses_debug.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[OK] Debug crop grid saved: {path}")


def classify_nearest_reference(values, refs):
    """Index of nearest reference gray value; ties go to the lower class."""
    values = np.asarray(values).astype(int)
    refs = np.asarray(refs).astype(int)
    return np.argmin(np.abs(values[:, None] - refs[None, :]), axis=1)


def make_classifier(num_classes, cross_values):
    """Return (classify_fn, method, exact (lo, hi) gray range per class)."""
    if len(cross_values) == num_classes and num_classes >= 1:
        method = "nearest_reference"

        def fn(v):
            return classify_nearest_reference(np.ravel(v), cross_values)
    else:
        method = "equal_width"
        edges = np.round(np.linspace(0, 256, num_classes + 1)).astype(int)

        def fn(v):
            idx = np.searchsorted(edges, np.ravel(v), side="right") - 1
            return np.clip(idx, 0, num_classes - 1)

    per_level = fn(np.arange(256))
    ranges = []
    for c in range(num_classes):
        levels = np.where(per_level == c)[0]
        ranges.append((int(levels.min()), int(levels.max())) if levels.size else (None, None))
    return fn, method, ranges


def fmt_range(rng):
    return "empty" if rng[0] is None else f"{rng[0]}-{rng[1]}"


def save_histogram(output_dir, fg_values, ranges, pcts):
    num_classes = len(ranges)
    bins = np.arange(257) - 0.5
    counts, _ = np.histogram(fg_values, bins=bins)
    max_h = counts.max() if counts.size and counts.max() > 0 else 1

    plt.figure(figsize=(10, 6))
    plt.hist(fg_values, bins=bins, color="gray")
    plt.title(f"Grayscale histogram (background/info bar/markers excluded, {num_classes} classes)")
    plt.xlabel("Gray value (0-255)")
    plt.ylabel("Pixel count")
    stagger = 0
    for c, (lo, hi) in enumerate(ranges):
        if lo is None:
            continue
        if c > 0:
            plt.axvline(lo - 0.5, color="red", linestyle="--", linewidth=1)
        local_max = counts[lo:hi + 1].max()
        if hi - lo + 1 < 15:
            stagger += 1
            extra = max_h * 0.10 * (stagger % 2 + 1)
        else:
            stagger = 0
            extra = max_h * 0.03
        plt.text((lo + hi) / 2, local_max + extra, f"{pcts[c]:.1f}%", ha="center",
                 va="bottom", fontsize=10, color="red", fontweight="bold")
    plt.ylim(top=max_h * 1.35)
    path = os.path.join(output_dir, "histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_class_map(output_dir, gray_array, foreground, classify, num_classes, class_labels):
    class_map = np.full(gray_array.shape, -1, dtype=int)
    class_map[foreground] = classify(gray_array[foreground])
    palette = np.array([CLASS_COLORS[c % len(CLASS_COLORS)] for c in range(num_classes)]) / 255.0
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("black")
    im = ax.imshow(np.ma.masked_where(class_map < 0, class_map), cmap=ListedColormap(palette),
                   vmin=-0.5, vmax=num_classes - 0.5)
    cbar = plt.colorbar(im, ax=ax, ticks=range(num_classes))
    cbar.ax.set_yticklabels(class_labels, color="white")
    cbar.ax.tick_params(colors="white")
    ax.set_title("Gray-value classification map\n(black = excluded background/info bar/markers)",
                 color="white")
    ax.axis("off")
    path = os.path.join(output_dir, "classified_image.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    return path


def run_calibration(calib, gray_array, foreground, output_dir):
    """Fit, validate and map Ti/(Ti+Fe). Returns a dict of results for the JSON."""
    from scipy import stats as spstats

    n = len(calib)
    g = calib["gray"].to_numpy(dtype=float)
    y = calib["Ti_ratio"].to_numpy(dtype=float)
    res = {"n_points": n}

    if n < 5:
        print(f"\n[WARNING] Only {n} calibration point(s). A line through so few points "
              "is a workflow demonstration, not a defensible calibration. 5+ points "
              "spanning a wide composition and gray range is a reasonable minimum.")

    fit = spstats.linregress(g, y)
    slope, intercept, r, p = fit.slope, fit.intercept, fit.rvalue, fit.pvalue
    r2 = r ** 2
    print(f"\nLinear calibration: Ti/(Ti+Fe) = {slope:.6e} * gray + {intercept:.4f}")
    print(f"R = {r:.4f}   R^2 = {r2:.4f}   p-value = {p:.4f}   n = {n}")
    if not np.isfinite(p) or p > 0.05:
        print("[CAUTION] p > 0.05 -- the gray/composition relationship is not statistically "
              "significant here. Do not present the map as a validated composition measurement.")

    resid = y - (slope * g + intercept)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    print(f"In-sample fit: RMSE = {rmse*100:.2f} pct-points, MAE = {mae*100:.2f} pct-points")

    loocv_rmse = None
    if n >= 3:
        errs = []
        for i in range(n):
            m = np.arange(n) != i
            if np.unique(g[m]).size < 2:
                continue
            f_i = spstats.linregress(g[m], y[m])
            errs.append(f_i.slope * g[i] + f_i.intercept - y[i])
        if errs:
            loocv_rmse = float(np.sqrt(np.mean(np.square(errs))))
            print(f"Leave-one-out cross-validation RMSE = {loocv_rmse*100:.2f} pct-points "
                  "(out-of-sample error -- the more honest number to quote)")
            if loocv_rmse > 2 * rmse:
                print("[CAUTION] LOOCV RMSE is much larger than the in-sample RMSE -- the line "
                      "depends heavily on which points are included (overfitting risk).")
    else:
        print("(Leave-one-out cross-validation skipped: needs at least 3 points.)")

    res.update({"slope": float(slope), "intercept": float(intercept), "r": float(r),
                "r2": float(r2), "p_value": float(p), "rmse_in_sample": rmse,
                "mae_in_sample": mae, "loocv_rmse": loocv_rmse,
                "gray_range": [int(g.min()), int(g.max())],
                "ti_ratio_range": [float(y.min()), float(y.max())]})

    plt.figure(figsize=(6, 5))
    plt.scatter(g, y * 100, color="black", zorder=3)
    for _, row in calib.iterrows():
        plt.annotate(f'pt{int(row["point"])}', (row["gray"], row["Ti_ratio"] * 100 + 0.2))
    xs = np.linspace(g.min() - 5, g.max() + 5, 50)
    plt.plot(xs, (slope * xs + intercept) * 100, "r--", label=f"R²={r2:.2f}, p={p:.3f}, n={n}")
    plt.xlabel("Gray value at marker (colored pixels excluded)")
    plt.ylabel("EDS Ti/(Ti+Fe) [%]")
    plt.title("Calibration: gray value vs. EDS composition")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "calibration_fit.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[OK] Calibration plot saved: {path}")

    pred_pct = (slope * gray_array.astype(float) + intercept) * 100
    fg = pred_pct[foreground]
    print(f"\n===== Estimated Ti/(Ti+Fe) over foreground ({foreground.sum():,} px) =====")
    print("(Spatial variation of the ESTIMATE across pixels -- not the uncertainty of any "
          "single estimate; see RMSE/LOOCV above for that.)")
    print(f"Spatial mean: {fg.mean():.3f}%")
    print(f"Spatial SD:   {fg.std():.3f}%")
    if fg.mean() != 0:
        print(f"Spatial CV:   {fg.std() / fg.mean() * 100:.2f}%")
    print(f"Min/Max in image: {fg.min():.2f}% / {fg.max():.2f}%")

    below, above = int((fg < 0).sum()), int((fg > 100).sum())
    if below or above:
        print(f"[NOTE] {below:,} px < 0% and {above:,} px > 100% from the raw linear "
              "formula (clipped to 0-100 for display).")
    extrap = foreground & ((gray_array < g.min()) | (gray_array > g.max()))
    n_ext = int(extrap.sum())
    ext_pct = n_ext / foreground.sum() * 100 if foreground.sum() else 0.0
    if n_ext:
        print(f"[NOTE] {n_ext:,} px ({ext_pct:.1f}% of foreground) have gray values outside the "
              f"calibration range ({int(g.min())}-{int(g.max())}) -- EXTRAPOLATIONS, shown with "
              "a solid gray overlay rather than a color.")
    res.update({"estimate_spatial_mean_pct": float(fg.mean()),
                "estimate_spatial_sd_pct": float(fg.std()),
                "extrapolated_pixels": n_ext, "extrapolated_percent": ext_pct})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("black")
    im = ax.imshow(np.ma.masked_where(~foreground | extrap, np.clip(pred_pct, 0, 100)),
                   cmap="viridis", vmin=max(0, y.min() * 100 - 2), vmax=min(100, y.max() * 100 + 2))
    if n_ext:
        ax.imshow(np.ma.masked_where(~extrap, np.ones_like(gray_array)), cmap="gray",
                  vmin=0, vmax=1, alpha=0.75)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Estimated Ti/(Ti+Fe) [%] (interpolated only)", color="white")
    cbar.ax.tick_params(colors="white")
    for _, row in calib.iterrows():
        ax.plot(row["x"], row["y"], "r+", markersize=12, markeredgewidth=2)
        ax.annotate(f'pt{int(row["point"])}', (row["x"], row["y"]), color="red", fontsize=9,
                    xytext=(5, 5), textcoords="offset points")
    extra = f", {ext_pct:.0f}% of pixels gray = extrapolated" if n_ext else ""
    ax.set_title(f"Estimated Ti/(Ti+Fe) map\n(linear, R²={r2:.2f}, n={n}{extra} -- see caveats)",
                 color="white", fontsize=11)
    ax.axis("off")
    path = os.path.join(output_dir, "composition_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"[OK] Composition map saved: {path}")

    # --- per-pixel 95% prediction interval
    if n >= 3:
        dof = n - 2
        s = float(np.sqrt(np.sum(resid ** 2) / dof))
        gm = float(g.mean())
        sxx = float(np.sum((g - gm) ** 2))
        t = float(spstats.t.ppf(0.975, dof))
        half = t * s * np.sqrt(1 + 1 / n + (gray_array.astype(float) - gm) ** 2 / sxx) * 100
        half_disp = np.ma.masked_where(~foreground, half)
        med = float(np.ma.median(half_disp))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("black")
        im = ax.imshow(half_disp, cmap="magma")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("95% prediction interval half-width [pct-points]", color="white")
        cbar.ax.tick_params(colors="white")
        for _, row in calib.iterrows():
            ax.plot(row["x"], row["y"], "c+", markersize=12, markeredgewidth=2)
        ax.set_title("Per-pixel prediction uncertainty\n(residual scatter + distance from the "
                     "calibration's mean gray value)", color="white", fontsize=11)
        ax.axis("off")
        path = os.path.join(output_dir, "uncertainty_map.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close()
        print(f"[OK] Uncertainty map saved: {path}")
        print(f"     (median 95% PI half-width over foreground: ±{med:.2f} pct-points -- "
              "compare with the spatial SD before calling two regions different)")
        res["median_pi_halfwidth_pct"] = med
    else:
        print("[NOTE] Uncertainty map skipped: needs at least 3 calibration points.")

    loocv_line = (f"   LOOCV RMSE was {loocv_rmse*100:.2f} pct-points -- quote that, not R², as "
                  "the\n   calibration's predictive error.\n" if loocv_rmse is not None else "")
    print("\n" + "=" * 70)
    print("CAVEATS -- read before using these composition numbers:")
    print("=" * 70)
    print(f"""
1. n = {n} calibration points from ONE image, covering only
   {y.min()*100:.1f}-{y.max()*100:.1f}% Ti/(Ti+Fe) and gray {int(g.min())}-{int(g.max())}.
   That is a workflow demonstration, not a validated calibration.
{loocv_line}2. Valid only for THIS image and THIS SEM session's brightness/contrast.
3. Assumes gray value reflects composition (Z-contrast) only. Unpolished
   powder adds topographic contrast (edges, tilt, shadows) to the same signal.
4. Gray-overlaid pixels are extrapolations: the calibration says nothing there.
5. Effective sample size is n = {n}, not the number of pixels.
6. The EDS interaction volume (~1 um or more) is much larger than a pixel.
7. Report as "EDS-calibrated grayscale estimate", not "measured composition".
""")
    return res

def main():
    print("=" * 70)
    print(" SEM grayscale analysis: area fractions + EDS-calibrated Ti/(Ti+Fe)")
    print("=" * 70)
    ocr_ok, ocr_status = configure_tesseract()
    print(f"[OCR] Tesseract: {ocr_status}\n")

    image_path = ask_existing_file("Enter the image file path (you can drag & drop the file here): ")
    xlsx_path = ask_existing_file("Enter the EDS spreadsheet (.xlsx) path "
                                  "[press Enter to skip = area analysis only]: ", allow_empty=True)
    default_name = os.path.splitext(os.path.basename(image_path))[0] or DEFAULT_OUTPUT_NAME
    output_dir = os.path.join(RESULTS_BASE_DIR, get_output_dir_name(default_name))
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput folder: {output_dir}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not read the image: {image_path}")
        sys.exit(1)
    gray_array = np.array(Image.open(image_path).convert("L"))
    if gray_array.shape != img_bgr.shape[:2]:
        print("[ERROR] Image size mismatch between readers; try re-saving the image as PNG/JPG.")
        sys.exit(1)
    height, width = gray_array.shape
    total_pixels = height * width

    eds = None
    if xlsx_path:
        try:
            eds = load_eds_table(xlsx_path)
            print(f"Loaded {len(eds)} EDS point(s) from {os.path.basename(xlsx_path)}: "
                  f"{sorted(eds.index.tolist())}")
        except Exception as e:
            print(f"\n[WARNING] Could not read the EDS spreadsheet ({e}). "
                  "Continuing with the area analysis only.")
            eds = None

    crosses = detect_yellow_crosses(img_bgr)
    crosses = sorted(crosses, key=lambda c: robust_gray_at(gray_array, c[0], c[1], img_bgr))
    print(f"\nDetected {len(crosses)} yellow cross marker(s).")
    while True:
        num_classes = ask_int(f"Press Enter to use {len(crosses)} classes, or type a different number: ",
                              len(crosses), 1, 64) if crosses else \
            ask_int("No crosses found. How many equal-width classes? [default: 4]: ", 4, 1, 64)
        if crosses and num_classes != len(crosses):
            print(f"\n[WARNING] {num_classes} classes vs {len(crosses)} crosses: classes will be an "
                  "equal-width split of 0-255, not based on the crosses' gray values.")
            if input("Type 'yes' to continue, or press Enter to re-enter: ").strip().lower() != "yes":
                continue
        break
    cross_values = [robust_gray_at(gray_array, cx, cy, img_bgr) for cx, cy in crosses]

    point_of_cross = [None] * len(crosses)
    if eds is not None and crosses:
        for i, (cx, cy) in enumerate(crosses):
            key = (round(cx), round(cy))
            point_of_cross[i] = MANUAL_POINT_OVERRIDE[key] if key in MANUAL_POINT_OVERRIDE \
                else read_point_number_near_cross(img_bgr, cx, cy)

    auto_thr, peak_idx = detect_background_threshold(gray_array)
    print(f"\nBackground peak at gray {peak_idx}; background/foreground valley at {auto_thr}.")
    bg_thr = ask_int(f"Press Enter to use gray <= {auto_thr} as background, or type another value: ",
                     auto_thr, 0, 254)
    background_mask, bg_labels, bg_stats, border_ids = compute_background_mask(gray_array, bg_thr)
    bk_location = find_background_sample_location(bg_labels, bg_stats, border_ids)

    info_bar_mask, used_ocr, manual_margin = get_annotation_exclusion_mask(
        gray_array, background_mask, image_path, output_dir, ocr_ok)
    marker_mask = compute_colored_pixel_mask(img_bgr)

    labels_for_image = [p if (eds is not None and p in eds.index) else None for p in point_of_cross]
    path = save_detected_crosses_image(image_path, output_dir, gray_array, img_bgr, crosses,
                                       labels_for_image, bk_location, info_bar_mask)
    print(f"\n[OK] Cross-detection image saved: {path}")
    print("     (red = crosses numbered dimmest->brightest; cyan BK must sit on black background; "
          "magenta = excluded info bar)")
    extra_margin = 0
    if used_ocr:
        info_bar_mask, extra_margin = ask_supplementary_bottom_margin(info_bar_mask)
        if extra_margin:
            path = save_detected_crosses_image(image_path, output_dir, gray_array, img_bgr, crosses,
                                               labels_for_image, bk_location, info_bar_mask)
            print(f"[OK] Updated with the extra margin: {path}")

    foreground = ~(background_mask | info_bar_mask | marker_mask)
    n_fg = int(foreground.sum())
    n_bg = int(background_mask.sum())
    n_bg_outside_bar = int((background_mask & ~info_bar_mask).sum())
    n_internal_low = int((gray_array <= bg_thr).sum()) - n_bg
    n_bar = int(info_bar_mask.sum())
    n_marker = int((marker_mask & ~info_bar_mask & ~background_mask).sum())

    print(f"\nImage size: {width} x {height}")
    print(f"  Background (gray <= {bg_thr}, connected to border): {n_bg:,} px "
          f"({n_bg / total_pixels * 100:.2f}%)")
    if n_internal_low:
        print(f"  Dark pixels ENCLOSED in sample (pores/gaps, kept as foreground): {n_internal_low:,} px "
              f"({n_internal_low / total_pixels * 100:.2f}%)")
    if n_bar:
        print(f"  Info bar / scale bar: {n_bar:,} px ({n_bar / total_pixels * 100:.2f}%)")
    if n_marker:
        print(f"  Colored marker/label pixels: {n_marker:,} px ({n_marker / total_pixels * 100:.2f}%)")
    print(f"  Foreground analysed: {n_fg:,} px ({n_fg / total_pixels * 100:.2f}%)")

    path = save_analysis_mask_image(output_dir, foreground, background_mask, info_bar_mask, marker_mask)
    print(f"[OK] Analysis mask saved: {path}  (white = counted pixels)")

    if n_fg == 0:
        print("\n[ERROR] No foreground pixels left. Try a lower background threshold.")
        sys.exit(1)

    fg_values = gray_array[foreground]
    mean_g, med_g, sd_g = float(fg_values.mean()), float(np.median(fg_values)), float(fg_values.std())
    print("\n===== Foreground gray statistics =====")
    print(f"Mean {mean_g:.2f}   Median {med_g:.2f}   SD {sd_g:.2f}   "
          f"Min {fg_values.min()}   Max {fg_values.max()}")

    classify, method, ranges = make_classifier(num_classes, cross_values)
    if method == "nearest_reference" and len(set(cross_values)) < len(cross_values):
        print("[WARNING] Two or more crosses have the same gray value; one of those classes "
              "will be empty.")
    fg_classes = classify(fg_values)
    class_counts = [int((fg_classes == c).sum()) for c in range(num_classes)]
    class_pcts = [cnt / n_fg * 100 for cnt in class_counts]

    class_point, class_ti, class_ref = [], [], []
    for c in range(num_classes):
        if method == "nearest_reference":
            pt = point_of_cross[c]
            ok = eds is not None and pt is not None and pt in eds.index
            class_point.append(pt if ok else None)
            class_ti.append(float(eds.loc[pt, "Ti_ratio"]) if ok else None)
            class_ref.append(cross_values[c])
        else:
            class_point.append(None)
            class_ti.append(None)
            class_ref.append(None)

    print(f"\n===== Area fractions by class ({method}) =====")
    if method == "nearest_reference":
        print(f"Reference gray values (one per cross): {cross_values}")
    if n_internal_low:
        print(f"[NOTE] The lowest class includes the {n_internal_low:,} enclosed dark "
              "(pore/gap) pixels.")
    class_labels = []
    for c in range(num_classes):
        tag = ""
        if class_point[c] is not None:
            tag = f"  [pt{class_point[c]}, EDS Ti/(Ti+Fe)={class_ti[c]*100:.2f}%]"
        print(f"Class {c + 1} (gray {fmt_range(ranges[c])}): {class_counts[c]:,} px, "
              f"{class_pcts[c]:.2f}%{tag}")
        class_labels.append(f"Class {c + 1}" + (f" (pt{class_point[c]})" if class_point[c] else ""))

    path = save_histogram(output_dir, fg_values, ranges, class_pcts)
    print(f"\n[OK] Histogram saved: {path}")
    path = save_class_map(output_dir, gray_array, foreground, classify, num_classes, class_labels)
    print(f"[OK] Classification map saved: {path}")

    path = os.path.join(output_dir, "results.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "width", "height", "background_threshold", "background_pixels",
                     "info_bar_pixels", "marker_pixels", "foreground_pixels",
                     "internal_low_gray_pixels", "classification_method", "class_index",
                     "reference_gray", "class_gray_low", "class_gray_high", "class_pixels",
                     "class_percent", "eds_point", "eds_Ti_ratio_percent",
                     "mean_gray", "median_gray", "sd_gray"])
        for c in range(num_classes):
            wr.writerow([os.path.basename(image_path), width, height, bg_thr, n_bg, n_bar, n_marker,
                         n_fg, n_internal_low, method, c + 1,
                         "" if class_ref[c] is None else class_ref[c],
                         "" if ranges[c][0] is None else ranges[c][0],
                         "" if ranges[c][1] is None else ranges[c][1],
                         class_counts[c], f"{class_pcts[c]:.4f}",
                         "" if class_point[c] is None else class_point[c],
                         "" if class_ti[c] is None else f"{class_ti[c]*100:.3f}",
                         f"{mean_g:.3f}", f"{med_g:.3f}", f"{sd_g:.3f}"])
    print(f"[OK] Results CSV saved: {path}")

    calibration = {"status": "skipped (no EDS spreadsheet)"}
    if eds is not None:
        print("\n" + "=" * 70)
        print(" PART B: EDS-calibrated Ti/(Ti+Fe) estimate")
        print("=" * 70)
        rows, unmatched = [], []
        for i, (cx, cy) in enumerate(crosses):
            pt = point_of_cross[i]
            if pt is None or pt not in eds.index:
                unmatched.append((cx, cy, pt))
            else:
                rows.append({"point": int(pt), "x": cx, "y": cy, "gray": cross_values[i],
                             "Ti_ratio": float(eds.loc[pt, "Ti_ratio"]),
                             "Ti_ratio_sd": float(eds.loc[pt, "Ti_ratio_sd"]),
                             "n_replicates": int(eds.loc[pt, "n_replicates"])})
        calib = pd.DataFrame(rows, columns=["point", "x", "y", "gray", "Ti_ratio",
                                            "Ti_ratio_sd", "n_replicates"])
        stop_reason = None

        if unmatched:
            print(f"\n[WARNING] {len(unmatched)} cross(es) could not be matched to an EDS point:")
            for cx, cy, pt in unmatched:
                why = "not read" if pt is None else f"read as {pt}, not in spreadsheet"
                print(f"  key=({round(cx)}, {round(cy)})  ({why})")
            print("  Fix: check detected_crosses_debug.png and fill MANUAL_POINT_OVERRIDE at the "
                  "top of this script, e.g. {(631, 283): 1, ...}, or fix the Tesseract install.")

        dups = sorted(calib["point"][calib["point"].duplicated(keep=False)].unique().tolist())
        if dups:
            print(f"\n[ERROR] Point number(s) {dups} were matched to MORE THAN ONE cross -- at "
                  "least one label was misread:")
            for _, row in calib[calib["point"].isin(dups)].iterrows():
                print(f"  point {int(row['point'])} at key=({round(row['x'])}, {round(row['y'])})")
            stop_reason = "duplicate point numbers (OCR misread) -- use MANUAL_POINT_OVERRIDE"
        elif len(calib) < 2:
            stop_reason = f"only {len(calib)} cross(es) matched to EDS points (need at least 2)"
        elif calib["gray"].nunique() < 2:
            stop_reason = "all matched crosses have the same gray value (no line can be fit)"

        if unmatched or dups:
            entries = [(r["x"], r["y"], int(r["point"])) for _, r in calib.iterrows()] + unmatched
            save_debug_crop_grid(img_bgr, entries, output_dir)

        if stop_reason:
            print(f"\n[SKIPPED] Calibration not run: {stop_reason}.")
            print("          Part A (area fractions) above is complete and unaffected.")
            calibration = {"status": f"skipped: {stop_reason}"}
        else:
            calib = calib.sort_values("point")
            print("\n===== Calibration points =====")
            print(calib[["point", "x", "y", "gray", "Ti_ratio", "Ti_ratio_sd", "n_replicates"]]
                  .to_string(index=False, formatters={
                      "x": "{:.0f}".format, "y": "{:.0f}".format,
                      "Ti_ratio": lambda v: f"{v*100:.2f}%",
                      "Ti_ratio_sd": lambda v: f"{v*100:.2f}%"}))
            cpath = os.path.join(output_dir, "calibration_points.csv")
            out = calib.copy()
            out["Ti_ratio_percent"] = out["Ti_ratio"] * 100
            out["Ti_ratio_sd_percent"] = out["Ti_ratio_sd"] * 100
            out[["point", "x", "y", "gray", "Ti_ratio_percent", "Ti_ratio_sd_percent",
                 "n_replicates"]].to_csv(cpath, index=False, encoding="utf-8-sig",
                                         float_format="%.3f")
            print(f"[OK] Calibration points saved: {cpath}")
            calibration = {"status": "done", **run_calibration(calib, gray_array, foreground,
                                                               output_dir)}
            calibration["points"] = [{"point": int(r["point"]), "x": round(float(r["x"]), 1),
                                      "y": round(float(r["y"]), 1), "gray": int(r["gray"]),
                                      "Ti_ratio": float(r["Ti_ratio"])}
                                     for _, r in calib.iterrows()]

    meta = {
        "image": os.path.basename(image_path),
        "image_path": os.path.abspath(image_path),
        "eds_file": os.path.abspath(xlsx_path) if xlsx_path else None,
        "image_size": {"width": width, "height": height},
        "ocr": {"tesseract": ocr_status, "used_for_info_bar": used_ocr,
                "manual_info_bar_margin_px": manual_margin,
                "extra_bottom_margin_px": extra_margin,
                "manual_point_override": {f"{k[0]},{k[1]}": v for k, v in MANUAL_POINT_OVERRIDE.items()}},
        "num_crosses_detected": len(crosses),
        "cross_positions": [[round(float(x), 1), round(float(y), 1)] for x, y in crosses],
        "cross_gray_values": cross_values,
        "cross_eds_points": point_of_cross,
        "num_classes": num_classes,
        "classification_method": method,
        "class_gray_ranges": [list(r) for r in ranges],
        "background_threshold": bg_thr,
        "background_threshold_auto": auto_thr,
        "background_peak_gray": peak_idx,
        "settings": {"colored_pixel_threshold": COLORED_PIXEL_THRESHOLD,
                     "marker_roi_half_size": MARKER_ROI_HALF_SIZE,
                     "yellowness_threshold": YELLOWNESS_THRESHOLD,
                     "min_cross_area": MIN_CROSS_AREA},
        "pixel_counts": {
            "total": total_pixels,
            "background": n_bg,
            "background_outside_info_bar": n_bg_outside_bar,
            "internal_low_gray_kept_as_foreground": n_internal_low,
            "info_bar": n_bar,
            "marker_outside_background_and_info_bar": n_marker,
            "foreground_analyzed": n_fg,
        },
        "foreground_gray_stats": {"mean": mean_g, "median": med_g, "sd": sd_g},
        "class_pixel_counts": class_counts,
        "class_percentages": class_pcts,
        "calibration": calibration,
    }
    path = os.path.join(output_dir, "analysis_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n[OK] Analysis metadata saved: {path}")
    print(f"\nAll done! Results are in: {output_dir}")


if __name__ == "__main__":
    main()
