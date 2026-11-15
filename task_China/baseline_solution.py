"""Baseline for task_china

It shows the whole input/output contract:
  1. read the labelled rubbings (train_labels.csv) and the crate list (crates.csv);
  2. build one average template per crate glyph from the labelled boxes;
  3. slide the template over every rubbing of every set and count the strong matches;
  4. write counts.csv (crate, set, glyph_id, count).

    python baseline_solution.py --data dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TEMPLATE = 40          # templates are TEMPLATE x TEMPLATE pixels
THRESHOLD = 0.45       # normalised correlation needed to call a match


def build_template(data: Path, labels: pd.DataFrame, glyph_id: str, max_boxes: int = 300):
    """Average of the labelled boxes of one glyph, and the median size of those boxes."""
    rows = labels[labels.glyph_id == glyph_id].head(max_boxes)
    crops = []
    for name, group in rows.groupby("image"):
        img = cv2.imread(str(data / "train" / "images" / name), cv2.IMREAD_GRAYSCALE)
        for r in group.itertuples():
            x, y, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
            crop = img[y:y + h, x:x + w]
            if crop.size:
                crops.append(cv2.resize(crop, (TEMPLATE, TEMPLATE)).astype(np.float32))
    template = np.mean(crops, axis=0)
    return template.astype(np.uint8), float(rows.w.median()), float(rows.h.median())


def count_matches(img: np.ndarray, template: np.ndarray, w: float, h: float) -> int:
    """Number of separate places where the template correlates above THRESHOLD."""
    # bring the rubbing to the scale at which a typical glyph is TEMPLATE pixels wide/high
    fx, fy = TEMPLATE / w, TEMPLATE / h
    small = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    if small.shape[0] <= TEMPLATE or small.shape[1] <= TEMPLATE:
        return 0
    score = cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED)
    # non-maximum suppression: one match per template-sized neighbourhood
    peaks = (score == cv2.dilate(score, np.ones((TEMPLATE, TEMPLATE), np.uint8))) & (score >= THRESHOLD)
    return int(peaks.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("counts.csv"))
    a = ap.parse_args()

    labels = pd.read_csv(a.data / "train_labels.csv")      # image, x, y, w, h, glyph_id
    crates = pd.read_csv(a.data / "crates.csv")            # crate, glyph_id, modern_character
    print(f"{labels.image.nunique()} labelled rubbings, {len(labels)} boxes, "
          f"{labels.glyph_id.nunique()} glyph ids")

    rows = []
    for crate, glyph_id in zip(crates.crate, crates.glyph_id):
        template, w, h = build_template(a.data, labels, glyph_id)
        per_set = []
        for set_dir in sorted((a.data / "crates" / crate).iterdir()):
            n = 0
            for f in sorted(set_dir.glob("*.jpg")):
                n += count_matches(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), template, w, h)
            per_set.append(n)
            rows.append([crate, set_dir.name, glyph_id, n])
        print(crate, glyph_id, "counts per set:", per_set)

    pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"]).to_csv(a.out, index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
