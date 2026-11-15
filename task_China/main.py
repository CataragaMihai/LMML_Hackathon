"""China · Carved in bones: count each crate's glyph per set with a YOLO detector.

    python main.py train    # build YOLO dataset from train_labels.csv, fine-tune yolo11s
    python main.py count    # detect on crates/, write counts.csv, print the per-set counts
"""
import sys
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).parent
DS = HERE / "yolo_data"
RUN = HERE / "runs" / "glyphs"
crates = pd.read_csv(HERE / "crates.csv", encoding="utf-8-sig")
CLASSES = list(crates.glyph_id) + ["other"]  # 8 crate glyphs + every other glyph (helps the model tell them apart)


def build_dataset():
    labels = pd.read_csv(HERE / "train_labels.csv")
    images = sorted(labels.image.unique())
    val = set(images[::10])  # every 10th rubbing held out
    for split in ("train", "val"):
        (DS / split / "images").mkdir(parents=True, exist_ok=True)
        (DS / split / "labels").mkdir(parents=True, exist_ok=True)
    for name, g in labels.groupby("image"):
        split = "val" if name in val else "train"
        src = HERE / "train" / "images" / name
        H, W = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE).shape
        link = DS / split / "images" / name
        if not link.exists():
            link.symlink_to(src.resolve())
        lines = []
        for r in g.itertuples():
            c = CLASSES.index(r.glyph_id) if r.glyph_id in CLASSES else len(CLASSES) - 1
            lines.append(f"{c} {(r.x + r.w / 2) / W:.6f} {(r.y + r.h / 2) / H:.6f} {r.w / W:.6f} {r.h / H:.6f}")
        (DS / split / "labels" / name.replace(".jpg", ".txt")).write_text("\n".join(lines))
    (DS / "data.yaml").write_text(
        f"path: {DS.resolve()}\ntrain: train/images\nval: val/images\nnames: {dict(enumerate(CLASSES))}\n")


def train():
    from ultralytics import YOLO
    build_dataset()
    YOLO("yolo11s.pt").train(data=str(DS / "data.yaml"), epochs=25, imgsz=640, batch=16, device="mps",
                             project=str(RUN.parent), name=RUN.name, exist_ok=True,
                             fliplr=0.0, mosaic=1.0, seed=0)  # no flips: a mirrored glyph can be a different glyph


def calibrate():
    """Count error on the held-out rubbings per confidence threshold; pick the one with least error and ~0 bias."""
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(str(RUN / "weights" / "best.pt"))
    labels = pd.read_csv(HERE / "train_labels.csv")
    val = sorted(labels.image.unique())[::10]  # same held-out rubbings as build_dataset
    truth = np.array([[(labels[labels.image == n].glyph_id == g).sum() for g in crates.glyph_id] for n in val])
    preds = []
    for n in val:
        b = model.predict(str(HERE / "train" / "images" / n), imgsz=640, conf=0.05, device="mps", verbose=False)[0].boxes
        preds.append((b.cls.cpu().numpy(), b.conf.cpu().numpy()))
    for c in (0.25, 0.3, 0.35, 0.4, 0.5):
        pred = np.array([[((cl == k) & (cf >= c)).sum() for k in range(len(crates))] for cl, cf in preds])
        print(f"conf {c}: abs count error {np.abs(pred - truth).sum()}, bias {(pred - truth).sum():+d} "
              f"(true total {truth.sum()})")


def count(conf=0.35):  # 0.35 = least count error with ~0 bias on held-out rubbings (see calibrate)
    from ultralytics import YOLO
    model = YOLO(str(RUN / "weights" / "best.pt"))
    rows = []
    for crate, gid in zip(crates.crate, crates.glyph_id):
        cls = CLASSES.index(gid)
        for set_dir in sorted((HERE / "crates" / crate).iterdir()):
            n = 0
            for f in sorted(set_dir.glob("*.jpg")):
                b = model.predict(str(f), imgsz=640, conf=conf, device="mps", verbose=False)[0].boxes
                n += int((b.cls == cls).sum())
            rows.append([crate, set_dir.name, gid, n])
    df = pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"])
    df.to_csv(HERE / "counts.csv", index=False)
    print(df.pivot(index="crate", columns="set", values="count"))
    decode(df)


def decode(df):
    """Each set is a bit: many copies of the glyph (9-19) = 1, almost none (0-4) = 0.
    The 5 bits of a crate (set_1 first) are a number 1-26 = a letter."""
    word = ""
    for crate, g in df.groupby("crate"):
        bits = "".join("1" if n >= 7 else "0" for n in g.sort_values("set")["count"])  # 7 sits in the gap between 4 and 9
        word += chr(ord("A") + int(bits, 2) - 1)
        print(crate, bits, int(bits, 2), word[-1])
    print(f"SIGMOID_{word}")


if __name__ == "__main__":
    {"train": train, "calibrate": calibrate, "count": count,
     "decode": lambda: decode(pd.read_csv(HERE / "counts.csv"))}[sys.argv[1]]()
