# China · Carved in bones, engraved in the heart

**Flag:** `SIGMOID_FENGSHUI`

## Run
Put the task data (`train/`, `crates/`, the CSVs) in this folder. Then:
```bash
pip install -r requirements.txt ultralytics
python main.py train       # optional, ~2 h on an M2: fine-tunes YOLO11s, writes runs/glyphs/weights/best.pt (committed)
python main.py calibrate   # count error vs confidence threshold on the held-out rubbings
python main.py count       # detects on all 1,200 crate rubbings, writes counts.csv, prints the flag
```

## Approach
1. **Detector, not templates.** The baseline averages one template per glyph and slides it over the rubbing. Rubbings differ in scale, ink, rotation and carving hand, so it misses and over-counts. I fine-tuned a YOLO11s detector on the 14,149 labelled boxes with 9 classes: the 8 crate glyphs, plus `other` for every other glyph (including `rare`), so the model learns what the crate glyph is *not*. Horizontal flips are turned off because a mirrored glyph can be a different character. Every 10th rubbing (180 total) is held out.
2. **Threshold for counting.** On the held-out rubbings, conf 0.35 gives the smallest count error with about zero bias (-3 out of 351 crate glyphs). Validation for the 8 crate glyphs: P 0.90, R 0.81, mAP50 0.88.
3. **Counts → letters.** Each set's count is either high (9–19) or low (0–4). The gap holds at every threshold from 0.25 to 0.5. Each set is one bit (high = 1), set_1 is the most significant bit, and 5 bits give a number 1–26, which is a letter (A = 1).

| crate | glyph | set 1–5 counts | bits | n | letter |
|---|---|---|---|---|---|
| crate_01 | g015 賓 | 1 1 14 17 1 | 00110 | 6 | F |
| crate_02 | g028 今 | 1 0 15 0 12 | 00101 | 5 | E |
| crate_03 | g005 其 | 0 14 16 17 2 | 01110 | 14 | N |
| crate_04 | g018 㞢 | 1 1 15 13 15 | 00111 | 7 | G |
| crate_05 | g003 王 | 18 4 0 15 16 | 10011 | 19 | S |
| crate_06 | g011 不 | 0 10 2 2 2 | 01000 | 8 | H |
| crate_07 | g001 貞 | 13 2 14 1 17 | 10101 | 21 | U |
| crate_08 | g010 雨 | 1 11 1 1 16 | 01001 | 9 | I |

"The bones answer whoever counts carefully enough": a weak counter blurs the high/low gap and the bits flip. Full counts are in `counts.csv`. Training curves are in `runs/glyphs/results.png`.
