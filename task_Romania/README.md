# Romania: Shaken, Still Standing (Hard, 300 pts)

Flag: `SIGMOID_CONVOLUTION`

## Run

```
python main.py
```

Needs `dataset/train.csv`, `dataset/sealed.csv`, `dataset/vault.npz` and, to print the flag, `evaluation_script.py` next to `main.py`.
It prints the group counts, the CV accuracy per group, writes `predictions.csv` and runs the evaluator.

## The idea

The records are **not one population**. There are three groups, each with its own rule,
and `sealed.csv` mixes them differently from `train.csv`:

| group | how to find it | rule (what fits it) | train | sealed |
|---|---|---|---|---|
| B: quiet | 36 of the 48 channels barely move (std / 4). 2-component Gaussian mixture, splits cleanly | quadratic in the 12 "plain" channels: SVM (rbf) | 1900 | 2000 |
| P1: loud, `c13 <= -0.6` | `c13`: invisible in histograms, found by adversarial validation | XOR-like interaction of c35, c43, c38...: gradient boosting | 3689 | 597 |
| P2: loud, `c13 > -0.6` | same | linear: logistic regression | 1911 | 2403 |

Plain channels (the ones that never go quiet): c03 c09 c12 c14 c15 c19 c27 c28 c30 c36 c39 c40.

A single model fitted to all of train learns mostly P1's rule (P1 is half of train), but sealed is mostly B and P2. That is why
the baseline fails even though its CV looks fine.

The evaluator needs about 92.5% of the sealed records right: a Reed-Solomon code built from the labels in `vault.npz`
decodes only with at least (n + k) / 2 = 4626 of 5000 correct.

## How I got there (5-fold CV on train.csv)

1. Single models on all records: logistic regression 0.60, SVM rbf 0.68, kNN 0.53, MLP 0.66, random forest 0.76,
   extra trees 0.69, gradient boosting (HGB) 0.85. Linear and kernel models fail, so the rule is interaction-heavy.
2. Per-channel kurtosis about 1.2 everywhere. A diagonal Gaussian mixture with 2 components finds the quiet group (31% of all records).
   Per-group models: HGB on the loud records 0.91. On the quiet records, plain channels only: HGB 0.84, logistic regression on x and x² 0.87, QDA 0.90, SVM rbf 0.915.
   With two experts: 0.90 overall, no flag.
3. I used hint 1: the loud records are not one population either, and train and sealed are mixed differently.
   Adversarial validation (HGB telling train records from sealed records, loud group only) gives AUC 0.73, almost all from c13.
   The c13 histogram of sealed jumps near -0.7; train's is smooth.
4. On the loud records, XOR(c35 > -0.15, c43 > -0.6) explains the label for low c13 (78%) and is at chance for high c13.
   Expert accuracy by c13 range: HGB 0.97 below -0.6 and 0.51 above. Logistic regression 0.51 below and 0.92 above. So the cut is c13 = -0.6.
5. Final experts: B SVM rbf 0.916, P1 HGB 0.968, P2 logistic regression (C=10) 0.927.
   CV overall 0.944. Weighted by the sealed mix: 0.9275, which passes the evaluator.

## How they are combined

A gated mixture of experts with hard gates. Gate 1 (quiet or loud) comes from the Gaussian mixture. Gate 2 is `c13 > -0.6`.
Each record goes to its group's expert. Each expert is trained only on its own group of train records.
