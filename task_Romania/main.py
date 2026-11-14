"""Task Romania: Shaken, Still Standing.

The records are not one population. There are three, each with its own rule, and sealed.csv mixes them
differently from train.csv (quiet group: 25% of train, 40% of sealed). So one model fitted to the whole
of train.csv cannot do well on sealed.csv. We split the records with two gates and fit one expert per group:

  gate 1  quiet?  For about a third of the records, a block of 36 channels barely moves (std / 4).
                  Found with a 2-component Gaussian mixture; it splits cleanly.
  gate 2  c13 > -0.6?  Only inside the loud records. Invisible in any histogram; found by adversarial
                  validation (a model telling train from sealed records uses c13 alone).

  group  B  (quiet)            label = quadratic function of the 12 "plain" channels -> SVM (rbf)
  group  P1 (loud, c13 <= -0.6) label = XOR-like interaction of c35, c43, c38, ...   -> gradient boosting
  group  P2 (loud, c13 >  -0.6) label = linear function of all channels               -> logistic regression

    python main.py      # CV report per group, writes predictions.csv, runs the evaluator
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
EVALUATOR = HERE / "evaluation_script.py"
C = [f"c{j + 1:02d}" for j in range(48)]
C13 = C.index("c13")

train = pd.read_csv(DATA / "train.csv")
sealed = pd.read_csv(DATA / "sealed.csv")
X_all = pd.concat([train[C], sealed[C]]).to_numpy()
n = len(train)
y = train.vrancea.to_numpy()

# Gate 1: quiet records (the mixture component with the small variances) ----------------------------
gmm = GaussianMixture(2, covariance_type="diag", random_state=0, n_init=3).fit(X_all)
quiet = np.argmin(gmm.covariances_.mean(1))
is_quiet = gmm.predict(X_all) == quiet
std_ratio = np.sqrt(gmm.covariances_[1 - quiet] / gmm.covariances_[quiet])
PLAIN = [j for j in range(48) if std_ratio[j] < 1.5]  # the 12 channels that do not go quiet

# Gate 2: inside the loud records, split on c13 -------------------------------------------------------
group = np.where(is_quiet, "B", np.where(X_all[:, C13] <= -0.6, "P1", "P2"))
for name, rows in (("train", slice(0, n)), ("sealed", slice(n, None))):
    print(f"{name:6s} group counts:", pd.Series(group[rows]).value_counts().sort_index().to_dict())
print("plain channels:", [C[j] for j in PLAIN])

# One expert per group --------------------------------------------------------------------------------
EXPERTS = {
    "B": (lambda: make_pipeline(StandardScaler(), SVC(C=3.0)), PLAIN),
    "P1": (lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, random_state=0), list(range(48))),
    "P2": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000)), list(range(48))),
}


def fit_predict(Xtr, ytr, gtr, Xte, gte):
    pred = np.zeros(len(Xte), dtype=int)
    for g, (make, cols) in EXPERTS.items():
        if (gte == g).any():
            model = make().fit(Xtr[gtr == g][:, cols], ytr[gtr == g])
            pred[gte == g] = model.predict(Xte[gte == g][:, cols])
    return pred


# Cross-validation on train.csv ------------------------------------------------------------------------
Xt, gt = X_all[:n], group[:n]
oof = np.zeros(n, dtype=int)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xt, y):
    oof[te] = fit_predict(Xt[tr], y[tr], gt[tr], Xt[te], gt[te])
acc = {g: np.mean(oof[gt == g] == y[gt == g]) for g in EXPERTS}
print("CV accuracy per group:", {g: round(a, 4) for g, a in acc.items()}, "| overall", round(np.mean(oof == y), 4))
gs = group[n:]
print("expected accuracy on sealed (its own group mix):", round(sum(acc[g] * np.mean(gs == g) for g in EXPERTS), 4))

# Sealed predictions + evaluator ----------------------------------------------------------------------
pred = fit_predict(Xt, y, gt, X_all[n:], gs)
pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred}).to_csv(HERE / "predictions.csv", index=False)
print("wrote predictions.csv")
sys.stdout.flush()
if EVALUATOR.exists():
    subprocess.run([sys.executable, str(EVALUATOR), str(HERE / "predictions.csv")], check=False)
