import sys

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL = "facebook/mms-1b-all"
proc = AutoProcessor.from_pretrained(MODEL, target_lang="zul")
model = Wav2Vec2ForCTC.from_pretrained(MODEL, target_lang="zul", ignore_mismatched_sizes=True).eval()
sr, x = wavfile.read(sys.argv[1])
x = x.astype(np.float32) / 32768
word = sys.argv[2]

# 1. per-frame log-probs for the whole file (20 s chunks, 2 s context each side) --------------------
FPS = 50                                                      # wav2vec2: one frame per 20 ms
logp = []
for s in range(0, len(x), 20 * sr):
    lo = max(0, s - 2 * sr)
    chunk = x[lo: s + 22 * sr]
    with torch.no_grad():
        lp = model(**proc(chunk, sampling_rate=sr, return_tensors="pt")).logits[0].log_softmax(-1).numpy()
    a = (s - lo) * FPS // sr
    logp.append(lp[a: a + 20 * FPS])
logp = np.concatenate(logp)
np.save("logp.npy", logp)

# 2. CTC keyword DP: states = blank, k1, blank, k2, ... ; free start anywhere -------------------------
vocab = proc.tokenizer.get_vocab()
blank = proc.tokenizer.pad_token_id
ids = [vocab[c] for c in word]
states = [blank]
for i in ids:
    states += [i, blank]
S, T = len(states), len(logp)
NEG = -1e9
D = np.full(S, NEG)
best_end = np.full(T, NEG)
start = np.zeros(S, dtype=int)
starts = np.zeros(T, dtype=int)
for t in range(T):
    e = logp[t, states]
    N = np.full(S, NEG)
    ns = np.zeros(S, dtype=int)
    for k in range(S):
        cands = [(D[k], start[k])]
        if k >= 1:
            cands.append((D[k - 1], start[k - 1]))
        if k >= 2 and states[k] != blank and states[k] != states[k - 2]:
            cands.append((D[k - 2], start[k - 2]))
        if k <= 1:
            cands.append((0.0, t))                            # the keyword may start here
        v, st = max(cands)
        N[k], ns[k] = v + e[k], st
    D, start = N, ns
    k = int(np.argmax(D[-2:])) + S - 2                        # ended on last letter or trailing blank
    best_end[t], starts[t] = D[k], start[k]

free = np.concatenate([[0], np.cumsum(logp.max(1))])
score = np.array([best_end[t] - (free[t + 1] - free[starts[t]]) for t in range(T)])

# 3. report the best non-overlapping hits ------------------------------------------------------------
def greedy(a, b):
    return proc.decode(logp[max(0, a):b].argmax(-1))

order = np.argsort(-score)
used = []
for t in order:
    if any(abs(t - u) < 2 * FPS for u in used):
        continue
    used.append(t)
    s0 = starts[t]
    print(f"{s0 / FPS:7.2f}s-{t / FPS:7.2f}s score {score[t]:7.2f} | ...{greedy(s0 - 2 * FPS, s0)} [{greedy(s0, t + 1)}] {greedy(t + 1, t + 2 * FPS)}...")
    if len(used) == 10:
        break
