import sys

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg

sr, x = wavfile.read(sys.argv[1])
x = x.astype(np.float64) / 32768
B, H, K, D = sr, sr // 2, 5, 4                       

xd = sg.decimate(x, D)
nfft = 1 << int(np.ceil(np.log2(len(xd) + B // D)))
Xd = np.fft.rfft(xd, nfft)
energy_d = np.convolve(xd ** 2, np.ones(B // D), "valid")


def best_offsets(start):
    """K best-matching block starts (full-rate samples), excluding the block itself."""
    q = xd[start // D: start // D + B // D]
    c = np.fft.irfft(Xd * np.conj(np.fft.rfft(q, nfft)), nfft)[: len(energy_d)]
    r = c / (np.sqrt(energy_d) * np.linalg.norm(q) + 1e-12)
    found = []
    for p in np.argsort(-r)[:400]:
        p_full = p * D
        if abs(p_full - start) > B and all(abs(p_full - f) > B // 2 for f in found):
            found.append(p_full)
        if len(found) == K:
            break
    refined = []
    q_full = x[start: start + B]
    for p in found:                                
        lags = range(max(0, p - 2 * D), min(len(x) - B, p + 2 * D) + 1)
        refined.append(max(lags, key=lambda l: np.dot(x[l: l + B], q_full)))
    return refined


win = np.hanning(B)
out = np.zeros(len(x))
norm = np.zeros(len(x))
for start in range(0, len(x) - B + 1, H):
    seg = x[start: start + B]
    copies = []
    for p in best_offsets(start):
        c = x[p: p + B]
        copies.append(c * np.dot(seg, c) / (np.dot(c, c) + 1e-12))   # least-squares gain
    noise = np.median(copies, axis=0)
    out[start: start + B] += (seg - noise) * win
    norm[start: start + B] += win
    if start % (60 * sr) == 0:
        print(f"{start // sr} s", flush=True)
y = out / np.maximum(norm, 1e-3)
y = 0.9 * y / np.abs(y).max()
wavfile.write(sys.argv[2], sr, (y * 32767).astype(np.int16))
print("wrote", sys.argv[2])
