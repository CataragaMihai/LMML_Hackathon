"""Remove the steady narrow tone lines (vuvuzela harmonics) that survive unloop.py.

High-resolution STFT (4096 samples = 256 ms, 3.9 Hz bins), so a vuvuzela harmonic sits in one or two bins
while a voice harmonic moves around. Per bin, the steady part = running median over 1.5 s;
subtract it (spectral subtraction with a floor).

    python delines.py voice_only.wav voice_delined.wav
"""
import sys

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg
from scipy.ndimage import median_filter

sr, x = wavfile.read(sys.argv[1])
x = x.astype(np.float64) / 32768
N, HOP = 4096, 512
f, t, X = sg.stft(x, sr, nperseg=N, noverlap=N - HOP)
mag = np.abs(X)
steady = median_filter(mag, size=(1, int(1.5 * sr / HOP)), mode="nearest")
gain = np.maximum(1 - 1.2 * steady / (mag + 1e-12), 0.05)
gain[f > 4500] = 0.05
_, y = sg.istft(X * gain, sr, nperseg=N, noverlap=N - HOP)
y = y[: len(x)]
y = 0.9 * y / np.abs(y).max()
wavfile.write(sys.argv[2], sr, (y * 32767).astype(np.int16))
print("wrote", sys.argv[2])
