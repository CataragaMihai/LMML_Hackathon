import sys

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg

sr, x = wavfile.read(sys.argv[1])
x = x.astype(np.float64) / 32768

ref = x if len(sys.argv) < 4 else wavfile.read(sys.argv[3])[1].astype(np.float64)  # measure lines here
f, P = sg.welch(ref, sr, nperseg=1 << 16)                    # 0.24 Hz resolution
peaks, _ = sg.find_peaks(10 * np.log10(P), prominence=10)
lines = [f[p] for p in peaks if 60 < f[p] < 7900]
print(f"{len(lines)} lines, first ones: {np.round(lines[:12], 1)}")

y = x.copy()
for fl in lines:
    b, a = sg.iirnotch(fl, Q=fl / 6.0, fs=sr)
    y = sg.filtfilt(b, a, y)
y = 0.9 * y / np.abs(y).max()
wavfile.write(sys.argv[2], sr, (y * 32767).astype(np.int16))
print("wrote", sys.argv[2])
