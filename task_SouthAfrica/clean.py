import sys

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg
from scipy.ndimage import percentile_filter

src, dst = sys.argv[1], sys.argv[2]
sr, x = wavfile.read(src)
x = x.astype(np.float64) / 32768

f, t, X = sg.stft(x, sr, nperseg=512, noverlap=384)        # 32 ms frames, 8 ms hop
mag = np.abs(X)
hop = (512 - 384) / sr
WIN, PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6, int(sys.argv[4]) if len(sys.argv) > 4 else 30
noise = percentile_filter(mag, PCT, size=(1, int(WIN / hop)), mode="nearest")  # low percentile over a short window per bin
gain = np.clip(1 - 1.5 * noise / (mag + 1e-12), 0, 1) ** 2   # keep what rises above the steady noise
gain[f > 4000] = 0
gain[f < 80] = 0
_, y = sg.istft(X * gain, sr, nperseg=512, noverlap=384)
y = y[: len(x)]
y = 0.9 * y / np.abs(y).max()
wavfile.write(dst, sr, (y * 32767).astype(np.int16))
print("wrote", dst)
