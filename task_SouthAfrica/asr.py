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
step = 15 * sr
for start in range(0, len(x), step):
    chunk = x[max(0, start - sr): start + step + sr]         
    inp = proc(chunk, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model(**inp).logits.argmax(-1)[0]
    print(f"{start // sr // 60:02d}:{start // sr % 60:02d}  {proc.decode(ids)}", flush=True)
