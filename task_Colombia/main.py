import json, hashlib, unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

d = json.load(open('reviews.json'))
T, S = d['trusted_reference'], d['samples']
text = lambda r: r['summary'] + ' . ' + r['review_text']

allowed = {c for r in T for c in text(r)}
uni = {s['sample_index'] for s in S if any(c not in allowed and ord(c) > 127 for c in text(s))}

vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
clf = LogisticRegression(C=10, max_iter=5000).fit(vec.fit_transform(map(text, T)), [t['rating'] >= 4 for t in T])
pred = clf.predict(vec.transform(map(text, S)))
rat = {s['sample_index'] for s, p in zip(S, pred) if (s['rating'] >= 4) != p}

altered = sorted(uni | rat)
print('unicode:', sorted(uni), '\nrating :', sorted(rat), '\naltered:', len(altered))

mask = hashlib.sha256(','.join(map(str, altered)).encode()).digest()
for r in T + S:
    msg = bytes(a ^ b for a, b in zip(bytes.fromhex(r['archive_tag']), mask)).rstrip(b'\0')
    if msg.startswith(b'SIGMOID_'):
        print('FLAG:', msg.decode())
