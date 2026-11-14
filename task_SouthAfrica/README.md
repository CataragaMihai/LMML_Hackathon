# South Africa: Ngiyakuzwa? 
Flag: `SIGMOID_ULWAZI` (*ulwazi* = "knowledge"). I used hint 1.

The voice note is 10 minutes long: a woman reads isiZulu phrases under stadium noise. And I needed the word she says right after *ikhodi* ("the code").

## What the noise is (the hint: look before you listen)

The spectrogram (`python` + `scipy.signal.spectrogram`, see the scripts) shows three layers:

1. **Regular part:** two vuvuzelas. These are sharp harmonic lines at fixed frequencies that never move for the whole file. This is what Hint 1 points at.
2. **Crowd:** broadband noise and whistles (5-7 kHz). It is built from clips that repeat: any second of crowd noise appears elsewhere in the file almost sample for sample.
3. **The voice:** below the noise.

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `notch.py voice_note.wav notched.wav` | measures every sharp line on the full-file spectrum (47 lines) and cuts each one with a ~6 Hz notch filter (zero-phase) |
| 2 | `unloop.py voice_note.wav voice_only.wav` | for each 1 s block, finds the 5 most similar places in the file, aligns them to the sample, takes the median (the voice differs in each copy, so only the noise survives) and subtracts it |
| 3 | `notch.py voice_only.wav voice_notched.wav voice_note.wav` | notches after unlooping, measuring the lines on the original |
| 4 | `asr.py <file>.wav > transcript.txt` | Meta's MMS speech model (`facebook/mms-1b-all`, isiZulu adapter `zul`), 15 s chunks with timestamps |
| 5 | `kws.py <file>.wav ikhodi` | keyword spotting: CTC dynamic program that scores how well "ikhodi" fits every time window |

Other helpers: `clean.py` (spectral mask), `delines.py` (median-based line removal). These were tried on the way.

## Where the word is

At **7:17-7:21**, under one of the loudest crowd bursts. Every cleaned version transcribes the phrase as "...njengoba usebenzisa **i-[khodi] ___** isicelo...", where *usebenzisa ikhodi* means "you use the code".
The next word comes out as `ulwaso`/`ilwaso` (notched raw), `udwazo`/`ungwazi` (unlooped) and `ungozi` (median-delined).
Scored with CTC against candidates, the notched raw audio ranked `ulwazi` first. Listening to `clip_0716_notched.wav` (7 s from 7:16) confirms it.

## What did not work

- A spectral mask alone (`clean.py`) and the MMS transcript: "ikhodi" was never transcribed.
- Guesses `UNGWAZI`, `INGWAZI`, `NGWAZI` (from the unlooped audio): wrong.
- So the notch filter (removing the regular part, the hint) gave the cleanest version of that word. Subtracting the crowd copies smeared it.

## Run

```
pip install numpy scipy torch transformers
python notch.py voice_note.wav notched.wav
python asr.py notched.wav > transcript_notched.txt
ffmpeg -ss 436 -t 7 -i notched.wav clip_0716_notched.wav
```
