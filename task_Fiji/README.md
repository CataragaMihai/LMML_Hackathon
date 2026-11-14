# Fiji: Walu Season

10 fishing spots, 1,000 casts a day, unknown noisy catches. Maximise the total catch (target: 5,300 kg/day average).
This is a multi-armed bandit: explore to find the walu spot, then exploit it.

## Approach: Gaussian Thompson sampling (`main.py`)
- Per spot, keep the number of casts, the sum of catches and the sum of squared catches.
- Each spot gets 5 forced casts first. With fewer, a walu spot that misses early looks bad and is dropped for good.
- Then every cast: draw a random guess of each spot's average (mean, with noise that shrinks as the spot gets more casts) and fish the spot with the highest guess.
- A variance floor of 4.0 stops the noise collapsing when early catches look similar.
- Randomness is seeded (`random.Random(0)`).

## Result
Practice reef: **6,738 kg/day** (threshold 5,300), about 9 s for the full run.
A first version (2 forced casts, variance floor 0.25) scored 4,942 and failed. See `notes.md`.

## Reproduce
```
cd task_Fiji/materials
python evaluation_script.py ../main.py --dataset train
```
Python 3.12, standard library only.

## Files
- `main.py`: the submission (only the `Strategy` class differs from the starter).
- `materials/`: organisers' simulator, starter and practice data.
- `notes.md`: working notes and risks. `overview.md`: task text.
