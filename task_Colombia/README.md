# Colombia: Looks Good to Me!

**Flag:** `SIGMOID_DATA_SANITIZATION`

**Run it:** put `reviews.json` in this folder, then `python main.py` (needs numpy and scikit-learn).

## The idea

Some suspect reviews were secretly changed. The archive note says the numbers of the changed reviews unlock the message. So: find the changed reviews, then use their numbers to decode.

## Step 1: reviews with strange characters (12)

The trusted reviews only use normal letters plus two special symbols, `’` and `–`. Any suspect review with another special character was tampered with. There are 12: six with an invisible zero-width space and six with a Cyrillic letter that looks like a Latin one (for example Cyrillic `а` instead of `a`).

## Step 2: reviews with a changed rating (12)

Some ratings were moved by one star, for example 3 to 4. This crosses the line between "bad" (1-3) and "good" (4-5), so the text and rating disagree.

1. Learn from the trusted reviews: train a simple model that reads the text and says "good" (4-5) or "bad" (1-3).
2. Run it on the suspect reviews.
3. Every review where the model disagrees with its rating is flagged. There are exactly 12.

## Step 3: decode the message

1. Combine the two groups: 24 review numbers.
2. Sort them, join them with commas (`36,112,119,...`), and take the SHA-256 of that text. This gives a 32-byte key.
3. XOR every `archive_tag` with the key and drop the trailing zero bytes.
4. Only one tag turns into readable text. That is the flag.

## Useful details

- You need both groups. Either one alone gives garbage instead of the flag.
- A model that predicts all 5 star ratings misses the flips, because the rating is only one star off. Splitting into just "good" and "bad" makes them clear.
