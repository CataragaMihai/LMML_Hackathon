# Task USA: The Great American Token Mystery

`main.py` finds the tokenizer whose encoding of the calibration text equals the given token IDs, then uses it to decode the challenge IDs.

## Run
```
pip install tiktoken transformers
python main.py train   # reads train-input.json, writes train-output.json
python main.py         # reads test-input.json, writes test-output.json
```
Run it from the folder that contains the input file. Training result: `p50k_base`.

## How it works
1. `candidates()` yields every tokenizer to try as `(name, encode, decode)`: all tiktoken encodings, then a list of Hugging Face models.
2. `main()` encodes the calibration text with each one and compares the full ID list to the given IDs.
3. On the first exact match it decodes the challenge IDs and writes `tokenizer_id` and `decoded_message`.

## The diagram

Along with the code I have attached the logic diagram. This is NOT a flowchart so the symbols are not to be treated like the ones in a flowchart. The purpose of the diagram is to simply visualeze the core logic, problem and solution implementation.