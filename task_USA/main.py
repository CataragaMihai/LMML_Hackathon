import json #handle json files
import sys #system operations

import tiktoken  #token library?
from huggingface_hub import list_models #search the Hugging Face Hub
from transformers import AutoTokenizer as at 

#add an array of huggin face tokenizers

HF_MODELS = [
    "gpt2", "EleutherAI/gpt-neox-20b", "facebook/opt-125m", "bigscience/bloom-560m",
    "Salesforce/codegen-350M-mono", "microsoft/phi-2", "bert-base-uncased",
    "bert-base-cased", "roberta-base", "xlm-roberta-base", "t5-small",
    "albert-base-v2", "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistralai/Mistral-7B-v0.1", "microsoft/deberta-v3-base",
]

# Instead of only a fixed list, also ask the Hugging Face Hub for the most downloaded models.
def hf_names(per_tag=60):
    seen = set(HF_MODELS)
    yield from HF_MODELS  # known tokenizers first: they are the most likely
    try:
        for tag in ("text-generation", "fill-mask"):
            for m in list_models(pipeline_tag=tag, sort="downloads", limit=per_tag):
                if m.id not in seen:
                    seen.add(m.id)
                    yield m.id
    except Exception as eror:  # offline: we still have the fixed list above
        print(f"hub search failed: {type(eror).__name__}", file=sys.stderr)

# create a function that will run through the canditates

def candidates():
    for name in tiktoken.list_encoding_names():
        enc = tiktoken.get_encoding(name)
        yield name, lambda t, e = enc: e.encode(t, disallowed_special =()), lambda ids, e = enc: e.decode(ids)
    for name in hf_names():
        try:
            tok = at.from_pretrained(name)
        except Exception as eror:
            print(f"skip {name}: {type(eror).__name__}", file=sys.stderr)
            continue 

        for special in (False, True):
            yield (name,
                   lambda t, k=tok, s=special: k.encode(t, add_special_tokens=s),
                   lambda ids, k=tok: k.decode(ids, clean_up_tokenization_spaces=False))


def main(prefix):
    data = json.load(open(f"{prefix}-input.json"))
    text = data["calibration"]["text"]
    target = data["calibration"]["token_ids"]
    challenge = data["challenge"]["token_ids"]

    for name, encode, decode in candidates():
        try:
            if encode(text) == target:
                result = {"tokenizer_id": name, "decoded_message": decode(challenge)}
                json.dump(result, open(f"{prefix}-output.json", "w"), ensure_ascii=False, indent=2)
                print("found a match:", name)
                return 
        except Exception as eror:
            print(f"fail {name}: {type(eror).__name__}", file=sys.stderr)
    sys.exit("no tokenizer matched: extend the candidate lists")
    

if __name__ == "__main__":
        main(sys.argv[1] if len(sys.argv) > 1 else "test")

