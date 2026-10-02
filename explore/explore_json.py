import httpx
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
msgs = [{"role": "user", "content": "Describe Tokyo."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
schema = {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
          "required": ["city", "population_millions"], "additionalProperties": False}
for extra in ({}, {"json_schema": schema}):
    r = httpx.post("http://localhost:8001/v1/completions",
                   json={"prompt": prompt, "max_tokens": 80, "stop": ["<|im_end|>"], **extra})
    print("constrained" if extra else "free", "→", repr(r.json()["choices"][0]["text"]))