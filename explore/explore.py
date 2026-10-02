import httpx
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
msgs = [{"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Capital of France?"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("PROMPT:\n" + prompt)

r = httpx.post("http://localhost:8001/v1/completions",
               json={"prompt": prompt, "max_tokens": 50, "stop": ["<|im_end|>"]})
print("REPLY:", r.json()["choices"][0]["text"])