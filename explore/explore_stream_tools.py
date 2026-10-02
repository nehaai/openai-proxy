import httpx, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
p = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city", "parameters": p}}]
prompt = tok.apply_chat_template([{"role": "user", "content": "Weather in Tokyo?"}], tools=tools, tokenize=False, add_generation_prompt=True)
with httpx.stream("POST", "http://localhost:8001/v1/completions",
                  json={"prompt": prompt, "max_tokens": 100, "stop": ["<|im_end|>"], "stream": True}) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            print(repr(json.loads(line[6:])["choices"][0]["text"]))