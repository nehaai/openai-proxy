import httpx
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tools = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Tokyo?"}]
prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
print("PROMPT:\n" + prompt)
r = httpx.post("http://localhost:8001/v1/completions",
               json={"prompt": prompt, "max_tokens": 100, "stop": ["<|im_end|>"]})
print("REPLY:", repr(r.json()["choices"][0]["text"]))