from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(model="qwen", stream=True,
    stream_options={"include_usage": True},
    messages=[{"role": "user", "content": "Write a haiku about the ocean."}])
for ch in stream:
    if ch.choices and ch.choices[0].delta.content:
        print(ch.choices[0].delta.content, end="", flush=True)
    if ch.usage: print("\n", ch.usage)