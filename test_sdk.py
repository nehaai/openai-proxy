from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
r = client.chat.completions.create(model="qwen", messages=[
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And Germany's?"}])
print(r.choices[0].message.content, "|", r.choices[0].finish_reason, "|", r.usage)