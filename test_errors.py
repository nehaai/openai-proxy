import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
try:
    client.chat.completions.create(model="qwen", temperature=5,
        messages=[{"role": "user", "content": "hi"}])
except openai.BadRequestError as e:
    print("OK:", e.status_code, e.body)