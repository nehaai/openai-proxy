from pydantic import BaseModel
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

class City(BaseModel):
    city: str
    population_millions: float

msgs = [{"role": "user", "content": "Describe Tokyo."}]
r = client.chat.completions.parse(model="qwen", messages=msgs, response_format=City)
print("parsed:", r.choices[0].message.parsed)
r = client.chat.completions.create(model="qwen", messages=msgs, max_tokens=8,
    response_format={"type": "json_object"})
print("truncated:", r.choices[0].finish_reason, repr(r.choices[0].message.content))