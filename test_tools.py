import json
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Tokyo?"}]
r = client.chat.completions.create(model="qwen", messages=msgs, tools=tools)
call = r.choices[0].message.tool_calls[0]
print("1:", r.choices[0].finish_reason, call.function.name, call.function.arguments)
msgs += [r.choices[0].message.model_dump(exclude_none=True),
         {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 22, "sky": "clear"})}]
r = client.chat.completions.create(model="qwen", messages=msgs, tools=tools)
print("2:", r.choices[0].message.content)