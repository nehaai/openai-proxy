from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
p = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city", "parameters": p}}]
for q in ["Weather in Tokyo?", "Tell me a joke."]:
    for ch in client.chat.completions.create(model="qwen", stream=True, tools=tools,
                                             messages=[{"role": "user", "content": q}]):
        d = ch.choices[0].delta if ch.choices else None
        if d and d.content: print(d.content, end="", flush=True)
        if d and d.tool_calls: print(d.tool_calls[0].model_dump(exclude_none=True))
        if ch.choices and ch.choices[0].finish_reason: print("\nfinish:", ch.choices[0].finish_reason)
with client.chat.completions.stream(model="qwen", tools=tools,
                                    messages=[{"role": "user", "content": "Weather in Tokyo?"}]) as s:
    print("assembled:", s.get_final_completion().choices[0].message.tool_calls)