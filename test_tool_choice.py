from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
p = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city", "parameters": p}},
         {"type": "function", "function": {"name": "get_time", "description": "Local time in a city", "parameters": p}}]
msgs = [{"role": "user", "content": "Tell me a joke."}]
for choice in ["auto", "required", {"type": "function", "function": {"name": "get_time"}}]:
    m = client.chat.completions.create(model="qwen", messages=msgs, tools=tools, tool_choice=choice).choices[0].message
    print(choice if isinstance(choice, str) else "named", "→", m.tool_calls[0].function if m.tool_calls else m.content)