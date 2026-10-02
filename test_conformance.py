import json, pytest
from pydantic import BaseModel
from openai import OpenAI, BadRequestError
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
P = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city", "parameters": P}}]
def ask(q): return [{"role": "user", "content": q}]

def test_basic_chat():
    r = client.chat.completions.create(model="qwen", messages=ask("Capital of France?"), temperature=0)
    assert r.object == "chat.completion" and r.choices[0].finish_reason == "stop"
    assert "Paris" in r.choices[0].message.content and r.usage.total_tokens > 0

def test_bad_param_is_400():
    with pytest.raises(BadRequestError):
        client.chat.completions.create(model="qwen", messages=ask("hi"), temperature=5)

def test_stream_contract():
    chunks = list(client.chat.completions.create(model="qwen", messages=ask("Say hi."), temperature=0,
                                                 stream=True, stream_options={"include_usage": True}))
    assert len({c.id for c in chunks}) == 1               # one id across all chunks
    assert chunks[0].choices[0].delta.role == "assistant"  # role comes first
    assert chunks[-1].choices == [] and chunks[-1].usage   # usage chunk comes last

def test_forced_tool_call():
    r = client.chat.completions.create(model="qwen", messages=ask("Tell me a joke."), tools=TOOLS, tool_choice="required")
    call = r.choices[0].message.tool_calls[0]
    assert r.choices[0].finish_reason == "tool_calls" and call.function.name == "get_weather"
    assert "city" in json.loads(call.function.arguments)  # arguments must be a JSON string

def test_streamed_tool_call_assembles():
    with client.chat.completions.stream(model="qwen", messages=ask("Weather in Tokyo?"),
                                        tools=TOOLS, tool_choice="required") as s:
        assert s.get_final_completion().choices[0].message.tool_calls[0].function.name == "get_weather" 

class City(BaseModel):
    city: str
    population_millions: float

def test_json_schema_parses():
    r = client.chat.completions.parse(model="qwen", messages=ask("Describe Tokyo."), response_format=City, temperature=0)
    assert isinstance(r.choices[0].message.parsed, City)