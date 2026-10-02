from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return "22C, clear"

llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="unused", model="qwen").bind_tools([get_weather])
print("invoke:", llm.invoke("What's the weather in Tokyo?").content)
print("stream:", [c.tool_call_chunks for c in llm.stream("What's the weather in Tokyo?") if c.tool_call_chunks])