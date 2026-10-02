# openai-proxy

An OpenAI-compatible `/v1/chat/completions` server built from scratch, in front of a local llama.cpp backend.

**The one rule:** the proxy only calls the backend's raw `/v1/completions` endpoint, which takes plain text in and returns plain text out. It never uses the backend's own chat endpoint. Everything chat-shaped is implemented here: prompt templating, the streaming format, tool-call parsing, `tool_choice` enforcement, and structured outputs.

The point was to find out exactly what the OpenAI API contract promises versus what a language model actually does underneath. The findings are in [`CONTRACT_VS_REALITY.md`](CONTRACT_VS_REALITY.md).

## Architecture

```
OpenAI SDK / LangChain  →  proxy (FastAPI, :8000)  →  llama-server (/v1/completions, :8001)
                           ├─ chat template (HF tokenizer)
                           ├─ SSE re-chunking
                           ├─ tool-call parser + stream state machine
                           └─ JSON-schema constrained decoding
```

- **Model:** Qwen2.5-1.5B-Instruct (GGUF, Q4_K_M, about 1 GB)
- **Backend:** llama.cpp's `llama-server`, running on Apple Silicon (Metal)
- **Proxy:** Python, FastAPI, httpx, and a Hugging Face tokenizer (no PyTorch needed)

## Run it

```bash
# 1. Backend
brew install llama.cpp
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8001

# 2. Proxy (new terminal)
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn httpx transformers pydantic openai pytest langchain-openai
uvicorn main:app --port 8000

# 3. Conformance tests (new terminal)
pytest -v test_conformance.py
```

Point any OpenAI client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
```

## What's implemented

| Feature | Details |
|---|---|
| Chat | Multi-turn conversations, all roles (`developer` is mapped to `system`), `stop`, `temperature`, `top_p`, `max_tokens` / `max_completion_tokens` |
| Errors | OpenAI error envelope; 400 for invalid input (no SDK retry), 503/502 for backend failures (SDK retries) |
| Streaming | `chat.completion.chunk` SSE, a role-first chunk, `stream_options.include_usage`, `[DONE]`; client disconnects cancel generation on the backend |
| Tool calling | `tools`, tool-result round-trips, `tool_choice` set to `auto`, `none`, `required`, or a named function (forced through constrained decoding plus prefill) |
| Streaming tool calls | A state machine detects `<tool_call>`, holds back possible partial tags, and sends the name delta followed by the arguments delta |
| Structured outputs | `response_format` set to `json_object` or `json_schema`, enforced by the backend's grammar-constrained sampling; works with the SDK's `parse()` |
| Compatibility | Verified with the official OpenAI Python SDK and LangChain (`ChatOpenAI.bind_tools`, both invoke and stream) |

## Tests

`test_conformance.py` checks the contract through the official SDK. Tests use `temperature=0` and `tool_choice="required"` so the results don't depend on sampling randomness.

- Basic chat: response shape, `finish_reason`, usage
- Invalid parameters raise `BadRequestError`
- Stream contract: one `id` across chunks, role first, usage chunk last
- Forced tool call: `arguments` is a JSON string
- Streamed tool call: the SDK's stream helper reassembles a complete call
- JSON schema: `parse()` returns a typed Pydantic object

## Known gaps (deliberate)

- `n > 1` is rejected.
- `logprobs` is not supported.
- Forced `tool_choice` produces exactly one call; parallel tool calls aren't supported in forced mode.
- `strict: true` schemas are passed through without OpenAI's up-front validation.
- `tools` combined with `response_format` is unresolved, because constrained JSON leaves no room for `<tool_call>` tags.
- Streamed tool-call arguments arrive in one delta, not incrementally.
- Malformed or truncated tool calls are dropped silently.
- Streamed `usage` is computed by re-tokenizing, so it can differ slightly from non-streamed counts.
- Unknown parameters are accepted and logged, not rejected the way OpenAI rejects them.
- `content` must be a string; multimodal content arrays aren't supported.
- Errors that happen after a stream has started aren't reported inside the stream.
- There's a single hardcoded model, and the `model` field is echoed back rather than routed.
- There's no authentication.

## Findings

The full log is in [`CONTRACT_VS_REALITY.md`](CONTRACT_VS_REALITY.md). The three most surprising:

1. **The chat template secretly injects a system prompt.** With no system message, Qwen's template adds "You are Qwen, created by Alibaba Cloud…". A 5-token question cost 33 prompt tokens, and the client never sent that text.
2. **Adding tools changes behavior even when no tool is called.** With tools defined and `tool_choice: "auto"`, the model refused to tell a joke. Tools aren't optional extras; they're system-prompt text that reshapes every response.
3. **Constrained decoding is the only hard guarantee in the stack, and even it breaks.** Roles, tool calls, and stop behavior are all formatting conventions the model was trained on. Grammar-constrained sampling truly prevents invalid JSON, but if `max_tokens` cuts the output off, the "guaranteed" JSON is invalid anyway.

## Layout

```
main.py                 # the proxy
models.py               # request models (Pydantic)
test_conformance.py     # automated contract tests
test_*.py               # manual exploration scripts, one per step
explore/                # raw-backend experiments behind each finding
CONTRACT_VS_REALITY.md  # promise vs. reality log
```