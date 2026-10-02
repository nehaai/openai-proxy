# Contract vs. Reality

What the OpenAI `/v1/chat/completions` API promises, what actually happens underneath (Qwen2.5-1.5B on llama-server, raw `/v1/completions` only), and what this proxy does about it.

## Backend

| Promise | Reality | Decision |
|---|---|---|
| The model answers questions | A raw completion continues text as a document: "The capital of France is" became a multiple-choice quiz | All chat behavior comes from the template |
| `finish_reason` says whether output is complete | The backend reports `stop` or `length`, and clients depend on it | Pass it through faithfully |
| A defined response schema | llama-server adds `timings`, cache stats, and a `chatcmpl-` ID on a *text* completion | Expose spec fields only |

## Chat

| Promise | Reality | Decision |
|---|---|---|
| Messages with roles | Roles are a text format the model was trained on (`<\|im_start\|>user … <\|im_end\|>`) | Render with the HF chat template |
| The assistant ends its turn | Only a stop token ends it | Always append `<\|im_end\|>` to the user's `stop` list |
| You pay for what you send | With no system message, the template injects "You are Qwen, created by Alibaba Cloud…": 33 prompt tokens for a 5-token question | Not suppressed; documented |
| Conversations | The API is stateless; the full history is re-sent and re-billed on every turn | Inherent |
| `developer` role | Qwen's template doesn't know it | Map to `system` |
| `max_completion_tokens` replaces `max_tokens` | Clients send either one | Accept both; prefer the new one |
| A default output length | No small fixed default is published | 512 tokens |
| Sampling defaults | An unset `temperature` falls back to the backend's default (about 0.8 here, versus OpenAI's 1.0), which varies by backend | Not overridden (gap) |

## Errors

| Promise | Reality | Decision |
|---|---|---|
| A standard error shape | FastAPI defaults to 422 in its own format | 400 with `{"error": {message, type, param, code}}` |
| Status codes are informational | They drive client behavior: 400 means no retry, 5xx makes the SDK retry automatically | 400 for bad input, 503 when the backend is down, 502 for backend errors |
| Unknown parameters are rejected | Frameworks send extras | Accept and log them (`IGNORED PARAMS`) |
| `n` greater than 1 | One completion per request | Reject `n > 1` |

## Streaming

| Promise | Reality | Decision |
|---|---|---|
| Live tokens in a defined format | The backend's SSE must be reshaped into `chat.completion.chunk` | One `id`/`created` per stream, a role-first chunk, a finish chunk, then `[DONE]` |
| `include_usage` gives usage | The proxy computes it by re-tokenizing | Approximate; can differ slightly from non-streamed counts |
| Errors come as status codes | After the first byte, a 200 has already been sent | Mid-stream errors must be reported in-band; not implemented (gap) |
| *(cancellation is unspecified)* | Without propagation, the backend keeps generating for clients who left | A disconnect closes the upstream request; llama-server cancels the task (verified in its logs) |

## Tool calling

| Promise | Reality | Decision |
|---|---|---|
| Structured function calls | Tool schemas are pasted into the system prompt, and the model writes `<tool_call>{json}</tool_call>` as text | Parse the text into `tool_calls` |
| Tools are optional extras | Tools change behavior even when none is called: with `auto`, the model refused to tell a joke | Documented; inherent |
| Tools cost nothing until used | Tool definitions cost prompt tokens on every request | Inherent |
| `arguments` is a JSON string | The model and the template use an object | Convert in both directions |
| `tool_call_id` links results to calls | The IDs are invented by the proxy and ignored by Qwen's template, so matching relies on message order | Generate `call_…` IDs |
| Calls are always well-formed | The model can emit invalid JSON | Dropped silently (gap) |
| `tool_choice: "none"` | No native backend support | Hide the tools from the prompt |
| `required` and named choices guarantee a call | Prompting can't guarantee anything | Prefill `<tool_call>\n`, then constrain output to a `{name, arguments}` schema |
| The call makes sense | `required` with an unrelated prompt invents one: "Tell me a joke" became `get_weather("New York")` | Inherent: the spec guarantees a call, not a sensible one |
| Parallel tool calls | Forced mode produces exactly one call | Gap |
| The same request gets the same behavior | `auto` called the tool in 3 of 4 runs | Tests use `temperature=0` and `tool_choice="required"` |
| *(no prefill in the API)* | Prefill keeps constrained output close to what the model was trained on | A capability only the proxy can use |

## Streaming tool calls

| Promise | Reality | Decision |
|---|---|---|
| The name arrives first, then argument fragments | Token boundaries don't match JSON boundaries: the name was split (`get` / `_weather`), and one token (`"}}\n`) closed both `arguments` and the wrapper | Buffer each call, then send a name delta and a full-arguments delta |
| Arguments stream gradually | They arrive in one burst | A tradeoff the spec allows |
| Tags are detectable | `<tool_call>` is a single special token in Qwen, but in other models tags can be split | Hold back text that could be the start of a tag |
| Text streams live | Held-back text lags by a few characters | Accepted |
| Calls are always complete | A call cut off by `max_tokens` is incomplete | Dropped silently (gap) |

## Structured outputs

| Promise | Reality | Decision |
|---|---|---|
| Valid JSON | Grammar-constrained sampling blocks invalid tokens; it's the only hard guarantee in the stack | Map `response_format` to llama-server's `json_schema` |
| Guaranteed JSON | Truncation by `max_tokens` breaks it: `{"error_code": "99` | Clients must check `finish_reason` |
| Correct data | Constrained decoding guarantees the shape, not the facts | Inherent |
| `json_object` gives useful JSON | It guarantees syntax only, so the model invented its own keys | Use `json_schema` to control structure |
| `json_object` requires "JSON" in the messages | The backend doesn't need it | Not enforced |
| `strict: true` is validated up front | There's no validation step | Schemas passed through (gap) |
| `tools` and `response_format` work together | Schema-constrained output can't contain `<tool_call>` tags | Unresolved |
| Parameters are portable | Constrained-decoding parameters differ by backend (llama-server's `json_schema` vs. vLLM's own parameter) | Backend-specific mapping |

## Real clients

| Promise | Reality | Decision |
|---|---|---|
| SDK compatibility means ecosystem compatibility | LangChain's `bind_tools` worked on both invoke and stream, and sent no parameters outside what's implemented | Verified with `IGNORED PARAMS` logging |