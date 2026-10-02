import time, uuid, httpx, json, re
from fastapi import FastAPI
from transformers import AutoTokenizer
from models import ChatRequest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

BACKEND = "http://localhost:8001/v1/completions"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
app = FastAPI()
http = httpx.AsyncClient(timeout=120)
OPEN, CLOSE = "<tool_call>", "</tool_call>"


def to_template_msg(m):
    d = {"role": "system" if m.role == "developer" else m.role, "content": m.content or ""}
    if m.tool_calls:
        d["tool_calls"] = [{"type": "function", "function": {"name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)}} for tc in m.tool_calls]
    return d


def build_prompt(req):
    tools = None if req.tool_choice == "none" else req.tools
    return tok.apply_chat_template([to_template_msg(m) for m in req.messages],
                                   tools=tools, tokenize=False, add_generation_prompt=True)


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def parse_tool_calls(text):
    calls = []
    for raw in TOOL_RE.findall(text):
        try: obj = json.loads(raw)
        except json.JSONDecodeError: continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))}})
    return (TOOL_RE.sub("", text).strip() or None), calls


def to_openai(req, out, forced=False):
    c = out["choices"][0]
    text = f"<tool_call>{c['text']}</tool_call>" if forced else c["text"]
    content, calls = parse_tool_calls(text) if req.tools else (c["text"], [])
    message = {"role": "assistant", "content": content}
    if calls: message["tool_calls"] = calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(time.time()), "model": req.model,
        "choices": [{"index": 0, "message": message, "logprobs": None,
                     "finish_reason": "tool_calls" if calls else c["finish_reason"]}],
        "usage": {k: out["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.model_extra: print("IGNORED PARAMS:", req.model_extra)
    stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    forced = forced_tool_schema(req)
    if forced is not None and not forced["anyOf"]:
        return oai_error(400, "tool_choice names a function not in tools", "invalid_request_error", "tool_choice")
    prompt = build_prompt(req) + ("<tool_call>\n" if forced else "")
    params = {"prompt": prompt, "stop": stops + ["<|im_end|>"],
              "max_tokens": req.max_completion_tokens or req.max_tokens or 512,
              "temperature": req.temperature, "top_p": req.top_p,
              "json_schema": forced or schema_for(req)}
    params = {k: v for k, v in params.items() if v is not None}
    if req.stream:
        return StreamingResponse(stream_chat(req, params, forced=bool(forced)), media_type="text/event-stream")
    try:
        r = await http.post(BACKEND, json=params)
    except httpx.ConnectError:
        return oai_error(503, "Model backend unavailable", "server_error")
    if r.status_code != 200:
        return oai_error(502, f"Backend error: {r.text[:200]}", "server_error")
    return to_openai(req, r.json(), forced=bool(forced))


def oai_error(status, message, type_, param=None, code=None):
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": type_, "param": param, "code": code}})


@app.exception_handler(RequestValidationError)
async def on_invalid(request: Request, exc: RequestValidationError):
    e = exc.errors()[0]
    param = ".".join(str(p) for p in e["loc"] if p != "body")
    return oai_error(400, f"{param}: {e['msg']}", "invalid_request_error", param)


def chunk(meta, delta, finish=None):
    data = {**meta, "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
    return f"data: {json.dumps(data)}\n\n"


async def stream_chat(req, params, forced=False):
    meta = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": req.model}
    yield chunk(meta, {"role": "assistant", "content": ""})
    finish, text, buf, n_calls, in_tool = None, "", "", 0, forced
    async with http.stream("POST", BACKEND, json={**params, "stream": True}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])["choices"][0]
            finish = c.get("finish_reason") or finish
            text += c["text"]; buf += c["text"]
            while True:
                if in_tool:
                    if CLOSE not in buf: break
                    raw, buf = buf.split(CLOSE, 1)
                    deltas = tool_deltas(meta, n_calls, raw)
                    for d in deltas: yield d
                    n_calls += bool(deltas); in_tool = False
                elif req.tools and OPEN in buf:
                    before, buf = buf.split(OPEN, 1)
                    if before.strip(): yield chunk(meta, {"content": before})
                    in_tool = True
                else:
                    safe, buf = split_safe(buf) if req.tools else (buf, "")
                    if safe: yield chunk(meta, {"content": safe})
                    break
    if in_tool:  # forced mode has no closing tag; a truncated call gets dropped
        deltas = tool_deltas(meta, n_calls, buf)
        for d in deltas: yield d
        n_calls += bool(deltas)
    elif buf:
        yield chunk(meta, {"content": buf})
    yield chunk(meta, {}, "tool_calls" if n_calls else (finish or "stop"))
    if req.stream_options and req.stream_options.include_usage:
        pt, ct = len(tok.encode(params["prompt"])), len(tok.encode(text))
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        yield f"data: {json.dumps({**meta, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"


def schema_for(req):
    rf = req.response_format or {}
    if rf.get("type") == "json_schema":
        return rf["json_schema"]["schema"]
    if rf.get("type") == "json_object":
        return {"type": "object"}
    return None


def forced_tool_schema(req):
    tc = req.tool_choice
    if not req.tools or tc in (None, "auto", "none"):
        return None
    fns = [t["function"] for t in req.tools]
    if isinstance(tc, dict):
        fns = [f for f in fns if f["name"] == tc["function"]["name"]]
    return {"anyOf": [{"type": "object", "properties": {"name": {"const": f["name"]},
             "arguments": f.get("parameters", {"type": "object"})},
             "required": ["name", "arguments"]} for f in fns]}


def split_safe(buf):
    """Split buf into (safe to send now, held back because it might be the start of OPEN)."""
    for i in range(min(len(OPEN) - 1, len(buf)), 0, -1):
        if OPEN.startswith(buf[-i:]):
            return buf[:-i], buf[-i:]
    return buf, ""


def tool_deltas(meta, idx, raw):
    try: obj = json.loads(raw)
    except json.JSONDecodeError: return []
    first = {"tool_calls": [{"index": idx, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
             "function": {"name": obj["name"], "arguments": ""}}]}
    args = {"tool_calls": [{"index": idx, "function": {"arguments": json.dumps(obj.get("arguments", {}))}}]}
    return [chunk(meta, first), chunk(meta, args)]