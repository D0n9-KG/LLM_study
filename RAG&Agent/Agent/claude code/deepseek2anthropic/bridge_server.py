import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "litellm_config.yaml"


def _load_deepseek_settings() -> Dict[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    api_base = os.getenv("DEEPSEEK_API_BASE", "").strip()
    model_name = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-chat"

    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        model_list = config.get("model_list") or []
        if model_list:
            params = (model_list[0] or {}).get("litellm_params") or {}
            raw_model = str(params.get("model") or "").strip()
            raw_api_base = str(params.get("api_base") or "").strip()
            raw_api_key = str(params.get("api_key") or "").strip()

            if raw_model:
                model_name = raw_model.split("/", 1)[-1]
            if raw_api_base and not api_base:
                api_base = raw_api_base
            if raw_api_key and not api_key:
                api_key = raw_api_key

    if not api_key:
        raise RuntimeError(
            "DeepSeek API key not found. Set DEEPSEEK_API_KEY or put api_key in litellm_config.yaml."
        )
    if not api_base:
        api_base = "https://api.deepseek.com/v1"

    return {
        "api_key": api_key,
        "api_base": api_base.rstrip("/"),
        "model": model_name,
    }


SETTINGS = _load_deepseek_settings()
CLIENT = OpenAI(api_key=SETTINGS["api_key"], base_url=SETTINGS["api_base"])


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoice(BaseModel):
    type: str
    function: Optional[ToolChoiceFunction] = None


class ToolSchema(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class MessageRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Dict[str, Any]]
    system: Optional[Any] = None
    tools: Optional[List[ToolSchema]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False


app = FastAPI(title="DeepSeek Anthropic Bridge")


def _anthropic_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
        return "".join(text_parts)
    return str(content)


def _system_messages(system: Any) -> List[Dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    if isinstance(system, list):
        return [
            {"role": "system", "content": _anthropic_text_from_content(system)}
        ]
    return [{"role": "system", "content": str(system)}]


def _assistant_message_from_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _convert_messages(request: MessageRequest) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    converted.extend(_system_messages(request.system))

    for msg in request.messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, list):
                tool_results = [
                    block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"
                ]
                if tool_results:
                    for block in tool_results:
                        converted.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": _anthropic_text_from_content(block.get("content")),
                            }
                        )
                else:
                    converted.append({"role": "user", "content": _anthropic_text_from_content(content)})
            else:
                converted.append({"role": "user", "content": _anthropic_text_from_content(content)})
        elif role == "assistant":
            if isinstance(content, list):
                converted.append(_assistant_message_from_blocks(content))
            else:
                converted.append({"role": "assistant", "content": _anthropic_text_from_content(content)})
        elif role == "system":
            converted.append({"role": "system", "content": _anthropic_text_from_content(content)})
        elif role == "tool":
            converted.append(msg)

    return converted


def _convert_tools(tools: Optional[List[ToolSchema]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def _convert_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return None
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        if choice_type == "tool":
            name = ((tool_choice.get("function") or {}).get("name")) or tool_choice.get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
    return None


def _anthropic_usage(resp: Any) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _anthropic_response(resp: Any, requested_model: str) -> Dict[str, Any]:
    choice = resp.choices[0]
    message = choice.message
    content: List[Dict[str, Any]] = []

    if getattr(message, "content", None):
        content.append({"type": "text", "text": message.content})

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        arguments = tool_call.function.arguments or "{}"
        try:
            parsed_input = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_input = {"raw_arguments": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": parsed_input,
            }
        )

    finish_reason = choice.finish_reason
    if tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "stop":
        stop_reason = "end_turn"
    else:
        stop_reason = "end_turn"

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage(resp),
    }


def _validate_request(request: MessageRequest, anthropic_version: Optional[str]) -> None:
    if request.stream:
        raise HTTPException(status_code=400, detail={"error": {"type": "invalid_request_error", "message": "stream=true is not supported by this bridge yet"}})
    if not anthropic_version:
        raise HTTPException(status_code=400, detail={"error": {"type": "invalid_request_error", "message": "Missing anthropic-version header"}})


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "provider": "deepseek",
        "api_base": SETTINGS["api_base"],
        "model": SETTINGS["model"],
        "routes": ["/v1/messages", "/anthropic/v1/messages"],
    }


def _handle_message_request(request: MessageRequest, anthropic_version: Optional[str]) -> Dict[str, Any]:
    _validate_request(request, anthropic_version)

    openai_messages = _convert_messages(request)
    openai_tools = _convert_tools(request.tools)
    openai_tool_choice = _convert_tool_choice(request.tool_choice)

    kwargs: Dict[str, Any] = {
        "model": SETTINGS["model"],
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
    }
    if openai_tools:
        kwargs["tools"] = openai_tools
    if openai_tool_choice is not None:
        kwargs["tool_choice"] = openai_tool_choice
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop_sequences:
        kwargs["stop"] = request.stop_sequences

    try:
        response = CLIENT.chat.completions.create(**kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": {"type": "api_error", "message": str(exc)}},
        ) from exc

    return _anthropic_response(response, request.model)


@app.post("/v1/messages")
def v1_messages(
    request: MessageRequest,
    anthropic_version: Optional[str] = Header(default=None, alias="anthropic-version"),
) -> Dict[str, Any]:
    return _handle_message_request(request, anthropic_version)


@app.post("/anthropic/v1/messages")
def anthropic_v1_messages(
    request: MessageRequest,
    anthropic_version: Optional[str] = Header(default=None, alias="anthropic-version"),
) -> Dict[str, Any]:
    return _handle_message_request(request, anthropic_version)

