# llm_huggingface.py
# --------------------------------------------------------------
# Full-featured LLM plugin for the Hugging Face Inference API.
# Features:
#   - Streaming and non-streaming
#   - JSON schema output (prompt.schema)
#   - Function calling / tools (prompt.tools)
#   - Rich Options with validation (temperature, top_p, max tokens, stop, delay)
#   - Response metadata logging into response.response_json
#   - Lazy client initialization (does not require key at import time)
# --------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Iterable, Iterator, Optional

import llm
from pydantic import Field, field_validator

from huggingface_hub import (
    AsyncInferenceClient,
    ChatCompletionInputFunctionDefinition,
    ChatCompletionInputJSONSchema,
    ChatCompletionInputResponseFormatJSONSchema,
    ChatCompletionInputResponseFormatJSONObject,
    ChatCompletionInputTool,
    InferenceClient,
    ModelInfo,
    list_models,
)


# --------------------------------------------------------------
# 1) File caching configuration
# --------------------------------------------------------------
# Cache settings (can be overridden via environment variables)
CACHE_EXPIRY_SECONDS = int(os.environ.get("LLM_HUGGINGFACE_CACHE_EXPIRY", 7 * 24 * 60 * 60))  # 7 days default
FORCE_REFRESH = os.environ.get("LLM_HUGGINGFACE_FORCE_REFRESH", "").lower() in ("1", "true", "yes")


def _get_cache_dir() -> Path:
    """Get or create the cache directory for this plugin."""
    cache_dir = llm.user_dir() / "llm-huggingface"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def _load_cached_models(cache_file: Path) -> Optional[list[str]]:
    """Load model list from cache if valid, return None if expired or invalid."""
    if FORCE_REFRESH:
        return None
        
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        # Check cache expiry
        cached_time = cache_data.get("timestamp", 0)
        if time.time() - cached_time > CACHE_EXPIRY_SECONDS:
            return None
        
        return cache_data.get("models", [])
    except (json.JSONDecodeError, IOError, KeyError):
        # Cache is corrupted, will refresh
        return None


def _save_models_to_cache(cache_file: Path, models: list[str]) -> None:
    """Save model list to cache file with timestamp."""
    cache_data = {
        "timestamp": time.time(),
        "models": models
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        raise llm.ModelError(f"Failed to save model list to cache: {cache_file}") from e


# --------------------------------------------------------------
# 2) Discover and cache available models
# --------------------------------------------------------------
def _get_text_generation_models() -> list[str]:
    """
    Return a list of model IDs that support the `text-generation` task.
    Uses file cache to avoid repeated Hub calls.
    """
    cache_file = _get_cache_dir() / "text-generation-models.json"
    
    # Try to load from cache
    cached_models = _load_cached_models(cache_file)
    if cached_models is not None:
        return cached_models
    
    # Fetch from API
    try:
        models: Iterable[ModelInfo] = list_models(inference_provider="all", filter="text-generation")
        model_ids = [model.id for model in models]
        
        # Save to cache
        _save_models_to_cache(cache_file, model_ids)
        
        return model_ids
    except Exception:
        # If API fails and we have any old cache, use it
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                return cache_data.get("models", [])
            except:
                pass
        # No cache and API failed, return empty list
        return []


def _get_image_text_to_text_models() -> list[str]:
    """
    Return a list of model IDs that support the `image-text-to-text` task.
    Uses file cache to avoid repeated Hub calls.
    """
    cache_file = _get_cache_dir() / "image-text-to-text-models.json"
    
    # Try to load from cache
    cached_models = _load_cached_models(cache_file)
    if cached_models is not None:
        return cached_models
    
    # Fetch from API
    try:
        models: Iterable[ModelInfo] = list_models(inference_provider="all", filter="image-text-to-text")
        model_ids = [model.id for model in models]
        
        # Save to cache
        _save_models_to_cache(cache_file, model_ids)
        
        return model_ids
    except Exception:
        # If API fails and we have any old cache, use it
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                return cache_data.get("models", [])
            except:
                pass
        # No cache and API failed, return empty list
        return []

# --------------------------------------------------------------
# 3) Register all discovered models with LLM
# --------------------------------------------------------------
@llm.hookimpl
def register_models(register):
    for model_id in _get_text_generation_models():
        register(
            HuggingFace(model_id),
            AsyncHuggingFace(model_id),
            aliases=[f"hf/{model_id}"],
            )
    for model_id in _get_image_text_to_text_models():
        register(
            HuggingFace(model_id, is_image_text_to_text=True),
            AsyncHuggingFace(model_id, is_image_text_to_text=True),
            aliases=[f"hf/{model_id}"],
            )


# --------------------------------------------------------------
# 4) Model implementation
# --------------------------------------------------------------
class _HuggingFaceBase():
    """
    LLM model that forwards chat completions to Hugging Face Inference API.

    Supports:
      - Streaming responses
      - JSON schema output (prompt.schema)
      - Tools/function-calling (prompt.tools)
      - Options for temperature, top-p, stop, max tokens
      - Rich response metadata logged via response.response_json
      - Attachments for multi-modal models (images)
    """
    def __init__(self, model_id: str, is_image_text_to_text: bool = False):
        super().__init__()
        # User-visible ID (prefix with "hf/" to disambiguate in `llm models`)
        self.model_id = f"hf/{model_id}"
        # Raw Hugging Face model ID
        self._hf_model_id = model_id
        # Lazy client – only created when first used
        self._hf_client: Optional[InferenceClient] = None
        self.is_image_text_to_text = is_image_text_to_text
        
        # Only set attachment_types for models that support multi-modal input
        # LLM will use this to validate attachments before passing to execute()
        if is_image_text_to_text:
            self.attachment_types = {
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
            }

    # Helper to validate key is available
    def _validate_key(self, key: str | None = None) -> str:
        if not key:
            raise llm.ModelError(
                "HuggingFace requires an API key — run `llm keys set huggingface` "
                "or set the HUGGINGFACE_TOKEN environment variable"
            )
        return key

    # ----------------------------------------------------------
    # Build the message list for chat-completions
    # ----------------------------------------------------------
    def _build_messages(
        self, prompt: llm.Prompt, conversation: llm.Conversation | llm.AsyncConversation | None = None
    ) -> list[dict]:
        messages: list[dict] = []

        # System instructions
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        if prompt.system_fragments:
            for fragment in prompt.system_fragments:
                messages.append({"role": "system", "content": fragment})

        # Conversation history
        if conversation:
            for resp in conversation.responses:
                # Prior user turn - handle attachments from history
                if resp.attachments:
                    # Had attachments in this prior turn
                    user_content = []
                    if resp.prompt.prompt:
                        user_content.append({"type": "text", "text": resp.prompt.prompt})
                    
                    for attachment in resp.attachments:
                        attachment_type = attachment.resolve_type()
                        if attachment_type.startswith("image/"):
                            if attachment.url:
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": attachment.url}
                                })
                            else:
                                base64_content = attachment.base64_content()
                                data_url = f"data:{attachment_type};base64,{base64_content}"
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": data_url}
                                })
                    
                    fragments = getattr(resp.prompt, "fragments", None) or []
                    for fragment in fragments:
                        user_content.append({"type": "text", "text": fragment})
                    
                    if user_content:
                        messages.append({"role": "user", "content": user_content})
                else:
                    # No attachments - simple text
                    if resp.prompt.prompt:
                        messages.append({"role": "user", "content": resp.prompt.prompt})
                    fragments = getattr(resp.prompt, "fragments", None) or []
                    for fragment in fragments:
                        messages.append({"role": "user", "content": fragment})

                # Prior assistant turn - may include tool calls and/or content
                assistant_msg: dict = {"role": "assistant"}
                # text content (if present)
                assistant_text = resp.text_or_raise() if isinstance(resp, llm.Response) else None
                if assistant_text:
                    assistant_msg["content"] = assistant_text

                # tool calls (if any)
                tool_calls_payload = []
                try:
                    prev_tool_calls = resp.tool_calls_or_raise() if isinstance(resp, llm.Response) else []
                except Exception:
                    prev_tool_calls = []
                for tc in prev_tool_calls or []:
                    tool_calls_payload.append(
                        {
                            "type": "function",
                            "id": tc.tool_call_id,
                            "function": {
                                "name": tc.name,
                                # API expects a JSON-encoded string
                                "arguments": json.dumps(tc.arguments or {}),
                            },
                        }
                    )
                if tool_calls_payload:
                    assistant_msg["tool_calls"] = tool_calls_payload

                # Only append if assistant had something
                if assistant_msg.keys() - {"role"}:
                    messages.append(assistant_msg)

                # Tool results from that prior turn (if any)
                if resp.prompt.tool_results:
                    for tr in resp.prompt.tool_results:
                        tool_msg = {
                            "role": "tool",
                            "content": tr.output,
                        }
                        if tr.tool_call_id:
                            tool_msg["tool_call_id"] = tr.tool_call_id
                        messages.append(tool_msg)

        # Current user turn - handle attachments if present
        if not prompt.attachments:
            # No attachments - simple text messages
            if prompt.prompt:
                messages.append({"role": "user", "content": prompt.prompt})
            if prompt.fragments:
                for fragment in prompt.fragments:
                    messages.append({"role": "user", "content": fragment})
        else:
            # Has attachments - model supports them since attachment_types is set
            user_content = []
            if prompt.prompt:
                user_content.append({"type": "text", "text": prompt.prompt})
            
            # Add each attachment
            for attachment in prompt.attachments:
                attachment_type = attachment.resolve_type()
                if attachment_type.startswith("image/"):
                    # For images, use URL if available, otherwise base64
                    if attachment.url:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": attachment.url}
                        })
                    else:
                        # Base64 encode the image
                        base64_content = attachment.base64_content()
                        data_url = f"data:{attachment_type};base64,{base64_content}"
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                else:
                    # Other attachment types not yet supported
                    raise llm.ModelError(f"Attachment type {attachment_type} is not supported")
            
            # Add fragments after attachments
            if prompt.fragments:
                for fragment in prompt.fragments:
                    user_content.append({"type": "text", "text": fragment})
            
            if user_content:
                messages.append({"role": "user", "content": user_content})

        # Any tool results already obtained in this turn
        if prompt.tool_results:
            for tr in prompt.tool_results:
                tool_msg = {
                    "role": "tool",
                    "content": tr.output,
                }
                if tr.tool_call_id:
                    tool_msg["tool_call_id"] = tr.tool_call_id
                messages.append(tool_msg)

        return messages

    # Helper: build response_format if schema provided
    def _build_response_format(self, prompt: llm.Prompt):
        """
        HF chat-completions follow the OpenAI-compatible response_format.

        If prompt.schema is provided (as a dict JSON Schema), request json_schema mode.
        Otherwise, return None. Users that want a generic JSON object can pass a minimal schema
        or we fall back to json_object if the schema is not a dict for any reason.
        """
        if not prompt.schema:
            return None

        # LLM ensures this is a dict even if provided as Pydantic in most cases.
        schema_obj = prompt.schema
        if isinstance(schema_obj, dict):
            return ChatCompletionInputResponseFormatJSONSchema(
                type="json_schema",
                json_schema=ChatCompletionInputJSONSchema(
                    name="response",
                    schema=schema_obj,
                    strict=True,
                ),
            )

        # Fallback: request a generic JSON object
        return ChatCompletionInputResponseFormatJSONObject(type="json_object")

    # Helper: build tools parameter if tools provided
    def _build_tools_param(self, prompt: llm.Prompt):
        if not prompt.tools:
            return None
        return [
            ChatCompletionInputTool(
                type="function",
                function=ChatCompletionInputFunctionDefinition(
                    name=tool.name,
                    parameters=tool.input_schema,
                    description=tool.description,
                ),
            )
            for tool in prompt.tools
        ]

    # Helper to set usage if provided by API
    def _maybe_set_usage(self, response, usage_obj):
        if not usage_obj:
            return
        input_tokens = getattr(usage_obj, "prompt_tokens", None)
        output_tokens = getattr(usage_obj, "completion_tokens", None)
        details = None
        # Try to serialize the full usage dict for details
        for attr in ("model_dump", "dict"):
            fn = getattr(usage_obj, attr, None)
            if callable(fn):
                try:
                    details = fn()
                    break
                except Exception:
                    pass
        if details is None:
            details = getattr(usage_obj, "__dict__", None)
        try:
            response.set_usage(
                input=int(input_tokens) if input_tokens is not None else None,
                output=int(output_tokens) if output_tokens is not None else None,
                details=details,
            )
        except Exception:
            # set_usage is optional; ignore if not available or incompatible
            pass


class HuggingFace(_HuggingFaceBase, llm.KeyModel):
    needs_key = "huggingface"
    key_env_var = "HUGGINGFACE_TOKEN"
    can_stream = True
    supports_schema = True
    supports_tools = True
    # attachment_types is set dynamically in __init__ based on model capabilities

    # Options (Pydantic 2 via llm.Options)
    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description="Sampling temperature (higher => more random)",
            default=None,
        )
        top_p: Optional[float] = Field(
            description="Nucleus sampling – keep top‑p probability mass",
            default=None,
        )
        max_tokens: Optional[int] = Field(
            description="Maximum number of new tokens to generate (chat-completions style)",
            default=None,
        )
        stop: Optional[list[str]] = Field(
            description="Stop sequences - generation halts if any is encountered",
            default=None,
        )

        @field_validator("temperature")
        @classmethod
        def _validate_temp(cls, v: Optional[float]) -> Optional[float]:
            if v is None:
                return v
            if not 0.0 <= v <= 2.0:
                raise ValueError("temperature must be between 0 and 2")
            return v

        @field_validator("top_p")
        @classmethod
        def _validate_top_p(cls, v: Optional[float]) -> Optional[float]:
            if v is None:
                return v
            if not 0.0 <= v <= 1.0:
                raise ValueError("top_p must be between 0 and 1")
            return v

        @field_validator("max_tokens")
        @classmethod
        def _validate_max_tokens(cls, v: Optional[int]) -> Optional[int]:
            if v is None:
                return v
            if v < 1:
                raise ValueError("max_tokens must be >= 1")
            return v

        @field_validator("max_tokens")
        @classmethod
        def _mutual_exclusion_max_tokens(cls, v, info):
            """
            Guard against both max_tokens and max_new_tokens being supplied.
            """
            data = info.data
            if v is not None and data.get("max_tokens") is not None:
                raise ValueError("Set either max_tokens or max_new_tokens, not both")
            return v

    def __init__(self, model_id: str, is_image_text_to_text: bool = False):
        super().__init__(model_id, is_image_text_to_text)
        self._hf_client = None

    def _client(self, key: str | None = None) -> "InferenceClient":
        if self._hf_client is not None:
            return self._hf_client
        key = key or self.get_key()
        key = self._validate_key(key)
        from huggingface_hub import InferenceClient
        self._hf_client = InferenceClient(provider="auto", api_key=key)
        return self._hf_client

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
        key: str | None = None,
    ) -> Iterator[str]:
        client = self._client(key)
        messages = self._build_messages(prompt, conversation)

        # Map options to HF chat-completions kwargs
        opts = prompt.options
        # Prefer max_tokens (chat style). If not set, fall back to max_new_tokens.
        max_tokens = getattr(opts, "max_tokens", None)

        hf_kwargs = {
            "temperature": getattr(opts, "temperature", None),
            "top_p": getattr(opts, "top_p", None),
            # HF currently accepts `max_tokens` for chat-completions.
            # We map a provided max_new_tokens as `max_tokens` for compatibility.
            "max_tokens": max_tokens,
            "stop": getattr(opts, "stop", None),
        }
        hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}

        response_format = self._build_response_format(prompt)
        tools_param = self._build_tools_param(prompt)

        if stream:
            last_meta = {}
            # Accumulate tool call deltas across stream
            # Map index -> {"id": str|None, "name": str|None, "arguments": str}
            accumulated_tool_calls: dict[int, dict] = {}

            for stream_result in client.chat_completion(
                messages=messages,
                model=self._hf_model_id,
                stream=True,
                response_format=response_format,
                tools=tools_param,
                **hf_kwargs,
            ):
                choice = None
                if getattr(stream_result, "choices", None):
                    choice = stream_result.choices[0]

                # Yield any new content
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    chunk = getattr(delta, "content", None)
                    if chunk:
                        yield chunk

                    # Accumulate tool call deltas
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        for tc_delta in delta_tool_calls:
                            idx = getattr(tc_delta, "index", 0) or 0
                            entry = accumulated_tool_calls.setdefault(
                                idx, {"id": None, "name": None, "arguments": ""}
                            )
                            tc_id = getattr(tc_delta, "id", None)
                            if tc_id:
                                entry["id"] = tc_id
                            function_obj = getattr(tc_delta, "function", None)
                            if function_obj:
                                fn_name = getattr(function_obj, "name", None)
                                if fn_name:
                                    entry["name"] = fn_name
                                fn_args = getattr(function_obj, "arguments", None)
                                if fn_args:
                                    entry["arguments"] += fn_args

                # Track final metadata when finish_reason appears
                finish_reason = getattr(choice, "finish_reason", None) if choice else None
                if finish_reason:
                    last_meta = {
                        "usage": getattr(stream_result, "usage", None),
                        "model": getattr(stream_result, "model", None),
                        "finish_reason": finish_reason,
                        "created": getattr(stream_result, "created", None),
                        "id": getattr(stream_result, "id", None),
                    }

            # Record any tool calls that were requested
            if accumulated_tool_calls:
                for entry in accumulated_tool_calls.values():
                    name = entry.get("name")
                    args_str = entry.get("arguments") or ""
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except Exception:
                        # Fall back to raw string if not valid JSON
                        args = {"__raw": args_str}
                    response.add_tool_call(
                        llm.ToolCall(
                            name=name or "",
                            arguments=args or {},
                            tool_call_id=entry.get("id"),
                        )
                    )

            # Write metadata after stream completion (if provided)
            if last_meta:
                response.response_json = last_meta
                # Attempt to set usage and resolved model for logs
                self._maybe_set_usage(response, last_meta.get("usage"))
                resolved_model = last_meta.get("model")
                if resolved_model:
                    try:
                        response.set_resolved_model(resolved_model)
                    except Exception:
                        pass

        else:
            result = client.chat_completion(
                messages=messages,
                model=self._hf_model_id,
                stream=False,
                response_format=response_format,
                tools=tools_param,
                **hf_kwargs,
            )

            # Extract full message content and tool calls
            content = None
            if getattr(result, "choices", None):
                message_obj = result.choices[0].message
                # content
                content = getattr(message_obj, "content", None) or getattr(
                    message_obj, "text", None
                )
                # tool calls
                tc_list = getattr(message_obj, "tool_calls", None) or []
                for tc in tc_list:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) if fn else None
                    args_str = getattr(fn, "arguments", None) if fn else None
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except Exception:
                        args = {"__raw": args_str}
                    response.add_tool_call(
                        llm.ToolCall(
                            name=name or "",
                            arguments=args or {},
                            tool_call_id=getattr(tc, "id", None),
                        )
                    )

            if content:
                yield content

            usage = getattr(result, "usage", None)
            resolved_model = getattr(result, "model", None)

            response.response_json = {
                "usage": usage,
                "model": resolved_model,
                "finish_reason": result.choices[0].finish_reason if result.choices else None,
                "created": getattr(result, "created", None),
                "id": getattr(result, "id", None),
            }

            # Attempt to set usage and resolved model for logs
            self._maybe_set_usage(response, usage)
            if resolved_model:
                try:
                    response.set_resolved_model(resolved_model)
                except Exception:
                    pass


class AsyncHuggingFace(_HuggingFaceBase, llm.AsyncKeyModel):
    needs_key = "huggingface"
    key_env_var = "HUGGINGFACE_TOKEN"
    can_stream = True
    supports_schema = True
    supports_tools = True
    # attachment_types is set dynamically in __init__ based on model capabilities

    # Options (Pydantic 2 via llm.Options)
    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description="Sampling temperature (higher => more random)",
            default=None,
        )
        top_p: Optional[float] = Field(
            description="Nucleus sampling – keep top‑p probability mass",
            default=None,
        )
        max_tokens: Optional[int] = Field(
            description="Maximum number of new tokens to generate (chat-completions style)",
            default=None,
        )
        stop: Optional[list[str]] = Field(
            description="Stop sequences - generation halts if any is encountered",
            default=None,
        )

        @field_validator("temperature")
        @classmethod
        def _validate_temp(cls, v: Optional[float]) -> Optional[float]:
            if v is None:
                return v
            if not 0.0 <= v <= 2.0:
                raise ValueError("temperature must be between 0 and 2")
            return v

        @field_validator("top_p")
        @classmethod
        def _validate_top_p(cls, v: Optional[float]) -> Optional[float]:
            if v is None:
                return v
            if not 0.0 <= v <= 1.0:
                raise ValueError("top_p must be between 0 and 1")
            return v

        @field_validator("max_tokens")
        @classmethod
        def _validate_max_tokens(cls, v: Optional[int]) -> Optional[int]:
            if v is None:
                return v
            if v < 1:
                raise ValueError("max_tokens must be >= 1")
            return v

        @field_validator("max_tokens")
        @classmethod
        def _mutual_exclusion_max_tokens(cls, v, info):
            """
            Guard against both max_tokens and max_new_tokens being supplied.
            """
            data = info.data
            if v is not None and data.get("max_tokens") is not None:
                raise ValueError("Set either max_tokens or max_new_tokens, not both")
            return v

    def __init__(self, model_id: str, is_image_text_to_text: bool = False):
        super().__init__(model_id, is_image_text_to_text)
        self._hf_async_client: Optional[AsyncInferenceClient] = None

    def _client(self, key: str | None = None) -> "AsyncInferenceClient":
        if self._hf_async_client is not None:
            return self._hf_async_client
        key = key or self.get_key()
        key = self._validate_key(key)
        from huggingface_hub import AsyncInferenceClient
        self._hf_async_client = AsyncInferenceClient(provider="auto", api_key=key)
        return self._hf_async_client

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
        key: str | None = None,
    ) -> AsyncGenerator[str, None]:
        client = self._client(key)
        messages = self._build_messages(prompt, conversation)

        # Map options to HF chat-completions kwargs
        opts = prompt.options
        max_tokens = getattr(opts, "max_tokens", None)

        hf_kwargs = {
            "temperature": getattr(opts, "temperature", None),
            "top_p": getattr(opts, "top_p", None),
            "max_tokens": max_tokens,
            "stop": getattr(opts, "stop", None),
        }
        hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}

        response_format = self._build_response_format(prompt)
        tools_param = self._build_tools_param(prompt)

        if stream:
            last_meta = {}
            # Accumulate tool call deltas across stream
            accumulated_tool_calls: dict[int, dict] = {}

            async for stream_result in await client.chat_completion(
                messages=messages,
                model=self._hf_model_id,
                stream=True,
                response_format=response_format,
                tools=tools_param,
                **hf_kwargs,
            ):
                choice = None
                if getattr(stream_result, "choices", None):
                    choice = stream_result.choices[0]

                # Yield any new content
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    chunk = getattr(delta, "content", None)
                    if chunk:
                        yield chunk

                    # Accumulate tool call deltas
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        for tc_delta in delta_tool_calls:
                            idx = getattr(tc_delta, "index", 0) or 0
                            entry = accumulated_tool_calls.setdefault(
                                idx, {"id": None, "name": None, "arguments": ""}
                            )
                            tc_id = getattr(tc_delta, "id", None)
                            if tc_id:
                                entry["id"] = tc_id
                            function_obj = getattr(tc_delta, "function", None)
                            if function_obj:
                                fn_name = getattr(function_obj, "name", None)
                                if fn_name:
                                    entry["name"] = fn_name
                                fn_args = getattr(function_obj, "arguments", None)
                                if fn_args:
                                    entry["arguments"] += fn_args

                # Track final metadata when finish_reason appears
                finish_reason = getattr(choice, "finish_reason", None) if choice else None
                if finish_reason:
                    last_meta = {
                        "usage": getattr(stream_result, "usage", None),
                        "model": getattr(stream_result, "model", None),
                        "finish_reason": finish_reason,
                        "created": getattr(stream_result, "created", None),
                        "id": getattr(stream_result, "id", None),
                    }

            # Record any tool calls that were requested
            if accumulated_tool_calls:
                for entry in accumulated_tool_calls.values():
                    name = entry.get("name")
                    args_str = entry.get("arguments") or ""
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except Exception:
                        # Fall back to raw string if not valid JSON
                        args = {"__raw": args_str}
                    response.add_tool_call(
                        llm.ToolCall(
                            name=name or "",
                            arguments=args or {},
                            tool_call_id=entry.get("id"),
                        )
                    )

            # Write metadata after stream completion (if provided)
            if last_meta:
                response.response_json = last_meta
                # Attempt to set usage and resolved model for logs
                self._maybe_set_usage(response, last_meta.get("usage"))
                resolved_model = last_meta.get("model")
                if resolved_model:
                    try:
                        response.set_resolved_model(resolved_model)
                    except Exception:
                        pass

        else:
            result = await client.chat_completion(
                messages=messages,
                model=self._hf_model_id,
                stream=False,
                response_format=response_format,
                tools=tools_param,
                **hf_kwargs,
            )

            # Extract full message content and tool calls
            content = None
            if getattr(result, "choices", None):
                message_obj = result.choices[0].message
                # content
                content = getattr(message_obj, "content", None) or getattr(
                    message_obj, "text", None
                )
                # tool calls
                tc_list = getattr(message_obj, "tool_calls", None) or []
                for tc in tc_list:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) if fn else None
                    args_str = getattr(fn, "arguments", None) if fn else None
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except Exception:
                        args = {"__raw": args_str}
                    response.add_tool_call(
                        llm.ToolCall(
                            name=name or "",
                            arguments=args or {},
                            tool_call_id=getattr(tc, "id", None),
                        )
                    )

            if content:
                yield content

            usage = getattr(result, "usage", None)
            resolved_model = getattr(result, "model", None)

            response.response_json = {
                "usage": usage,
                "model": resolved_model,
                "finish_reason": result.choices[0].finish_reason if result.choices else None,
                "created": getattr(result, "created", None),
                "id": getattr(result, "id", None),
            }

            # Attempt to set usage and resolved model for logs
            self._maybe_set_usage(response, usage)
            if resolved_model:
                try:
                    response.set_resolved_model(resolved_model)
                except Exception:
                    pass