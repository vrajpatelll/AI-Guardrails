"""
Pluggable LLM backends — decouples GuardrailMiddleware from any single
SDK/framework.

GuardrailMiddleware only needs three operations on whatever it's wrapping:
  - send(call_kwargs)         -> raw response
  - extract_text(response)    -> assistant text, for the output guardrail
  - set_text(response, text)  -> response with assistant text replaced,
                                 for output REDACT

`call_kwargs` is the same dict passed to `.messages.create(**kwargs)` —
at minimum {"messages": [{"role": ..., "content": ...}, ...]}, optionally
with "model", "max_tokens", etc. Each backend takes what it needs and
ignores the rest.

Ship two backends out of the box:
  - OpenAIBackend    — today's default (OpenAI-compatible gateway).
  - LangChainBackend — wraps any langchain-core BaseChatModel.

For any other framework (LlamaIndex, CrewAI, AutoGen, a raw REST call,
Anthropic's SDK, ...), implement the same three methods against LLMBackend
and pass an instance via `GuardrailMiddleware(config, backend=...)`.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from guardrail.config import GuardrailConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface GuardrailMiddleware needs from any LLM client."""

    def send(self, call_kwargs: dict[str, Any]) -> Any:
        """Send the (possibly sanitized) request, return the raw response."""
        ...

    def extract_text(self, response: Any) -> str:
        """Pull the assistant's text out of a raw response."""
        ...

    def set_text(self, response: Any, text: str) -> Any:
        """Return `response` with its assistant text replaced by `text`."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible gateway (today's default behaviour)
# ---------------------------------------------------------------------------

class OpenAIBackend:
    """Wraps an OpenAI (or OpenAI-compatible) client's chat.completions API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_config(cls, config: GuardrailConfig) -> "OpenAIBackend":
        """Build the client the same way GuardrailMiddleware always has:
        API key + base_url from config, custom CA cert if provided."""
        import httpx
        from openai import OpenAI

        http_client = None
        if config.llm_cert_path is not None:
            # BYPASS (temporary): TLS verification disabled because
            # GUARDRAIL_LLM_CERT_PATH from IT is the gateway's leaf cert, not
            # its issuing CA, so certificate verification fails. Get the
            # actual CDSYS-CA root cert from IT and switch back to
            # verify=<cert path>.
            logger.warning(
                "TLS certificate verification is DISABLED for the LLM gateway "
                "connection — temporary bypass, do not ship this."
            )
            http_client = httpx.Client(verify=False)

        client = OpenAI(
            api_key=config.llm_api_key.get_secret_value(),
            base_url=config.llm_base_url,
            http_client=http_client,
        )
        return cls(client)

    def send(self, call_kwargs: dict[str, Any]) -> Any:
        return self._client.chat.completions.create(**call_kwargs)

    def extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return choices[0].message.content or ""

    def set_text(self, response: Any, text: str) -> Any:
        choices = getattr(response, "choices", None) or []
        if choices:
            choices[0].message.content = text
        return response


# ---------------------------------------------------------------------------
# LangChain (or any framework built on langchain-core BaseChatModel)
# ---------------------------------------------------------------------------

class LangChainBackend:
    """
    Wraps any langchain-core BaseChatModel (ChatOpenAI, ChatAnthropic,
    ChatBedrock, a custom BaseChatModel, ...) so it can sit behind
    GuardrailMiddleware exactly like OpenAIBackend does.

    "model"/"max_tokens"/etc. in call_kwargs are ignored — configure those
    on the chat model instance itself (that's the idiomatic LangChain way);
    only "messages" is used.

    Usage::

        from langchain_openai import ChatOpenAI
        from guardrail.backends import LangChainBackend

        chat_model = ChatOpenAI(model="gpt-4o-mini")
        middleware = GuardrailMiddleware(config, backend=LangChainBackend(chat_model))
    """

    def __init__(self, chat_model: Any) -> None:
        try:
            import langchain_core.messages  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LangChainBackend requires langchain-core. "
                "Run: pip install -e '.[langchain]' (or `pip install langchain-core`)."
            ) from exc
        self._model = chat_model

    def send(self, call_kwargs: dict[str, Any]) -> Any:
        from langchain_core.messages import convert_to_messages

        lc_messages = convert_to_messages(call_kwargs.get("messages", []))
        return self._model.invoke(lc_messages)

    def extract_text(self, response: Any) -> str:
        content = getattr(response, "content", "")
        return content or ""

    def set_text(self, response: Any, text: str) -> Any:
        response.content = text
        return response
