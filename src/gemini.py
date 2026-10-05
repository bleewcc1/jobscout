"""Small adapter around Google's Gemini Developer API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from google import genai
from google.genai import types


class Gemini:
    """Expose the message-shaped interface used by JobScout agents."""

    def __init__(self) -> None:
        self._client = genai.Client()
        self.messages = _Messages(self._client)


class _Messages:
    def __init__(self, client: genai.Client) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        output_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        user_content = "\n\n".join(
            message["content"] for message in messages
            if message.get("role") == "user"
        )
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if output_config:
            output_format = output_config["format"]
            config_kwargs.update({
                "response_mime_type": "application/json",
                "response_schema": output_format["schema"],
            })

        response = self._client.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=response.text)]
        )
