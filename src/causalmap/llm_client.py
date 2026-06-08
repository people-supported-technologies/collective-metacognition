"""Provider-agnostic LLM client for structured JSON extraction.

Supports DeepSeek (default, OpenAI-compatible), OpenAI, Gemini, and Anthropic.
Selected via LLM_PROVIDER env var.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Type

from pydantic import BaseModel

from .config import (
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 2.0


def generate_structured(
    prompt: str,
    schema: Type[BaseModel],
    system_prompt: str = "",
    temperature: float = 0.0,
) -> BaseModel:
    """Call the configured LLM and parse the response into the given Pydantic schema."""
    provider = LLM_PROVIDER.lower()

    for attempt in range(MAX_RETRIES):
        try:
            if provider == "deepseek":
                raw = _call_deepseek(prompt, system_prompt, temperature)
            elif provider == "openai":
                raw = _call_openai(prompt, system_prompt, temperature)
            elif provider == "gemini":
                raw = _call_gemini(prompt, system_prompt, temperature)
            elif provider == "anthropic":
                raw = _call_anthropic(prompt, system_prompt, temperature)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

            parsed = _parse_json_response(raw, schema)
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY_S * (attempt + 1))

    raise RuntimeError("Exhausted retries")


def _parse_json_response(raw: str, schema: Type[BaseModel]) -> BaseModel:
    """Extract JSON from LLM response text and validate against schema."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    data = json.loads(text)
    return schema.model_validate(data)


def _build_json_schema_instruction(schema: Type[BaseModel]) -> str:
    """Generate a JSON schema description for the prompt."""
    s = schema.model_json_schema()
    return json.dumps(s, indent=2)


def _call_deepseek(prompt: str, system_prompt: str, temperature: float) -> str:
    """DeepSeek via OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_openai(prompt: str, system_prompt: str, temperature: float) -> str:
    """OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, system_prompt: str, temperature: float) -> str:
    """Google Gemini via google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text


def _call_anthropic(prompt: str, system_prompt: str, temperature: float) -> str:
    """Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt or "",
        messages=messages,
    )
    return response.content[0].text
