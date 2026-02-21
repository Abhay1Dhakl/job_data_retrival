from __future__ import annotations

from typing import Any, Dict

import httpx


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful career assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
