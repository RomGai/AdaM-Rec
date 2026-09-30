"""Concurrent item profiling through a vLLM OpenAI-compatible server."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from typing import Any, Dict, List, Optional


class ProfileBackendError(RuntimeError):
    """A serving failure that must not silently fall back to text profiling."""


class VLLMItemProfiler:
    """Generate text/VL profiles using a shared vLLM server."""

    def __init__(self, model_name, base_url="http://127.0.0.1:8000/v1", timeout=600,
                 max_new_tokens=None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else int(os.getenv("out_seq_length", "16384"))
        if self.max_new_tokens < 1:
            raise ValueError("profile max_new_tokens must be positive")
        self.do_sample = os.getenv("greedy", "false").lower() != "true"
        self.top_p = float(os.getenv("top_p", "0.8"))
        self.top_k = int(os.getenv("top_k", "20"))
        self.temperature = float(os.getenv("temperature", "0.7"))
        self.repetition_penalty = float(os.getenv("repetition_penalty", "1.0"))
        self.json_retry = int(os.getenv("json_retry", "1"))
        self.base_url = base_url.rstrip("/")
        if urlparse(self.base_url).scheme not in {"http", "https"}:
            raise ValueError("vLLM base URL must use http or https")
        if timeout <= 0:
            raise ValueError("vLLM timeout must be positive")
        self.timeout = timeout
        self.api_key = os.getenv("VLLM_API_KEY", "")

    def _request(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(f"{self.base_url}/{path}", data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ProfileBackendError(f"vLLM HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise ProfileBackendError(f"vLLM request failed: {exc}") from exc

    def check_connection(self):
        models = self._request("models")
        available = [entry.get("id") for entry in models.get("data", [])]
        if self.model_name not in available:
            raise ProfileBackendError(
                f"vLLM must serve {self.model_name!r}; available models: {available}"
            )

    def _image_url(self, source):
        if source.startswith("data:image/"):
            return source
        if urlparse(source).scheme in {"http", "https"}:
            with urlopen(source, timeout=min(self.timeout, 30)) as response:
                content = response.read()
        else:
            content = Path(source).read_bytes()
        from PIL import Image

        with Image.open(BytesIO(content)) as image:
            mime = Image.MIME.get(image.format, "image/jpeg")
            image.verify()
        return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"

    def _generate_text(self, messages, force_greedy=False):
        converted = []
        for message in messages:
            parts = []
            for part in message["content"]:
                if part["type"] == "image":
                    parts.append({"type": "image_url", "image_url": {
                        "url": self._image_url(part["image"]),
                    }})
                else:
                    parts.append(dict(part))
            converted.append({"role": message["role"], "content": parts})
        sampling = not force_greedy and self.do_sample
        payload = {
            "model": self.model_name,
            "messages": converted,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if sampling else 0.0,
            "top_p": self.top_p if sampling else 1.0,
            "top_k": self.top_k if sampling else -1,
            "repetition_penalty": self.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        result = self._request("chat/completions", payload)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProfileBackendError("vLLM returned an invalid chat completion") from exc
        if choice.get("finish_reason") == "length":
            raise ProfileBackendError(
                "vLLM profile output was truncated; increase --profile-max-new-tokens "
                "and ensure the server context can hold both input and output."
            )
        if not isinstance(content, str) or not content.strip():
            raise ProfileBackendError("vLLM returned no profile text")
        return content

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        stripped = text.strip()

        # 1) direct decode
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        # 2) markdown json code fence
        if "```" in stripped:
            parts = stripped.split("```")
            for part in parts:
                candidate = part.replace("json", "", 1).strip()
                if not candidate:
                    continue
                try:
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue

        # 3) find first decodable JSON object from any '{' start
        for i, ch in enumerate(stripped):
            if ch != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(stripped, i)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        return None


    @staticmethod
    def _normalize_image_paths(image_paths: List[str]) -> List[str]:
        """Drop empty/obviously invalid image entries before processor ingestion."""
        cleaned: List[str] = []
        for p in image_paths:
            cand = str(p or "").strip()
            if not cand:
                continue
            # Avoid placeholders that are known to break image loading.
            if cand in {".", "./", "..", "../"}:
                continue
            cleaned.append(cand)
        return cleaned


    def extract(
        self,
        prompt: str,
        image_paths: List[str],
        text_only_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        valid_image_paths = self._normalize_image_paths(image_paths)
        image_messages = [{"type": "image", "image": path} for path in valid_image_paths]
        if not image_messages and text_only_prompt is not None:
            prompt = text_only_prompt
        messages = [
            {
                "role": "user",
                "content": [
                    *image_messages,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            generated_text = self._generate_text(messages)
        except Exception as exc:
            if not image_messages or isinstance(exc, ProfileBackendError):
                raise
            # Some rows contain invalid image URLs/paths; fallback to text-only profiling.
            print(f"[VLLMItemProfiler] image loading failed, fallback to text-only: {exc}")
            image_messages = []
            if text_only_prompt is not None:
                prompt = text_only_prompt
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            generated_text = self._generate_text(messages)

        parsed = self._try_json_decode(generated_text)
        if parsed is not None:
            if not image_messages and text_only_prompt is not None:
                parsed["visual_tags"] = {}
            return parsed

        # Retry with stricter formatting instruction to reduce JSON parsing failures.
        for retry_idx in range(self.json_retry):
            strict_messages = [
                {
                    "role": "user",
                    "content": [
                        *image_messages,
                        {
                            "type": "text",
                            "text": (
                                prompt
                                + "\n\nIMPORTANT: Output exactly one valid JSON object only. "
                                + "Do not include markdown/code fences/comments/trailing text."
                            ),
                        },
                    ],
                }
            ]
            if not image_messages:
                strict_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    prompt
                                    + "\n\nIMPORTANT: Output exactly one valid JSON object only. "
                                    + "Do not include markdown/code fences/comments/trailing text."
                                ),
                            }
                        ],
                    }
                ]
            generated_text = self._generate_text(strict_messages, force_greedy=True)
            parsed = self._try_json_decode(generated_text)
            if parsed is not None:
                if not image_messages and text_only_prompt is not None:
                    parsed["visual_tags"] = {}
                return parsed

        raise ValueError(
            "Model output is not valid JSON after retries. "
            f"Last output (truncated): {generated_text[:2000]}"
        )

