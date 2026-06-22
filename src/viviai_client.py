"""ViviAI multimodal API client (OpenAI-compatible).

Wraps gemini-3-pro-preview / gpt-5.5 / gpt-image-2 etc. via the viviai gateway.

Usage:
    from viviai_client import ViviClient

    cli = ViviClient(api_key=...)
    text = cli.chat_text("gemini-3-pro-preview", "say hi", temp=0.5)
    score = cli.chat_vision("gemini-3-pro-preview", "is this image fake?", image_path="...")
    img_b64 = cli.gen_image("gpt-image-2", "a portrait of an astronaut")
"""
from __future__ import annotations
import os
import base64
import json
import time
from pathlib import Path
from typing import Optional, Union

import requests


BASE_URL = "https://api.viviai.cc"
DEFAULT_TIMEOUT = 90


def _encode_image(image: Union[str, Path, bytes]) -> str:
    """Return base64 of image (PNG/JPG)."""
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return base64.b64encode(image).decode()


class ViviClient:
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = BASE_URL,
                 timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key or os.environ.get("VIVIAI_KEY")
        if not self.api_key:
            raise ValueError("Set VIVIAI_KEY env or pass api_key=...")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sess = requests.Session()
        self._sess.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    # ----------------------- core -------------------------------
    def chat(self, model: str, messages: list, temperature: float = 0.1,
             max_tokens: int = 800, retry: int = 5, **kw) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens, **kw}
        last_err = None
        for attempt in range(retry):
            try:
                r = self._sess.post(url, json=payload, timeout=self.timeout)
                # 显式 retry on 5xx (BUG #7 修: 503 是 transient server error)
                if r.status_code in (502, 503, 504):
                    last_err = RuntimeError(f"server {r.status_code}")
                    time.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"chat call failed after {retry}: {last_err}")

    # ----------------------- helpers ----------------------------
    def chat_text(self, model: str, prompt: str, system: Optional[str] = None,
                  temperature: float = 0.1, max_tokens: int = 800) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self.chat(model, msgs, temperature=temperature, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"]

    def chat_vision(self, model: str, prompt: str,
                    image: Union[str, Path, bytes],
                    image_mime: str = "image/png",
                    temperature: float = 0.1, max_tokens: int = 1200) -> str:
        """Multimodal chat with one image attached."""
        b64 = _encode_image(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_mime};base64,{b64}"}}
            ]
        }]
        resp = self.chat(model, messages, temperature=temperature, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"]

    def chat_vision_json(self, model: str, prompt: str, image, **kw) -> dict:
        """Same as chat_vision but parse JSON from response (★ Q18: robust)."""
        text = self.chat_vision(model, prompt, image, **kw)
        try:
            from robustness import parse_json_robust
            return parse_json_robust(text)
        except Exception:
            # last-resort fallback
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            s, e = text.find("{"), text.rfind("}")
            if s >= 0 and e > s:
                return json.loads(text[s:e+1])
            raise

    def gen_image(self, model: str, prompt: str, n: int = 1,
                  size: str = "1024x1024", retry: int = 4, **kw) -> list:
        """Returns list of base64 image strings (or URLs).
        显式 retry on 5xx (BUG #7 修: nano_banana_pro 503 频繁).
        """
        url = f"{self.base_url}/v1/images/generations"
        payload = {"model": model, "prompt": prompt, "n": n, "size": size, **kw}
        last_err = None
        for attempt in range(retry):
            try:
                r = self._sess.post(url, json=payload, timeout=self.timeout * 2)
                if r.status_code in (502, 503, 504):
                    last_err = RuntimeError(f"server {r.status_code}")
                    time.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
                data = r.json().get("data", [])
                return [d.get("b64_json") or d.get("url") for d in data]
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"gen_image failed after {retry}: {last_err}")


# ---------------------- smoke test --------------------------------
if __name__ == "__main__":
    cli = ViviClient(api_key=os.environ.get("VIVIAI_KEY", ""))
    print("=== chat_text test ===")
    out = cli.chat_text("gemini-2.5-flash",
                        "Say only 'pong' and nothing else.",
                        temperature=0.0, max_tokens=10)
    print(f"  response: {out!r}")

    print("\n=== chat_vision test (need a real image) ===")
    sample = "/data/disk4/lyx_ICML/hf_models_lyx/04_id_preserving/InstantX__InstantID/examples/0.png"
    if Path(sample).exists():
        out = cli.chat_vision_json(
            "gemini-3-pro-preview",
            'Reply strictly as JSON: {"description": "<one sentence>"}.',
            sample, temperature=0.1, max_tokens=100
        )
        print(f"  vision JSON: {out}")
