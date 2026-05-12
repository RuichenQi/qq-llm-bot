"""Compare gpt-5.4-nano vs gpt-4o-mini on the same prompt, and verify that
gpt-image-1-mini low actually works.

Just runs the calls and reports tokens / wall-time / cost. No assertions —
the user reads the numbers and decides.
"""
from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import httpx
from dotenv import dotenv_values

# Prices in USD per 1M tokens (input, output). Verified via web search May 2026.
TEXT_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini":   (0.15, 0.60),
    "gpt-5-nano":    (0.05, 0.40),
    "gpt-5.4-nano":  (0.20, 1.25),
    "gpt-5.4-mini":  (0.25, 2.00),
}

# Single-image cost (USD) at the listed size/quality
IMAGE_PRICES = {
    ("dall-e-2", "256x256", "standard"):       0.016,
    ("dall-e-2", "1024x1024", "standard"):     0.020,
    ("gpt-image-1", "1024x1024", "low"):       0.011,
    ("gpt-image-1-mini", "1024x1024", "low"):  0.005,
    ("gpt-image-1-mini", "1024x1024", "medium"): 0.018,
}


PROMPT = "用三句话解释什么是 quantum tunneling，要通俗。"


async def chat(client: httpx.AsyncClient, base: str, key: str, model: str):
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    # gpt-5.x family uses max_completion_tokens; gpt-4o/o1 use max_tokens.
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = 400
    else:
        body["max_tokens"] = 400
        body["temperature"] = 0.3
    t0 = time.time()
    r = await client.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
    )
    dt = time.time() - t0
    if r.status_code >= 400:
        return None, dt, f"HTTP {r.status_code}: {r.text[:200]}"
    d = r.json()
    text = d["choices"][0]["message"]["content"].strip()
    usage = d.get("usage", {})
    return (text, usage), dt, None


async def generate_image(client: httpx.AsyncClient, base: str, key: str,
                         model: str, size: str, quality: str | None):
    body: dict = {"model": model, "prompt": "a tiny shiba inu icon",
                  "n": 1, "size": size}
    if quality is not None:
        body["quality"] = quality
    if model.startswith("dall-e"):
        body["response_format"] = "b64_json"
    t0 = time.time()
    r = await client.post(
        f"{base}/images/generations",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
    )
    dt = time.time() - t0
    if r.status_code >= 400:
        return None, dt, f"HTTP {r.status_code}: {r.text[:300]}"
    d = r.json()
    item = d["data"][0]
    raw = base64.b64decode(item["b64_json"]) if item.get("b64_json") else None
    return raw, dt, None


def cost_text(model: str, usage: dict) -> float:
    if model not in TEXT_PRICES:
        return float("nan")
    pin, pout = TEXT_PRICES[model]
    return (usage.get("prompt_tokens", 0) * pin
            + usage.get("completion_tokens", 0) * pout) / 1_000_000


async def main() -> int:
    v = dotenv_values("f:/qqbot/.env")
    key = v["OPENAI_API_KEY"]
    base = (v.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    async with httpx.AsyncClient(timeout=120.0) as c:
        print("========== TEXT: same prompt, four models ==========")
        print(f"prompt: {PROMPT}\n")
        for model in ("gpt-4o-mini", "gpt-5-nano", "gpt-5.4-nano", "gpt-5.4-mini"):
            print(f"--- {model} ---")
            res, dt, err = await chat(c, base, key, model)
            if err:
                print(f"  ERROR: {err}\n")
                continue
            text, usage = res
            cost = cost_text(model, usage)
            print(f"  time: {dt:.2f}s  in_tokens: {usage.get('prompt_tokens')}"
                  f"  out_tokens: {usage.get('completion_tokens')}"
                  f"  cost: ${cost:.6f}")
            print(f"  output: {text[:200]}")
            print()

        print("========== IMAGE: dall-e-2 vs gpt-image-1-mini low ==========\n")
        out_dir = Path("f:/qqbot/data")
        out_dir.mkdir(parents=True, exist_ok=True)
        for model, size, quality in [
            ("dall-e-2", "256x256", None),
            ("gpt-image-1-mini", "1024x1024", "low"),
        ]:
            label = f"{model}@{size}/{quality or 'std'}"
            print(f"--- {label} ---")
            raw, dt, err = await generate_image(c, base, key, model, size, quality)
            if err:
                print(f"  ERROR: {err}\n")
                continue
            price = IMAGE_PRICES.get((model, size, quality or "standard"), float("nan"))
            out = out_dir / f"cmp_{model}_{size}_{quality or 'std'}.png".replace(":", "_")
            out.write_bytes(raw or b"")
            print(f"  time: {dt:.2f}s  bytes: {len(raw or b'')}  cost: ${price:.4f}  → {out.name}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
