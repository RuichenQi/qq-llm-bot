"""List OpenAI models available on your account; filter by family."""
from __future__ import annotations

import asyncio
import sys

import httpx
from dotenv import dotenv_values


async def main() -> int:
    v = dotenv_values("f:/qqbot/.env")
    key = v.get("OPENAI_API_KEY") or ""
    base = v.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not key:
        print("OPENAI_API_KEY missing in .env", file=sys.stderr)
        return 1

    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{base.rstrip('/')}/models",
                        headers={"Authorization": f"Bearer {key}"})
    if r.status_code >= 400:
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        return 1
    ids = sorted(m["id"] for m in r.json().get("data", []))

    print(f"-- account sees {len(ids)} models total --\n")

    interesting = {
        "image family": [x for x in ids if "image" in x or "dall-e" in x],
        "gpt-5 family": [x for x in ids if x.startswith("gpt-5")],
        "gpt-4o family (for comparison)": [x for x in ids if x.startswith("gpt-4o")],
    }
    for label, items in interesting.items():
        print(f"## {label} ({len(items)})")
        for x in items:
            print(f"  - {x}")
        print()

    # Targeted existence checks
    targets = ["gpt-image-1", "gpt-image-1-mini", "gpt-5.4-nano",
               "gpt-5-nano", "gpt-5-mini", "gpt-5"]
    print("## existence check for asked-about names")
    for t in targets:
        print(f"  {'✓' if t in ids else '✗'} {t}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
