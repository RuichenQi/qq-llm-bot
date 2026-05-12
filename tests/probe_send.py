"""Bypass the bot and ask NapCat directly to send a test message.

Usage:
    python -m tests.probe_send <group_id> "test message"
"""
import asyncio
import json
import sys
import uuid

import websockets


async def main():
    gid = int(sys.argv[1])
    text = sys.argv[2] if len(sys.argv) > 2 else "probe from f:\\qqbot test"
    echo = uuid.uuid4().hex

    print(f"connecting to ws://127.0.0.1:3001 ...")
    async with websockets.connect("ws://127.0.0.1:3001",
                                   ping_interval=10, ping_timeout=10) as ws:
        req = {"action": "send_group_msg",
               "params": {"group_id": gid, "message": text,
                          "auto_escape": False},
               "echo": echo}
        await ws.send(json.dumps(req))
        print(f"sent: {req}")

        deadline = asyncio.get_event_loop().time() + 10
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(raw)
            if data.get("echo") == echo:
                print(f"\n=== NapCat response ===")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                return
            else:
                # ignore unrelated frames (events, heartbeats from another channel)
                pass


asyncio.run(main())
