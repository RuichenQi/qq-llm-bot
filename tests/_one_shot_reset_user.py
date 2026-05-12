"""One-shot: clear a user's per-(group, user) chat memory in all groups."""
import asyncio
import sys
from bot.storage import Storage


async def main():
    user_id = int(sys.argv[1])
    store = await Storage.get()
    assert store._conn is not None
    async with store._lock:
        cur = await store._conn.execute(
            "DELETE FROM memory WHERE user_id=?", (user_id,)
        )
        deleted = cur.rowcount or 0
        await store._conn.commit()
    print(f"cleared {deleted} memory rows for user {user_id}")
    await store.close()


asyncio.run(main())
