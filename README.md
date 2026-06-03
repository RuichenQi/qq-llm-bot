# qq-llm-bot

A lightweight **QQ group LLM bot** that runs on an old Android phone via Termux.

- **OneBot v11 WebSocket** (NapCatQQ / Lagrange) for QQ I/O
- **DeepSeek as the default** text model *and* the LLM router (cheap)
- **OpenAI on demand** for premium text, vision, image generation, and image editing
- Async Python, single-command launch, JSON-file persistence (no DB)

## Architecture

```
QQ group message
  → OneBot WS adapter (NapCat / Lagrange)
  → bot.onebot_client        (asyncio websockets)
  → bot.message_parser       (CQ codes / array segments → ParsedMessage)
  → bot.command_handler      (allow-list, rule-based /commands, rate/quota gates)
  → bot.router               (DeepSeek strict-JSON router decides backend)
  → providers/*              (DeepSeek chat | OpenAI text/vision/image/edit)
  → bot.command_handler.send_*  (split long replies, base64:// for images)
  → QQ group
```

Persistence files (auto-created):

- `data/state.db` — SQLite (WAL): conversation memory, daily quotas, runtime
  group allow-list, daily-report bookkeeping. Survives `kill -9` on Termux.
- `data/images/<sha>.dat` — cached image bytes (TTL-swept) so `/edit` and
  `/vision` keep working after QQ's image URLs expire.
- `data/logs/bot.log` — daily rotating log.
- Legacy `memory.json` / `quota.json` are migrated on first start and renamed
  to `*.migrated` (you can delete those once you're happy).

## Quick start (Termux on Android)

```bash
# 1. Install Termux from F-Droid (Play Store version is unmaintained).
# 2. In Termux:
pkg update && pkg upgrade -y
pkg install -y python git rust binutils      # rust+binutils help build wheels on ARM
termux-setup-storage                          # optional, lets the bot read shared files

# 3. Get the OneBot adapter running first.
#    For NapCatQQ: see https://napneko.github.io/ (Android build).
#    Configure it to expose a forward-WebSocket endpoint, e.g. ws://127.0.0.1:3001
#    with an access token if you want one.

# 4. Clone & configure this project:
git clone <your-repo> qq-llm-bot
cd qq-llm-bot
cp .env.example .env
nano .env                                     # fill in API keys + ALLOWED_GROUPS

# 5. Run:
chmod +x run.sh
./run.sh
```

`run.sh` creates `.venv`, installs `requirements.txt`, and starts `main.py`.

To keep the bot alive across SSH/Termux sessions, run inside `tmux`/`termux-wake-lock`:

```bash
pkg install -y tmux
termux-wake-lock        # prevent Android from killing the process when screen locks
tmux new -s bot
./run.sh
# Ctrl+B then D to detach; `tmux attach -t bot` to come back
```

## Commands

| Command | Effect |
| --- | --- |
| `/ask <q>` | Default text chat (DeepSeek) |
| `/think <q>` | DeepSeek Reasoner (deeper reasoning) |
| `/gpt <q>` | Force OpenAI text — quota-limited |
| `/image <prompt>` | OpenAI image generation — strict quota |
| `/vision <q>` | Analyse the most recent image with OpenAI vision |
| `/edit <instruction>` | Edit the most recent image with OpenAI |
| `/file [q]` | Ask about an attached / quoted file (txt/pdf/docx/code/audio/video) |
| `/search <q>` | Force a web search + summary (Tavily) |
| `/news [topic]` | Pull current headlines into a short paragraph (Tavily) |
| `/teach <rule>` | Pin a verbatim rule into this group's system prompt |
| `/remember` | List what the bot has saved for this group |
| `/forget <ids|all|rules\|...>` | Drop one / many / kind / all lessons |
| `/recap [period]` | Summarise recent group activity |
| `/recall [date\|kw]` | Query long-term daily recaps |
| `/timewarp [period]` | Bot writes a short nostalgic riff about that time |
| `/start` / `/stop` | Bring the bot back / silence it in this group (super-user) |
| `/clear` | Clear *your* memory in this group |
| `/balance` | Show today's usage vs. limits |
| `/help` | List commands |

Messages **without** a `/command` are first checked by rule-based routing, then
delegated to the DeepSeek router which returns strict JSON like
`{"route": "deepseek_chat", "confidence": 0.9, "reason": "...", "normalized_prompt": "..."}`.

## Config (`.env`)

See [.env.example](./.env.example) for the full list. The minimum:

```
DEEPSEEK_API_KEY=sk-deepseek-xxx
OPENAI_API_KEY=sk-openai-xxx           # optional; bot still runs without it
ONEBOT_WS_URL=ws://127.0.0.1:3001
ONEBOT_ACCESS_TOKEN=                   # match what the adapter expects
ALLOWED_GROUPS=123456789,987654321
```

Quota defaults (override per-env if needed):

| Route | Per group / day | Per user / day |
| --- | --- | --- |
| `openai_text` | 20 | 3 |
| `openai_image` | 10 | 2 |
| `openai_image_edit` | 5 | 1 |
| `openai_vision` | 20 | 5 |

When a user hits a quota wall they receive: `今天这个功能的额度用完了，请明天再试吧~`

## Testing without a real QQ

Two layers of offline testing.

**Live offline harness** (talks to real DeepSeek, no QQ needed):

```bash
python -m tests.fake_event "/ask 你好"
python -m tests.fake_event "帮我画一只柴犬"        # routes to openai_image
python -m tests.fake_event "/help"
python -m tests.fake_event "/edit 加点雪" --image https://example.com/cat.png
```

**Pure unit tests** (no API keys needed):

```bash
pip install pytest
PYTHONPATH=. pytest tests -q
```

The suite covers the OneBot parser (array + CQ-string), router JSON coercion,
memory/quota persistence, and the trigger-mode gate.

## Adapter modes

Set `ONEBOT_MODE=forward` (default) or `ONEBOT_MODE=reverse` in `.env`.

- **forward**: bot dials `ONEBOT_WS_URL`. Configure your adapter to expose a
  forward WebSocket and put that URL in `.env`.
- **reverse**: bot runs `ws://0.0.0.0:3001/onebot/v11/ws`. Configure your
  adapter's "reverse WebSocket" / "WebSocket client" setting to point at it.
  Token-based auth (`ONEBOT_ACCESS_TOKEN`) is checked on the way in.

## Trigger modes

In a noisy group you usually don't want the bot answering everything. Set
`TRIGGER_MODE` in `.env`:

| Mode | Effect |
| --- | --- |
| `always` | Reply to every group message (commands work too) |
| `mention` | Only reply when the bot is `@`ed |
| `prefix` | Only reply when the message starts with `TRIGGER_PREFIX` (default `#`) |

`/commands` always bypass the gate.

## Admin commands

Add your QQ id to `SUPERUSERS=...` in `.env` and use `/admin`:

| Command | Effect |
| --- | --- |
| `/admin status` | This group's per-route usage today |
| `/admin usage` | Today's usage across every group |
| `/admin reset_quota` | Clear today's quota across all groups/users |
| `/admin clear <uid\|@user>` | Drop one user's conversation memory in this group |
| `/admin reset confirm` | Wipe everything in this group (memory + group log + recaps + lessons); preserves allow-list and pause state |
| `/admin allow_group <gid>` | Add a group to the runtime allow-list |
| `/admin disallow_group <gid>` | Remove a group (env-pinned groups can't be removed) |
| `/admin list_groups` | Show every allowed group (`*` = pinned in `.env`, `⏸` = `/stop`-paused) |
| `/admin report` | Push today's daily-report right now |
| `/admin ping` | OneBot WS health: connected? last event? last heartbeat? reconnects? |
| `/admin save_recap [day]` | Force-write a daily recap for today / yesterday / `YYYY-MM-DD` |

## Streaming + daily report + reply quotes

- **Streaming** — DeepSeek replies arrive paragraph-by-paragraph (toggle with
  `STREAM_REPLIES=0`). Flush threshold is `STREAM_FLUSH_CHARS`.
- **Daily report** — set `DAILY_REPORT_GROUP=<gid>` and `DAILY_REPORT_TIME=HH:MM`
  and the bot posts a usage summary to that group at the configured time. A
  SQLite row prevents duplicate sends if the process restarts within the day.
  Run `/admin report` to fire one immediately.
- **Reply quotes** — if a user uses QQ's "reply" feature, the quoted message
  text is fetched via OneBot `get_msg` and prepended to the prompt as
  `[被引用的消息] ... [我的问题] ...`, so follow-up questions ("更详细一点",
  "再改一下") have context.
- **`/admin ping`** — for diagnosing reverse-WS flakiness. Shows mode,
  connection state, last received event, last heartbeat (OneBot
  `meta_event/heartbeat`), cumulative disconnect count, and the reason of the
  last disconnect. Especially useful on Termux where the OS may kill the
  adapter behind the bot's back.

## Notes / next steps

- Provider classes are intentionally isolated — drop a `providers/gemini.py`
  next to the others and register it in `command_handler`.
- Memory + quota are JSON files; swap for SQLite later by reimplementing
  `Memory` / `Quota`.
- API keys are redacted in logs (any `sk-...` token is replaced with `sk-***`).
- Long replies are auto-split by `MAX_REPLY_CHARS` (default 1800 chars).
