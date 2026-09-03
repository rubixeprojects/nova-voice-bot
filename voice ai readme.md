# Voice AI — How to Run

Everything now runs in **one container stack**. You no longer need 3 terminals —
the voice WebSocket server and the HTML test client are part of the stack.

> **TL;DR (project already set up):**
> ```
> cd <path-to>\nova-voice-bot
> podman compose -f docker-compose.windows.yml up -d
> ```
> Wait ~2 min, then open **http://localhost:8000/voice**

---

## 1. Prerequisites (once per computer)

- **Podman Desktop** installed and the Podman machine started
  (`podman machine start`). Docker Desktop also works — replace `podman` with
  `docker` in every command below.
- A compose provider. Test with `podman compose version`.
  If it says *"no compose provider"*, install the Python one:
  ```
  pip install podman-compose
  ```
  then use `podman-compose` instead of `podman compose` everywhere below
  (identical arguments).
- **~6 GB RAM free** for the containers. On an 8 GB machine see
  [Low-RAM setup](#low-ram--windows-wsl-setup).

---

## 2. First-time setup (once per clone)

### 2a. Get the code
```
git clone <repo-url>
cd <path-to>\nova-voice-bot
```

### 2b. Create the `.env` file
```
copy .env.example .env
```
Then edit `.env` (path: `<path-to>\nova-voice-bot\.env`) and set:

| Key | What to put |
|-----|-------------|
| `SARVAM_API_KEY` | your Sarvam AI key (STT / TTS / LLM) |
| `HF_TOKEN` | your Hugging Face token (used by the reranker) |
| `UNIVERSAL_USER_ID` | **required** — any valid UUID. Generate one: `python -c "import uuid; print(uuid.uuid4())"` |

Leave `BGE_M3_FORCE_LOCAL=1` unless you specifically want embeddings to run on
Hugging Face's cloud instead of locally (then set it to `0`).

### 2c. Build the images
```
podman compose -f docker-compose.windows.yml build
```
First build downloads PyTorch + PaddleOCR etc. — **15–40 min** depending on
network. Later builds are cached.

---

## 3. Run it (every time)

**One terminal:**
```
cd <path-to>\nova-voice-bot
podman compose -f docker-compose.windows.yml up -d
```

- Database migrations run automatically (the `migrate` container runs once and
  exits `0` — that's normal).
- The **API takes ~1–2 min on first start** to load the local embedding model.
  It's ready when `podman logs nova-voice-bot_api_1` shows
  `Application startup complete`.

Check everything is up:
```
podman ps
```
You should see these running:

| Container | Port | Role |
|-----------|------|------|
| `nova-voice-bot_postgres_1` | 5432 | database |
| `nova-voice-bot_qdrant_1` | 6333 | vector search |
| `nova-voice-bot_opensearch_1` | 9200 | keyword search |
| `nova-voice-bot_redis_1` | 6379 | task queue |
| `nova-voice-bot_api_1` | **8000** | REST API + `/docs` + `/voice` |
| `nova-voice-bot_worker_1` | — | document ingestion |
| `nova-voice-bot_voice_1` | **8766** | voice WebSocket server |
| `nova-voice-bot_ui_1` | 8503 | Streamlit UI (optional) |

(`nova-voice-bot_migrate_1` will show as `Exited (0)` — correct.)

---

## 4. Use it

Open in your browser:

```
http://localhost:8000/voice
```

This one page does **both text chat and voice**:
- **Text chat** → calls the API at `http://localhost:8000`
- **Voice** → click *Connect + Start Mic*, allow microphone, pick a language,
  start talking (connects to `ws://localhost:8766`)
- **Documents** → upload / list / delete PDFs from the same page

Other URLs:
- API docs (Swagger): http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard
- Streamlit UI (alternative front-end): http://localhost:8503

---

## 5. Stop it

```
podman compose -f docker-compose.windows.yml down
```
Add `-v` to also wipe the database / indexes (fresh start next time).

---

## What changed from the old 3-terminal setup

| Old way | Now |
|---------|-----|
| Terminal 1: `docker compose ... up -d` | Same, but `podman compose` — and it also starts the voice server |
| Terminal 2: `python voice_ws_server.py` on the host | **Gone** — runs as the `voice` container automatically |
| Terminal 3: `python -m http.server 8080` for the HTML | **Gone** — the API serves it at `/voice` |
| Open `http://localhost:8080/voice_client.html` | Open `http://localhost:8000/voice` |
| `.env` had no `UNIVERSAL_USER_ID` | Now **required** in `.env` (see 2b) |
| `requirements.txt` missing `websockets` / `aiohttp` | Added — voice server deps |
| Embeddings always via Hugging Face cloud | `BGE_M3_FORCE_LOCAL=1` runs them locally (HF's free tier is unreliable) |

---

## Low-RAM / Windows WSL setup

The stack needs ~6 GB. On an 8 GB laptop, give the Podman/WSL VM enough memory —
create `C:\Users\<you>\.wslconfig`:

```
[wsl2]
memory=6GB
swap=10GB
processors=12
```
Then `wsl --shutdown` and `podman machine start` again.

If your `C:` drive is low on space, move the Podman VM disk to another drive:
```
wsl --shutdown
wsl --export podman-machine-default D:\podman-machine.tar
wsl --unregister podman-machine-default
wsl --import podman-machine-default D:\podman-machine D:\podman-machine.tar --version 2
del D:\podman-machine.tar
podman machine start
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `no compose provider` | `pip install podman-compose`, use `podman-compose` |
| First chat request takes ~40 s | Normal — local embedding model warming up. Fast after that. |
| API returns HTTP 500, Postgres restarted | VM out of RAM. Raise `.wslconfig` memory, confirm `BGE_M3_FORCE_LOCAL=1`. |
| Build fails with `Input/output error` | Disk full — move Podman storage off `C:` (see above). |
| `pydantic ... universal_user_id Field required` | `UNIVERSAL_USER_ID` missing from `.env`. |
| Voice mic connects then drops | Check `podman logs nova-voice-bot_voice_1`; make sure the `api` container is healthy first. |

---

## Optional: run the voice server / client on the host (old way)

Only if you don't want them in containers. Needs the deps installed in a venv
(`ten-vad` has no wheel on Python 3.14 — use 3.11):

```
cd <path-to>\nova-voice-bot
pip install -r requirements.txt
python voice_ws_server.py            # terminal A — ws://0.0.0.0:8766

cd app\static
python -m http.server 8080           # terminal B
```
Then open `http://localhost:8080/voice_client.html`.
(The API's CORS already allows `http://localhost:8080`.)
