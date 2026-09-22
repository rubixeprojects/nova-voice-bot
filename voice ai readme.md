

Voice ai readme · MD
# Voice AI — How to Run
 
Everything now runs in **one container stack**. You no longer need 3 terminals —
the voice WebSocket server and the HTML test client are part of the stack.
 
> **TL;DR (project already set up):**
> ```
> cd <path-to>\nova-voice-bot
> podman-compose -f docker-compose.windows-cpu.yml up -d --no-build
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
 
Leave `BGE_M3_FORCE_LOCAL=0` unless you specifically want embeddings to run
locally on CPU instead of via Hugging Face's Inference API (then set it to
`1` — but note this is much slower and can peg the CPU on low-resource
machines).
 
### 2c. Build the images
 
**Normal (GPU-capable) build:**
```
podman-compose -f docker-compose.windows.yml build
```
 
**CPU-friendly build:**
```
podman-compose -f docker-compose.windows-cpu.yml build
```
 
First build downloads PyTorch + PaddleOCR etc. — **15–40 min** depending on
network. Later builds are cached.
 
---
 
## 3. Run it (every time)
 
**One terminal:**
```
cd <path-to>\nova-voice-bot
```
 
**Normal (GPU-capable):**
```
podman-compose -f docker-compose.windows.yml up -d
```
 
**CPU-friendly (recommended on most Windows/laptop setups):**
```
podman-compose -f docker-compose.windows-cpu.yml up -d --no-build
```
Drop `--no-build` the first time, or any time you've changed a Dockerfile —
it forces a fresh build instead of reusing existing images.
 
- Database migrations run automatically (the `migrate` container runs once and
  exits `0` — that's normal).
- The **API takes ~1–2 min on first start** to load the local embedding model.
  It's ready when `podman logs updated-nova-voicebot_api_1` shows
  `Application startup complete`.
Check everything is up:
```
podman ps
```
You should see these running:
 
| Container | Port | Role |
|-----------|------|------|
| `updated-nova-voicebot_postgres_1` | 5432 | database |
| `updated-nova-voicebot_qdrant_1` | 6333 | vector search |
| `updated-nova-voicebot_opensearch_1` | 9200 | keyword search |
| `updated-nova-voicebot_redis_1` | 6379 | task queue |
| `updated-nova-voicebot_api_1` | **8000** | REST API + `/docs` + `/voice` |
| `updated-nova-voicebot_worker_1` | — | document ingestion |
| `updated-nova-voicebot_voice_1` | **8766** | voice WebSocket server |
| `updated-nova-voicebot_ui_1` | 8503 | Streamlit UI (optional) |
 
(`updated-nova-voicebot_migrate_1` will show as `Exited (0)` — correct.)
 
> If `podman ps` ever shows a mix of hyphenated (`updated-nova-voicebot-api-1`)
> and underscored (`updated-nova-voicebot_api_1`) container names at the same
> time, that's a stale/duplicate stack from a mismatched project name — stop
> and remove the hyphenated set and keep only the underscored one, which
> matches this repo's compose files.
 
If a container was stopped rather than fully removed, `up -d` may fail to
restart it (e.g. `given PID did not die within timeout`). In that case:
```
podman start updated-nova-voicebot_api_1
```
(swap in whichever container name failed).
 
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
- **Documents** → upload / list / delete PDF, DOCX, TXT, and MD files from the
  same page
Other URLs:
- API docs (Swagger): http://localhost:8000/docs
- Admin panel: http://localhost:8000/admin
- Qdrant dashboard: http://localhost:6333/dashboard
- Streamlit UI (alternative front-end): http://localhost:8503
---
 
## 5. Stop it
 
**Normal:**
```
podman-compose -f docker-compose.windows.yml down
```
 
**CPU-friendly:**
```
podman-compose -f docker-compose.windows-cpu.yml down
```
 
Add `-v` to also wipe the database / indexes (fresh start next time).
 
---
 
## What changed from the old 3-terminal setup
 
| Old way | Now |
|---------|-----|
| Terminal 1: `docker compose ... up -d` | Same, but `podman-compose` — and it also starts the voice server |
| Terminal 2: `python voice_ws_server.py` on the host | **Gone** — runs as the `voice` container automatically |
| Terminal 3: `python -m http.server 8080` for the HTML | **Gone** — the API serves it at `/voice` |
| Open `http://localhost:8080/voice_client.html` | Open `http://localhost:8000/voice` |
| `.env` had no `UNIVERSAL_USER_ID` | Now **required** in `.env` (see 2b) |
| `requirements.txt` missing `websockets` / `aiohttp` | Added — voice server deps |
| Embeddings always via Hugging Face cloud | `BGE_M3_FORCE_LOCAL=0` tries HF's API first, falls back to local CPU only if it fails |
| Only PDF uploads supported | DOCX, TXT, and MD uploads also supported |
 
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
| API returns HTTP 500, Postgres restarted | VM out of RAM, or `BGE_M3_FORCE_LOCAL=1` overloading CPU. Raise `.wslconfig` memory, confirm `BGE_M3_FORCE_LOCAL=0`. |
| Build fails with `Input/output error` | Disk full — move Podman storage off `C:` (see above). |
| `pydantic ... universal_user_id Field required` | `UNIVERSAL_USER_ID` missing from `.env`. |
| Voice mic connects then drops | Check `podman logs updated-nova-voicebot_voice_1`; make sure the `api` container is healthy first. |
| `rootlessport listen tcp4 ... bind: address already in use` | An old container (often a stale hyphenated-name duplicate) is still holding that port. `podman ps -a`, then `podman stop`/`podman rm` the stale one. |
| `podman ps` shows nothing for containers you know are running | You're likely pointed at the wrong compose file/project name. Run `podman-compose -f <your-file>.yml config \| findstr /i "name"` to check, and always pass the same `-f <file>.yml` consistently. |
 
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
 
