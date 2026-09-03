# Voice AI — How to Run

You need **3 terminals** open at the same time.

---

## Terminal 1 — Backend (Docker)

```
cd F:\1. rubixe ai\nova-assamese\nova-voice-bot
docker compose -f docker-compose.windows.yml up -d
```

Check it's running:
```
docker ps
```
You should see `nova-voice-bot-api-1` up on port 8000.

---

## Terminal 2 — Voice WebSocket Server

```
cd F:\1. rubixe ai\nova-assamese\nova-voice-bot
python voice_ws_server.py
```

You should see:
```
Voice WebSocket server listening on ws://0.0.0.0:8766
```

---

## Terminal 3 — Voice Client (HTML)

```
cd F:\1. rubixe ai\nova-assamese\nova-voice-bot\app\static
python -m http.server 8080
```

---

## Open in browser

```
http://localhost:8080/voice_client.html
```

Click **Connect + Start Mic**, allow microphone access, pick a language from the dropdown, and start talking.

---

