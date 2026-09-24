import os
from dotenv import load_dotenv

load_dotenv()

class VoiceConfig:
    WS_PORT = int(os.getenv("VOICE_WS_PORT", "8766"))
    WS_HOST = os.getenv("VOICE_WS_HOST", "0.0.0.0")
    
    # Audio settings (must match frontend client)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit
    
    # TEN VAD settings. These match ten_vad_python/config.py.
    VAD_HOP_SIZE_MS = 16
    VAD_PREFIX_PADDING_MS = 240
    VAD_SILENCE_DURATION_MS = 1000
    VAD_THRESHOLD = 0.7
    
    # API URLs
    RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1/chat/text")
