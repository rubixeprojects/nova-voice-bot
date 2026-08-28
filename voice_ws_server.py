import asyncio
import json
import logging
import wave
import io
import time
import uuid
import base64
import re
import os

import websockets
import aiohttp
import numpy as np
from ten_vad import TenVad
from typing import Optional

from voice_config import VoiceConfig
from app.llm.sarvam_client import text_to_speech, transcribe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("voice_ws_server")

# VAD class
class VADDetector:
    def __init__(self):
        self.sample_rate = VoiceConfig.SAMPLE_RATE
        self.hop_size = self.sample_rate * VoiceConfig.VAD_HOP_SIZE_MS // 1000
        self.bytes_per_hop = self.hop_size * VoiceConfig.SAMPLE_WIDTH
        self.prefix_window_size = (
            VoiceConfig.VAD_PREFIX_PADDING_MS // VoiceConfig.VAD_HOP_SIZE_MS
        )
        self.silence_window_size = (
            VoiceConfig.VAD_SILENCE_DURATION_MS // VoiceConfig.VAD_HOP_SIZE_MS
        )
        self.window_size = max(self.prefix_window_size, self.silence_window_size)
        self.vad = TenVad(self.hop_size)
        self.audio_buffer = bytearray()
        self.probe_window: list[float] = []
        self.recent_audio: list[bytes] = []
        self.speech_buffer = bytearray()
        self.is_speaking = False

    def process_chunk(self, chunk: bytes):
        """Apply TEN's probability-window VAD and return completed utterances."""
        self.audio_buffer.extend(chunk)
        segments = []

        while len(self.audio_buffer) >= self.bytes_per_hop:
            audio_hop = bytes(self.audio_buffer[:self.bytes_per_hop])
            del self.audio_buffer[:self.bytes_per_hop]
            probe, _flag = self.vad.process(
                np.frombuffer(audio_hop, dtype=np.int16)
            )
            self.probe_window.append(probe)
            if len(self.probe_window) > self.window_size:
                self.probe_window.pop(0)

            self.recent_audio.append(audio_hop)
            if len(self.recent_audio) > self.prefix_window_size:
                self.recent_audio.pop(0)

            if not self.is_speaking:
                if len(self.probe_window) == self.window_size:
                    prefix_probes = self.probe_window[-self.prefix_window_size:]
                    if all(probe >= VoiceConfig.VAD_THRESHOLD for probe in prefix_probes):
                        self.is_speaking = True
                        self.speech_buffer.extend(b"".join(self.recent_audio))
            elif len(self.probe_window) == self.window_size:
                self.speech_buffer.extend(audio_hop)

                # Safety valve: never let a segment buffer forever in a noisy
                # room where a proper silence gap may never occur.
                max_speech_bytes = self.sample_rate * VoiceConfig.SAMPLE_WIDTH * 8  # 8s cap
                if len(self.speech_buffer) >= max_speech_bytes:
                    segments.append(bytes(self.speech_buffer))
                    self.speech_buffer.clear()
                    self.is_speaking = False
                    continue

                silence_probes = self.probe_window[-self.silence_window_size:]
                if all(probe < VoiceConfig.VAD_THRESHOLD for probe in silence_probes):
                    self.is_speaking = False
                    if len(self.speech_buffer) >= self.sample_rate * VoiceConfig.SAMPLE_WIDTH // 2:
                        segments.append(bytes(self.speech_buffer))
                    self.speech_buffer.clear()

        return segments

    @property
    def has_confirmed_barge_in(self):
        return self.is_speaking
    def reset(self):
        """Clear internal speech-detection state. Call at the moment the bot
        starts speaking, so background noise picked up during the earlier
        PROCESSING wait can't be misread as a fresh barge-in."""
        self.audio_buffer.clear()
        self.probe_window.clear()
        self.recent_audio.clear()
        self.speech_buffer.clear()
        self.is_speaking = False
    
class AmplitudeGate:
    """
    Tracks an adaptive ambient noise floor and decides whether a given audio
    segment is loud enough above it to count as genuine close-mic speech —
    filters out background chatter, keyboard clicks, media playing nearby.
    """
    def __init__(self, margin_db: float = 12.0, floor_alpha: float = 0.02):
        self.noise_floor_db = -60.0  # starting assumption: reasonably quiet room
        self.margin_db = margin_db
        self.floor_alpha = floor_alpha

    @staticmethod
    def _rms_db(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return -100.0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return -100.0
        rms = np.sqrt(np.mean(samples ** 2))
        if rms < 1e-6:
            return -100.0
        return 20 * np.log10(rms / 32768.0)

    def update_noise_floor(self, pcm_bytes: bytes):
        """Call only on audio known NOT to be active user speech."""
        db = self._rms_db(pcm_bytes)
        self.noise_floor_db = (
            (1 - self.floor_alpha) * self.noise_floor_db + self.floor_alpha * db
        )

    def is_close_talking(self, pcm_bytes: bytes, margin_db: float = None) -> bool:
        if margin_db is None:
            margin_db = self.margin_db
        db = self._rms_db(pcm_bytes)
        return db > (self.noise_floor_db + margin_db)
    def force_seed_floor(self, pcm_bytes: bytes, alpha: float = 0.3):
        """Fast-converge the noise floor during initial calibration,
        regardless of VAD state."""
        db = self._rms_db(pcm_bytes)
        self.noise_floor_db = (1 - alpha) * self.noise_floor_db + alpha * db

def pcm_to_wav(pcm_data: bytes) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(VoiceConfig.CHANNELS)
            wav_file.setsampwidth(VoiceConfig.SAMPLE_WIDTH)
            wav_file.setframerate(VoiceConfig.SAMPLE_RATE)
            wav_file.writeframes(pcm_data)
        return wav_io.getvalue()
    
def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Read a WAV file's actual playback duration, so we can size the
    playback-wait timeout to the real answer length instead of a flat guess."""
    try:
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0
        
def split_into_sentences(text: str) -> list[str]:
    # Simple split on punctuations
    sentences = re.split(r'(?<=[.!?।।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def clean_answer(text: str) -> str:
    """Remove source markers before displaying or speaking the RAG answer."""
    if not text:
        return ""

    text = re.sub(
        r"\[\s*s\d+(?:\s*,\s*s\d+)*\s*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<!\w)s\d+(?!\w)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()

def strip_source_markers(text: str) -> str:
    """Lightweight citation-marker stripper for streamed text — same patterns
    as clean_answer(), safe to apply to partial/incremental text."""
    if not text:
        return ""
    text = re.sub(r"\[\s*s\d+(?:\s*,\s*s\d+)*\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)s\d+(?!\w)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\]", "", text)
    return text.strip()

class InterruptController:
    def __init__(self):
        self.turn_id = 0
        self.current_task: Optional[asyncio.Task] = None
        
    def new_turn(self):
        self.turn_id += 1
        self.cancel_current()
        return self.turn_id
        
    def cancel_current(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.current_task = None

class VoiceSession:
    def __init__(self, websocket):
        self.ws = websocket
        self.vad = VADDetector()
        self.interrupt_ctrl = InterruptController()
        self.amplitude_gate = AmplitudeGate(margin_db=12.0)
        self.state = "IDLE" # IDLE, LISTENING, PROCESSING, SPEAKING
        self.user_id = os.environ["UNIVERSAL_USER_ID"]
        self.request_id = str(uuid.uuid4())
        self.speaking_started_at = None
        self.calibration_start = time.time()
        self.calibration_duration = 1.5  # seconds
        self.calibrated = False
        self.playback_done_event = asyncio.Event()
        self.selected_language = "en_IN"

    async def send_state(self):
        await self.ws.send(json.dumps({"type": "state", "state": self.state}))

    async def handle_audio(self, audio_data: bytes):
        if not self.calibrated:
            self.amplitude_gate.force_seed_floor(audio_data)
            if time.time() - self.calibration_start >= self.calibration_duration:
                self.calibrated = True
            return
        if self.state == "IDLE":
            self.state = "LISTENING"
            await self.send_state()

        # Keep learning the room's ambient noise level whenever nobody is actively speaking.
        if self.state == "LISTENING" and not self.vad.is_speaking:
            self.amplitude_gate.update_noise_floor(audio_data)

        speech_segments = self.vad.process_chunk(audio_data)

        # Only interrupt active TTS if the incoming speech is meaningfully louder
        # than ambient background — i.e. actually close to the mic — and we're
        # past the initial grace period after the bot started speaking.
        if (self.state == "SPEAKING"
                and self.speaking_started_at
                and (time.time() - self.speaking_started_at) > 0.4
                and self.vad.has_confirmed_barge_in):
            if self.amplitude_gate.is_close_talking(bytes(self.vad.speech_buffer), margin_db=20.0):
                logger.info("Barge-in detected (confirmed close-talking)")
                self.interrupt_ctrl.new_turn()
                self.state = "LISTENING"
                await self.ws.send(json.dumps({"type": "cmd", "name": "stop_playback"}))
                await self.send_state()
            else:
                logger.debug("VAD triggered but too quiet vs. noise floor — ignoring as background")

        for segment in speech_segments:
            if self.state != "LISTENING":
                continue

            # Same near-field check for the initial utterance — filters phantom
            # transcriptions from background chatter/keyboard/media.
            if not self.amplitude_gate.is_close_talking(segment):
                logger.debug("Ignoring low-energy segment as background noise")
                continue

            turn_id = self.interrupt_ctrl.new_turn()
            self.state = "PROCESSING"
            # Unconditionally wipe any client-side audio the instant a new
            # utterance is confirmed — regardless of what state we thought
            # we were in. Guards against edge cases (e.g. a playback-wait
            # timeout) where the server believes playback ended but the
            # client is still actually playing it.
            await self.ws.send(json.dumps({"type": "cmd", "name": "stop_playback"}))
            await self.send_state()

            loop = asyncio.get_running_loop()
            task = loop.create_task(self.process_pipeline(segment, turn_id))
            self.interrupt_ctrl.current_task = task
            
    async def process_pipeline(self, pcm_data: bytes, expected_turn_id: int):
        try:
            logger.info(f"Turn {expected_turn_id}: Processing {len(pcm_data)} bytes of audio")
            # 1. STT
            wav_data = pcm_to_wav(pcm_data)
            stt_res = await transcribe(wav_data, language_code=self.selected_language)
            transcript = stt_res.get("transcript", "").strip()

            if not transcript:
                self.state = "LISTENING"
                await self.send_state()
                return

            logger.info(f"Turn {expected_turn_id}: Transcript: {transcript}")
            await self.ws.send(json.dumps({"type": "transcript", "text": transcript}))

            lang_code = self.selected_language

            self.state = "SPEAKING"
            self.vad.reset()
            self.speaking_started_at = time.time()
            await self.send_state()

            self.playback_done_event.clear()
            total_audio_seconds = 0.0
            sentence_buffer = ""
            full_answer_parts: list[str] = []
            pending_sentence = None

            async def speak(text: str, is_final: bool):
                nonlocal total_audio_seconds
                if self.interrupt_ctrl.turn_id != expected_turn_id:
                    return
                clean_text = strip_source_markers(text)
                if not clean_text:
                    return
                logger.info(f"Turn {expected_turn_id}: TTS for: {text}")
                tts_wav = await text_to_speech(text, lang_code)
                if self.interrupt_ctrl.turn_id != expected_turn_id:
                    return
                total_audio_seconds += wav_duration_seconds(tts_wav)
                audio_b64 = base64.b64encode(tts_wav).decode('utf-8')
                await self.ws.send(json.dumps({
                    "type": "audio",
                    "audio": audio_b64,
                    "is_final": is_final
                }))

            stream_url = VoiceConfig.RAG_API_URL.replace("/chat/text", "/chat/text/stream")
            headers = {
                "Content-Type": "application/json",
                "X-User-Id": self.user_id,
                "X-Language": lang_code,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(stream_url, json={"message": transcript}, headers=headers) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.content:
                        if self.interrupt_ctrl.turn_id != expected_turn_id:
                            break
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        try:
                            payload = json.loads(data_str)
                        except Exception:
                            continue

                        if "detail" in payload:
                            raise RuntimeError(f"RAG stream error: {payload['detail']}")

                        if "delta" in payload:
                            delta = payload["delta"]
                            sentence_buffer += delta
                            full_answer_parts.append(delta)
                            await self.ws.send(json.dumps({"type": "answer_delta", "text": delta}))

                            pieces = split_into_sentences(sentence_buffer)
                            if len(pieces) > 1:
                                ready_sentences = pieces[:-1]
                                sentence_buffer = pieces[-1]
                                for s in ready_sentences:
                                    if pending_sentence is not None:
                                        await speak(pending_sentence, is_final=False)
                                    pending_sentence = s

            if self.interrupt_ctrl.turn_id == expected_turn_id and sentence_buffer.strip():
                if pending_sentence is not None:
                    await speak(pending_sentence, is_final=False)
                pending_sentence = sentence_buffer.strip()

            if self.interrupt_ctrl.turn_id == expected_turn_id and pending_sentence is not None:
                await speak(pending_sentence, is_final=True)

            answer = clean_answer("".join(full_answer_parts))
            logger.info(f"Turn {expected_turn_id}: RAG Answer (full): {answer}")
            await self.ws.send(json.dumps({"type": "answer", "text": answer}))

            if self.interrupt_ctrl.turn_id == expected_turn_id:
                playback_timeout = max(10.0, total_audio_seconds + 5.0)
                try:
                    await asyncio.wait_for(self.playback_done_event.wait(), timeout=playback_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Turn {expected_turn_id}: playback ack timeout after {playback_timeout:.1f}s, proceeding anyway")

            if self.interrupt_ctrl.turn_id == expected_turn_id:
                self.state = "LISTENING"
                await self.send_state()

        except asyncio.CancelledError:
            logger.info(f"Turn {expected_turn_id}: Pipeline cancelled via barge-in")
        except Exception as e:
            logger.error(f"Turn {expected_turn_id}: Pipeline error: {e}", exc_info=True)
            if self.interrupt_ctrl.turn_id == expected_turn_id:
                self.state = "LISTENING"
                await self.send_state()

async def ws_handler(websocket):
    logger.info("New WebSocket connection")
    session = VoiceSession(websocket)
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary audio data
                await session.handle_audio(message)
            elif isinstance(message, str):
                # Text commands
                try:
                    data = json.loads(message)
                    if data.get("type") == "cmd" and data.get("name") == "start":
                        session.state = "LISTENING"
                        await session.send_state()
                    elif data.get("type") == "cmd" and data.get("name") == "playback_finished":
                        session.playback_done_event.set()
                    elif data.get("type") == "cmd" and data.get("name") == "set_language":
                        new_lang = data.get("language")
                        if new_lang in ("en-IN", "hi-IN", "as-IN", "kn-IN"):
                            session.selected_language = new_lang
                            logger.info(f"Language pinned to: {new_lang}")
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        session.interrupt_ctrl.cancel_current()

async def main():
    logger.info(f"Voice WebSocket server listening on ws://{VoiceConfig.WS_HOST}:{VoiceConfig.WS_PORT}")
    async with websockets.serve(ws_handler, VoiceConfig.WS_HOST, VoiceConfig.WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
