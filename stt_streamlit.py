import io
import os
import re
import uuid
import base64
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from voice_component import voice_component

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

LANGUAGES = {
    "English":  "en-IN",
    "Hindi":    "hi-IN",
    "Kannada":  "kn-IN",
    "Assamese": None,
}


def clean_answer(text):
    if not text:
        return ""
    text = re.sub(r"\[\s*s\d+(?:\s*,\s*s\d+)*\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)s\d+(?!\w)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
# Per-language conversation storage: {lang: {messages, conversation_id}}
if "lang_convos" not in st.session_state:
    st.session_state.lang_convos = {}
if "page" not in st.session_state:
    st.session_state.page = "Bot"

st.set_page_config(page_title="Multilingual RAG Assistant", page_icon="\U0001f916", layout="wide")

st.markdown("""<style>
html, body { overflow: hidden !important; height: 100% !important; }
[data-testid="stAppViewContainer"] { overflow: hidden !important; height: 100vh !important; }
[data-testid="stMain"] { overflow: hidden !important; }
.main .block-container { overflow: hidden !important; padding-bottom: 0 !important; padding-top: 1rem !important; }
[data-testid="stBottom"] { padding-top: 0 !important; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("\U0001f916 RAG Assistant")
    for _p in ["\U0001f916 Bot", "\U0001f3a4 Voice AI", "\U0001f680 Intelligent Voice AI", "\U0001f4c4 Upload Document"]:
        if st.button(_p, use_container_width=True, type="primary" if st.session_state.page == _p else "secondary"):
            st.session_state.page = _p
            st.rerun()

# Language persisted in session so all pages share it
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"
if "last_language" not in st.session_state:
    st.session_state.last_language = st.session_state.selected_language

def _convo(lang: str) -> dict:
    """Get or create the conversation bucket for a language."""
    if lang not in st.session_state.lang_convos:
        st.session_state.lang_convos[lang] = {"messages": [], "conversation_id": None}
    return st.session_state.lang_convos[lang]


def get_headers():
    lang_code = LANGUAGES[st.session_state.selected_language]
    headers = {"X-User-Id": st.session_state.user_id}
    if lang_code:
        headers["X-Language"] = lang_code
    return headers


page = st.session_state.page

# ── Bot ──────────────────────────────────────────────────────────────────────
if page == "\U0001f916 Bot":
    _title_col, _clear_col, _lang_col = st.columns([5, 1, 1], vertical_alignment="bottom")
    with _title_col:
        st.title("\U0001f916 Bot")
    with _clear_col:
        if _convo(st.session_state.selected_language)["messages"] and st.button("\U0001f5d1\ufe0f Clear", use_container_width=True):
            st.session_state.lang_convos[st.session_state.selected_language] = {"messages": [], "conversation_id": None}
            st.rerun()
    with _lang_col:
        st.session_state.selected_language = st.selectbox(
            "Language", options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language),
            label_visibility="collapsed",
        )
    # Save current language state then switch — no data lost
    if st.session_state.selected_language != st.session_state.last_language:
        st.session_state.last_language = st.session_state.selected_language
        st.rerun()

    # Build the full chat history as one scrollable HTML block
    def _bubble(msg):
        text = msg["content"]
        if msg["role"] == "user":
            return (
                f'<div style="display:flex;justify-content:flex-end;margin:6px 0">'
                f'<div style="background:#DCF8C6;padding:10px 14px;border-radius:16px 16px 4px 16px;'
                f'max-width:70%;word-wrap:break-word">{text}</div></div>'
            )
        srcs = msg.get("sources", [])
        src_html = ""
        if srcs:
            items = "".join(
                f'<div style="font-size:0.78em;color:#555;margin-top:4px">'
                f'📄 {s.get("document_name","")} p.{s.get("page_number","?")}</div>'
                for s in srcs
            )
            src_html = f'<details style="margin-top:6px"><summary style="font-size:0.8em;cursor:pointer">📚 {len(srcs)} source(s)</summary>{items}</details>'
        return (
            f'<div style="display:flex;justify-content:flex-start;margin:6px 0">'
            f'<div style="background:#F0F0F0;padding:10px 14px;border-radius:16px 16px 16px 4px;'
            f'max-width:70%;word-wrap:break-word">{text}{src_html}</div></div>'
        )

    _c = _convo(st.session_state.selected_language)

    bubbles = "".join(_bubble(m) for m in _c["messages"])
    st.markdown(
        f'<div id="chat-scroll" style="height:calc(100vh - 320px);overflow-y:auto;border:1px solid #e0e0e0;'
        f'border-radius:8px;padding:12px;background:#fff">{bubbles}'
        f'</div>'
        f'<script>var s=document.getElementById("chat-scroll");s.scrollTop=s.scrollHeight;</script>',
        unsafe_allow_html=True,
    )

    if prompt := st.chat_input(f"Ask a question in {st.session_state.selected_language}..."):
        _c["messages"].append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/chat/text",
                    headers=get_headers(),
                    json={
                        "message": prompt,
                        "conversation_id": _c["conversation_id"],
                    },
                    timeout=180,
                )
                if resp.ok:
                    data = resp.json()
                    _c["conversation_id"] = str(data.get("conversation_id", ""))
                    answer = clean_answer(data.get("answer", ""))
                    sources = data.get("sources", [])
                    _c["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    _c["messages"].append(
                        {"role": "assistant",
                         "content": f"Request failed ({resp.status_code})\n```\n{resp.text}\n```",
                         "sources": []}
                    )
            except Exception as e:
                _c["messages"].append(
                    {"role": "assistant", "content": f"\u274c {e}", "sources": []}
                )
        st.rerun()

# ── Voice AI ──────────────────────────────────────────────────────────────────
elif page == "\U0001f3a4 Voice AI":
    st.title("\U0001f3a4 Voice AI")
    st.write(f"Record your question in **{st.session_state.selected_language}**.")

    audio_bytes = audio_recorder(
        text="Click to record",
        pause_threshold=3600,
        recording_color="#ff4b4b",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x",
        key="voice_ai_recorder",
    )

    if audio_bytes:
        st.success(f"Recording captured \u2014 {len(audio_bytes):,} bytes")
        st.audio(audio_bytes, format="audio/wav")

        st.subheader("1\ufe0f\u20e3 Speech-to-Text")
        if st.button("Transcribe Audio", key="transcribe_button", type="primary"):
            with st.spinner(f"Transcribing in {st.session_state.selected_language}..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/api/v1/chat/voice/transcribe",
                        headers=get_headers(),
                        files={"file": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")},
                        timeout=120,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.session_state["transcript"] = data.get("transcript", "")
                        st.session_state["stt_data"] = data
                    else:
                        st.error(f"STT failed ({resp.status_code})")
                        st.code(resp.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")

        if "transcript" in st.session_state:
            st.subheader("\U0001f4dd Transcript")
            st.text_area(
                "STT Output",
                value=st.session_state["transcript"],
                height=120,
                disabled=True,
            )

        st.subheader("2\ufe0f\u20e3 Voice \u2192 RAG \u2192 Answer")
        if st.button("Ask RAG Using This Audio", key="voice_rag_button"):
            with st.spinner("Running STT \u2192 Retrieval \u2192 LLM..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/api/v1/chat/voice",
                        headers=get_headers(),
                        files={"file": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")},
                        timeout=180,
                    )
                    if resp.ok:
                        st.session_state["voice_result"] = resp.json()
                    else:
                        st.error(f"Voice RAG failed ({resp.status_code})")
                        st.code(resp.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")

        if "voice_result" in st.session_state:
            result = st.session_state["voice_result"]
            st.subheader("\U0001f4dd Transcribed Question")
            st.text_area("Question", value=result.get("transcript", ""), height=100, disabled=True)
            st.subheader("\U0001f916 RAG Answer")
            st.write(clean_answer(result.get("answer", "")))
            audio_b64 = result.get("audio_base64")
            if audio_b64:
                try:
                    st.subheader("\U0001f50a Assistant Voice")
                    st.audio(base64.b64decode(audio_b64), format="audio/wav")
                except Exception as e:
                    st.error(f"Could not play audio: {e}")
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"\U0001f4da Sources ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")
                        st.write(src)

# ── Intelligent Voice AI ──────────────────────────────────────────────────────
elif page == "\U0001f680 Intelligent Voice AI":
    st.title("\U0001f680 Intelligent Voice AI")
    st.write("This mode supports continuous listening and interruption (barge-in).")
    voice_component()

# ── Upload Document ───────────────────────────────────────────────────────────
elif page == "\U0001f4c4 Upload Document":
    st.title("\U0001f4c4 Upload Document")

    # ── Upload form ──────────────────────────────────────────────────────────
    with st.expander("\U0001f4e4 Upload a new PDF", expanded=True):
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file:
            st.caption(f"{uploaded_file.name} — {uploaded_file.size:,} bytes")
            if st.button("Upload & Ingest", type="primary", key="upload_document_button"):
                with st.spinner("Uploading and starting ingestion..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/api/v1/documents",
                            headers=get_headers(),
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                            timeout=300,
                        )
                        if resp.ok:
                            st.success("\u2705 Uploaded — ingestion started.")
                            st.rerun()
                        else:
                            st.error(f"\u274c Upload failed ({resp.status_code})")
                            st.code(resp.text)
                    except Exception as e:
                        st.error(f"\u274c {e}")

    # ── Document list ────────────────────────────────────────────────────────
    st.subheader("\U0001f4c2 Your Documents")
    try:
        list_resp = requests.get(
            f"{API_BASE_URL}/api/v1/documents",
            headers=get_headers(),
            params={"limit": 50},
            timeout=30,
        )
        if list_resp.ok:
            docs = list_resp.json().get("items", [])
            if not docs:
                st.info("No documents uploaded yet.")
            else:
                for doc in docs:
                    col_name, col_status, col_rename, col_del = st.columns([4, 1.5, 1, 0.8])
                    with col_name:
                        st.markdown(f"**{doc['display_name']}**")
                        st.caption(f"{doc.get('page_count') or '?'} pages · {doc.get('file_size_bytes', 0)//1024:,} KB")
                    with col_status:
                        status_icon = {"ready": "\u2705", "processing": "\u23f3", "uploaded": "\u23f3", "error": "\u274c"}.get(doc["status"], "\u2022")
                        st.markdown(f"{status_icon} {doc['status']}")
                    with col_rename:
                        new_name = st.text_input("", value=doc["display_name"], key=f"rename_{doc['id']}", label_visibility="collapsed")
                        if new_name != doc["display_name"]:
                            if st.button("Save", key=f"save_{doc['id']}"):
                                requests.patch(
                                    f"{API_BASE_URL}/api/v1/documents/{doc['id']}",
                                    headers=get_headers(),
                                    json={"name": new_name},
                                    timeout=15,
                                )
                                st.rerun()
                    with col_del:
                        if st.button("\U0001f5d1\ufe0f", key=f"del_{doc['id']}", help="Delete"):
                            requests.delete(
                                f"{API_BASE_URL}/api/v1/documents/{doc['id']}",
                                headers=get_headers(),
                                timeout=15,
                            )
                            st.rerun()
                    st.divider()
        else:
            st.error(f"Could not load documents ({list_resp.status_code})")
    except Exception as e:
        st.error(f"\u274c {e}")
