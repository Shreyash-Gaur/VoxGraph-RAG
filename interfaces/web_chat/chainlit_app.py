"""
VoxGraph-RAG: Hybrid Voice & Text Agent Interface
Run with: chainlit run interfaces/web_chat/chainlit_app.py -w --port 8001
"""
import os
import sys
import requests
import chainlit as cl
import io
import wave
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(os.path.dirname(current_dir)) == "interfaces":
    root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
else:
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))

plugins_path = os.path.join(root_dir, "interfaces", "voice_mode", "plugins")
models_path = os.path.join(root_dir, "interfaces", "voice_mode", "models", "en_US-ryan-high.onnx")

if plugins_path not in sys.path:
    sys.path.insert(0, plugins_path)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"Project Root: {root_dir}")
print(f"Plugins Path: {plugins_path}")

try:
    from backend.core.config import settings
    print("Backend Config Loaded")
except ImportError as e:
    print(f"Backend Import Error: {e}")

try:
    from local_livekit_plugins import FasterWhisperSTT, PiperTTS
    VOICE_ENABLED = True
    print("Voice plugins loaded successfully!")
except ImportError as e:
    print(f"Voice disabled. Import Error: {e}")
    VOICE_ENABLED = False

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

stt_engine = None
tts_engine = None


def setup_voice_engines():
    global stt_engine, tts_engine
    if not VOICE_ENABLED:
        return
    print("Loading Voice Models...")
    try:
        stt_instance = FasterWhisperSTT(model_size="medium", device="auto")
        stt_engine = stt_instance._model
        print("Whisper STT Loaded")
    except Exception as e:
        print(f"Failed to load STT: {e}")
    try:
        if os.path.exists(models_path):
            tts_instance = PiperTTS(model_path=models_path, use_cuda=False)
            tts_engine = tts_instance.voice
            print("Piper TTS Loaded")
        else:
            print(f"Piper Model not found at: {models_path}")
    except Exception as e:
        print(f"Failed to load TTS: {e}")


def transcribe_audio_from_file(file_path: str) -> str:
    if not stt_engine:
        return ""
    try:
        segments, info = stt_engine.transcribe(file_path, beam_size=5)
        return " ".join([s.text for s in segments]).strip()
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""


def generate_speech(text: str):
    if not tts_engine:
        return None
    wav_io = io.BytesIO()
    from piper.config import SynthesisConfig
    conf = SynthesisConfig(length_scale=1.0, volume=1.0)
    try:
        with wave.open(wav_io, "wb") as wav_file:
            tts_engine.synthesize_wav(text, wav_file, syn_config=conf, set_wav_format=True)
        wav_io.seek(0)
        return cl.Audio(name="reply.wav", content=wav_io.read(), mime="audio/wav", auto_play=True)
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def backend_query(
    query: str,
    mode: str = "concise",
    max_tokens: int = settings.MAX_TOKENS,
    temperature: float = 0.0,
    top_k: int = settings.TOP_K_RETRIEVAL,
    bypass_cache: bool = False,
) -> dict:
    url     = f"{BACKEND_URL}/query"
    payload = {
        "query":        query,
        "mode":         mode,
        "max_tokens":   max_tokens,
        "temperature":  temperature,
        "top_k":        top_k,
        "bypass_cache": bypass_cache,
    }
    try:
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def upload_file_to_backend(file_element):
    url = f"{BACKEND_URL}/ingest/upload"
    try:
        with open(file_element.path, "rb") as f:
            files = {"file": (file_element.name, f, file_element.mime)}
            response = requests.post(url, files=files, timeout=60)
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


@cl.on_chat_start
async def start():
    if VOICE_ENABLED and (stt_engine is None or tts_engine is None):
        setup_voice_engines()
    await cl.Message(content="# VoxGraph-RAG Ready\nI am listening. Type or click the microphone to speak.").send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunks.append(chunk.data)


@cl.on_audio_end
async def on_audio_end():
    audio_chunks = cl.user_session.get("audio_chunks", [])
    if not audio_chunks:
        return

    audio_blob = b"".join(audio_chunks)
    msg = cl.Message(content="Listening...")
    await msg.send()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_blob)
        tmp_path = tmp.name

    try:
        text = await cl.make_async(transcribe_audio_from_file)(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text:
        msg.content = "Could not understand audio."
        await msg.update()
        return

    msg.content = f"You said: \"{text}\""
    await msg.update()
    await main(cl.Message(content=text))


@cl.on_message
async def main(message: cl.Message):
    # --- File uploads ---
    if message.elements:
        uploaded = []
        for element in message.elements:
            if isinstance(element, cl.File):
                ok = await cl.make_async(upload_file_to_backend)(element)
                if ok:
                    uploaded.append(element.name)
        if uploaded:
            await cl.Message(
                content=f"Uploaded: {', '.join(uploaded)}\n"
                        f"*Files are being indexed and will be searchable shortly.*"
            ).send()
        if not message.content:
            return

    user_query = message.content.strip()

    wait_msg = cl.Message(content="Thinking...")
    await wait_msg.send()

    # Regular concise query — uses semantic cache
    resp = await cl.make_async(backend_query)(
        user_query,
        mode="concise",
        max_tokens=settings.MAX_TOKENS,
        temperature=0.0,
    )

    if "error" in resp:
        wait_msg.content = f"Error: {resp['error']}"
        await wait_msg.update()
        return

    answer  = resp.get("answer", "No answer generated.")
    sources = resp.get("sources", [])

    source_text = ""
    if sources:
        source_text = "\n\n---\n**Sources:**\n"
        for i, s in enumerate(sources, 1):
            meta    = s.get("metadata") or s.get("meta") or {}
            text    = s.get("text") or meta.get("text") or ""
            preview = text[:100].replace("\n", " ") + "..."
            source_text += f"{i}. {preview}\n"

    audio_element = None
    if VOICE_ENABLED and tts_engine:
        audio_element = await cl.make_async(generate_speech)(answer)

    wait_msg.content  = f"{answer}{source_text}"
    wait_msg.elements = [audio_element] if audio_element else []
    await wait_msg.update()

    long_tokens = settings.MAX_TOKENS * 2
    actions = [
        cl.Action(
            name="long_answer",
            payload={"query": user_query, "max_tokens": long_tokens},
            label=f"Long answer ({long_tokens} tokens)",
        ),
        cl.Action(
            name="creative_answer",
            payload={"query": user_query},
            label="Creative answer",
        ),
    ]
    await cl.Message(content="Refine this answer:", actions=actions).send()


@cl.action_callback("long_answer")
async def on_long_answer(action):
    query      = action.payload["query"]
    req_tokens = action.payload["max_tokens"]

    msg = cl.Message(content="Generating detailed answer...")
    await msg.send()

    resp = await cl.make_async(backend_query)(
        query,
        mode="detailed",       # was missing
        max_tokens=req_tokens,
        temperature=0.0,
        bypass_cache=True,     # was missing — caused cache hit
    )

    if "error" in resp:
        msg.content = f"Error: {resp['error']}"
    else:
        answer      = resp.get("answer", "No answer.")
        sources     = resp.get("sources", [])
        source_text = ""
        if sources:
            source_text = "\n\n---\n**Sources:**\n"
            for i, s in enumerate(sources, 1):
                meta    = s.get("metadata") or s.get("meta") or {}
                text    = s.get("text") or meta.get("text") or ""
                preview = text[:100].replace("\n", " ") + "..."
                source_text += f"{i}. {preview}\n"
        msg.content = f"**Detailed answer:**\n\n{answer}{source_text}"

        if VOICE_ENABLED and tts_engine:
            audio_element = await cl.make_async(generate_speech)(answer)
            if audio_element:
                msg.elements = [audio_element]

    await msg.update()


@cl.action_callback("creative_answer")
async def on_creative_answer(action):
    query = action.payload["query"]

    msg = cl.Message(content="Thinking creatively...")
    await msg.send()

    resp = await cl.make_async(backend_query)(
        query,
        mode="detailed",
        max_tokens=settings.MAX_TOKENS,
        temperature=0.7,
        bypass_cache=True,     # was missing — caused cache hit
    )

    if "error" in resp:
        msg.content = f"Error: {resp['error']}"
    else:
        answer      = resp.get("answer", "No answer.")
        sources     = resp.get("sources", [])
        source_text = ""
        if sources:
            source_text = "\n\n---\n**Sources:**\n"
            for i, s in enumerate(sources, 1):
                meta    = s.get("metadata") or s.get("meta") or {}
                text    = s.get("text") or meta.get("text") or ""
                preview = text[:100].replace("\n", " ") + "..."
                source_text += f"{i}. {preview}\n"
        msg.content = f"**Creative answer:**\n\n{answer}{source_text}"

        if VOICE_ENABLED and tts_engine:
            audio_element = await cl.make_async(generate_speech)(answer)
            if audio_element:
                msg.elements = [audio_element]

    await msg.update()