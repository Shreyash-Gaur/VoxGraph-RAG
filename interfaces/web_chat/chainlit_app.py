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

# --- 1. Robust Path Setup for Voice Plugins & Backend ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Safely find the project root (VoxGraph-RAG)
if os.path.basename(os.path.dirname(current_dir)) == "interfaces":
    root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
else:
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))

plugins_path = os.path.join(root_dir, "interfaces", "voice_mode", "plugins")
models_path = os.path.join(root_dir, "interfaces", "voice_mode", "models", "en_US-ryan-high.onnx")

# Inject paths into Python's brain
if plugins_path not in sys.path:
    sys.path.insert(0, plugins_path)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"📂 Project Root: {root_dir}")
print(f"🔌 Plugins Path: {plugins_path}")

# --- 2. Now it is safe to import from Backend and Plugins! ---
try:
    from backend.core.config import settings
    print("✅ Backend Config Loaded")
except ImportError as e:
    print(f"❌ Backend Import Error: {e}")

try:
    from local_livekit_plugins import FasterWhisperSTT, PiperTTS
    VOICE_ENABLED = True
    print("✅ Voice plugins loaded successfully!")
except ImportError as e:
    print(f"⚠️ Voice disabled. Import Error: {e}")
    VOICE_ENABLED = False

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

stt_engine = None
tts_engine = None

def setup_voice_engines():
    global stt_engine, tts_engine
    if not VOICE_ENABLED: return

    print("🎙️ Loading Voice Models... (This may take a moment)")
    
    try:
        stt_instance = FasterWhisperSTT(model_size="medium", device="auto")
        stt_engine = stt_instance._model
        print("✅ Whisper STT Loaded")
    except Exception as e:
        print(f"❌ Failed to load STT: {e}")

    try:
        if os.path.exists(models_path):
            tts_instance = PiperTTS(model_path=models_path, use_cuda=False)
            tts_engine = tts_instance.voice 
            print("✅ Piper TTS Loaded")
        else:
            print(f"❌ Piper Model not found at: {models_path}")
    except Exception as e:
        print(f"❌ Failed to load TTS: {e}")

# --- Helper Functions ---

def transcribe_audio_from_file(file_path: str) -> str:
    """Converts audio file to text using Whisper."""
    if not stt_engine: return ""
    try:
        segments, info = stt_engine.transcribe(file_path, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        return text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

def generate_speech(text: str):
    """Converts text to an audio element using Piper."""
    if not tts_engine: return None

    wav_io = io.BytesIO()
    from piper.config import SynthesisConfig
    conf = SynthesisConfig(length_scale=1.0, volume=1.0) 

    try:
        with wave.open(wav_io, "wb") as wav_file:
            tts_engine.synthesize_wav(text, wav_file, syn_config=conf, set_wav_format=True)
        
        wav_io.seek(0)
        return cl.Audio(
            name="reply.wav", 
            content=wav_io.read(), 
            mime="audio/wav", 
            auto_play=True 
        )
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def backend_query(query: str, top_k: int = settings.TOP_K_RETRIEVAL, max_tokens: int = settings.MAX_TOKENS, temperature: float = 0.0):
    url = f"{BACKEND_URL}/query"
    payload = {
        "query": query,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(url, json=payload, timeout=150)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_file_to_backend(file_element):
    """
    Uploads a file element to the backend /ingest/upload endpoint.
    """
    url = f"{BACKEND_URL}/ingest/upload"
    try:
        # Chainlit files are stored in a temp path
        with open(file_element.path, "rb") as f:
            files = {"file": (file_element.name, f, file_element.mime)}
            response = requests.post(url, files=files, timeout=60)
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

# --- Chainlit Handlers ---

@cl.on_chat_start
async def start():
    if VOICE_ENABLED and (stt_engine is None or tts_engine is None):
        setup_voice_engines()
    await cl.Message(content="# 🎙️ VoxGraph-RAG Ready\nI am listening. Type or click the microphone to speak.").send()

@cl.on_audio_start
async def on_audio_start():
    """Triggered when the microphone is clicked (Required for Chainlit 2.x)."""
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk):
    """Accumulate raw PCM audio chunks."""
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunks.append(chunk.data)

@cl.on_audio_end
async def on_audio_end():
    """Triggered when recording stops."""
    audio_chunks = cl.user_session.get("audio_chunks", [])
    if not audio_chunks: return

    # Combine chunks into raw bytes
    audio_blob = b"".join(audio_chunks)
    
    msg = cl.Message(content="👂 **Listening...**")
    await msg.send()
    
    # Save raw PCM bytes as a WAV file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp, "wb") as wav_file:
            wav_file.setnchannels(1) # Mono
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(24000) # Chainlit 2.x default sampling rate
            wav_file.writeframes(audio_blob)
        tmp_path = tmp.name

    try:
        # 1. Transcribe
        text = await cl.make_async(transcribe_audio_from_file)(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    if not text:
        msg.content = "❌ **Could not understand audio.**"
        await msg.update()
        return

    msg.content = f"🗣️ **You said:** \"{text}\""
    await msg.update()

    # 2. Feed text into main chat handler
    mock_message = cl.Message(content=text)
    await main(mock_message)

@cl.on_message
async def main(message: cl.Message):

    # --- 0. HANDLE FILE UPLOADS ---
    if message.elements:
        # If the user attached files, process them
        uploaded_files = []
        for element in message.elements:
            # Check if it's a file
            if isinstance(element, cl.File):
                success = await cl.make_async(upload_file_to_backend)(element)
                if success:
                    uploaded_files.append(element.name)
        
        if uploaded_files:
            await cl.Message(
                content=f"✅ **Uploaded:** {', '.join(uploaded_files)}\n"
                        f"*The watcher is processing these files. They will be available for search shortly.*"
            ).send()
            
        # If there is no text query with the file, stop here
        if not message.content:
            return

    # --- 1. HANDLE TEXT QUERY ---
    user_query = message.content.strip()
    
    wait_msg = cl.Message(content="🔎 **Thinking...**")
    await wait_msg.send()

    resp = await cl.make_async(backend_query)(user_query)

    if "error" in resp:
        wait_msg.content = f"❌ **Error:** {resp['error']}"
        await wait_msg.update()
        return

    answer = resp.get("answer", "No answer generated.")
    sources = resp.get("sources", [])
    
    source_text = ""
    if sources:
        source_text = "\n\n---\n**📚 Sources:**\n"
        for i, s in enumerate(sources, 1):
            meta = s.get("metadata") or s.get("meta") or {}
            text = s.get("text") or meta.get("text") or "No text available"
            preview = text[:100].replace("\n", " ") + "..."
            source_text += f"{i}. {preview}\n"

    audio_element = None
    if VOICE_ENABLED and tts_engine:
        audio_element = await cl.make_async(generate_speech)(answer)

    elements = [audio_element] if audio_element else []
    wait_msg.content = f"{answer}{source_text}"
    wait_msg.elements = elements
    await wait_msg.update()

    base_tokens = settings.MAX_TOKENS
    long_tokens = base_tokens * 2

    actions = [
        cl.Action(
            name="long_answer", 
            payload={"query": user_query, "max_tokens": long_tokens}, 
            label=f"📝 Long Answer ({long_tokens} tokens)"
        ),
        cl.Action(
            name="creative_answer", 
            payload={"query": user_query}, 
            label="🎨 Creative Answer"
        )
    ]
    await cl.Message(content="**Refine this answer:**", actions=actions).send()

@cl.action_callback("long_answer")
async def on_long_answer(action):
    """Handler for Long Answer button."""
    query = action.payload["query"]
    req_tokens = action.payload["max_tokens"]
    
    msg = cl.Message(content="📝 **Generating detailed answer...**")
    await msg.send()

    resp = await cl.make_async(backend_query)(query, max_tokens=req_tokens)

    if "error" in resp:
        msg.content = f"❌ Error: {resp['error']}"
    else:
        answer = resp.get("answer", "No answer.")
        msg.content = f"**Detailed Answer:**\n\n{answer}"
    
    if VOICE_ENABLED and tts_engine:
            audio_element = await cl.make_async(generate_speech)(answer)
            if audio_element:
                msg.elements = [audio_element]
    
    await msg.update()

@cl.action_callback("creative_answer")
async def on_creative_answer(action):
    """Handler for Creative Answer button."""
    query = action.payload["query"]
    
    msg = cl.Message(content="🎨 **Thinking creatively...**")
    await msg.send()

    resp = await cl.make_async(backend_query)(query, temperature=0.7)

    if "error" in resp:
        msg.content = f"❌ Error: {resp['error']}"
    else:
        answer = resp.get("answer", "No answer.")
        msg.content = f"**Creative Answer:**\n\n{answer}"
    
    if VOICE_ENABLED and tts_engine:
            audio_element = await cl.make_async(generate_speech)(answer)
            if audio_element:
                msg.elements = [audio_element]
    
    await msg.update()