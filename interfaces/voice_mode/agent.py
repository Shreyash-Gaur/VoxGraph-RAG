import logging
import os
import sys
import aiohttp
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import silero

# --- PATH CONFIGURATION ---
# Add the 'plugins' folder to the system path so we can import 'local_livekit_plugins'
current_dir = os.path.dirname(os.path.abspath(__file__))
plugins_path = os.path.join(current_dir, "plugins")
sys.path.append(plugins_path)

# Now we can import the local plugins
try:
    from plugins.local_livekit_plugins import FasterWhisperSTT, PiperTTS
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import local plugins from {plugins_path}")
    print("Did you copy the 'local_livekit_plugins' folder into 'interfaces/voice_mode/plugins/'?")
    raise e

# --- CONFIGURATION ---
# Load .env from project root (go up two levels: interfaces/voice_mode -> interfaces -> root)
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
load_dotenv(os.path.join(root_dir, ".env"))

# RAG Backend URL (Running in Docker or Localhost)
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/query")
SESSION_ID = "voxgraph-hybrid-session"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voxgraph-agent")

class VoxGraphAssistant(Agent):
    def __init__(self):
        super().__init__()

    async def query_brain(self, text: str):
        """Sends user voice query to the RAG Brain via HTTP."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": text,
                "mode": "concise",       # Force short answers for voice
                "session_id": SESSION_ID # Shared memory with Chat UI
            }
            try:
                async with session.post(RAG_API_URL, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("answer", "I'm not sure.")
                    else:
                        logger.error(f"RAG Error: {resp.status}")
                        return "I am having trouble connecting to my brain."
            except Exception as e:
                logger.error(f"Brain Disconnected: {e}")
                return "System offline. Check backend connection."

async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    # 1. Initialize Local Hearing (Whisper)
    # Uses GPU if available (CUDA), otherwise CPU
    stt = FasterWhisperSTT(
        model_size="medium", 
        device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("USE_CUDA") else "cpu",
        compute_type="float16" if os.getenv("USE_CUDA") else "int8"
    )

    # 2. Initialize Local Speech (Piper)
    # Path to the .onnx model file
    model_path = os.path.join(current_dir, "models", "en_US-ryan-high.onnx")
    
    if not os.path.exists(model_path):
        logger.error(f"Voice model not found at: {model_path}")
        raise FileNotFoundError("Please download the Piper .onnx model and place it in the models folder.")

    tts = PiperTTS(
        model_path=model_path,
        use_cuda=False # Piper is often faster on CPU for single streams
    )

    # 3. Create Session (No LLM, manual logic)
    session = AgentSession(
        stt=stt, 
        tts=tts, 
        llm=None, # We disable the default LLM loop to use our Graph RAG
        vad=silero.VAD.load()
    )

    # 4. The Conversation Loop
    @session.on("user_input_transcribed")
    async def on_user_speech(ev):
        user_text = ev.transcript.strip()
        if not user_text: return
        
        logger.info(f"User (Voice): {user_text}")
        
        # A. Get answer from RAG Brain
        answer = await assistant.query_brain(user_text)
        logger.info(f"VoxGraph (Answer): {answer}")

        # B. Speak answer
        await session.speak(tts.synthesize(answer))

    assistant = VoxGraphAssistant()
    await session.start(room=ctx.room, agent=assistant)
    
    # Initial Greeting
    await session.speak(tts.synthesize("Vox Graph System Online. I am listening."))

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))