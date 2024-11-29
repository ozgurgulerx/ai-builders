import asyncio
import os
import base64
import json
from dotenv import load_dotenv
import websockets
import numpy as np
import sounddevice as sd

class ConversationSystem:
    def __init__(self):
        load_dotenv()
        self.setup_initial_state()
        self.setup_vad_params()

    def setup_initial_state(self):
        self.input_stream = None
        self.output_stream = None
        self.is_speaking = False
        self.input_buffer = []
        self.speech_detected = False
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found")
        self.url = (
            f"wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&api-key={self.api_key}"
        )

    def setup_vad_params(self):
        self.vad_threshold = 1000
        self.silence_duration = 0
        self.max_silence = 15
        self.min_speech_frames = 24000

    async def setup_audio(self):
        self.output_stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.input_stream = sd.InputStream(samplerate=24000, channels=1, dtype=np.int16,
                                         callback=self.audio_callback,
                                         blocksize=4800)
        self.output_stream.start()
        self.input_stream.start()
        print("Audio ready")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Input error: {status}")
            return

        if not self.is_speaking:
            audio_level = np.abs(indata).mean()
            
            if audio_level > self.vad_threshold:
                self.speech_detected = True
                self.silence_duration = 0
                self.input_buffer.extend(indata.tobytes())
            elif self.speech_detected:
                self.silence_duration += 1
                self.input_buffer.extend(indata.tobytes())

    async def setup_session(self, ws):
        session_payload = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": "You are a helpful AI assistant. Keep responses brief and engaging.",
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                }
            }
        }
        await ws.send(json.dumps(session_payload))
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "session.created":
                print("Session setup complete")
                break
            elif data.get("type") == "error":
                raise Exception("Error creating session")

    async def process_audio_input(self, ws):
        self.speech_detected = False
        self.silence_duration = 0
        self.input_buffer.clear()
        
        print("Listening...")
        while True:
            if self.speech_detected and self.silence_duration >= self.max_silence:
                if len(self.input_buffer) >= self.min_speech_frames * 2:
                    return True
            elif not self.speech_detected and len(self.input_buffer) > 48000:
                self.input_buffer.clear()
            await asyncio.sleep(0.1)

    async def start_conversation(self):
        try:
            async with websockets.connect(self.url) as ws:
                print("Connected")
                await self.setup_session(ws)
                await self.send_message(ws, "Hello")
                await self.handle_response(ws)

                while True:
                    if await self.process_audio_input(ws):
                        print("Processing speech...")
                        await self.send_audio(ws)
                        await self.handle_response(ws)

        except Exception as e:
            print(f"Error: {e}")

    async def send_message(self, ws, text):
        message_payload = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        await ws.send(json.dumps(message_payload))
        
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def send_audio(self, ws):
        audio_data = bytes(self.input_buffer)
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, ws):
        self.is_speaking = True
        try:
            while True:
                response = await ws.recv()
                data = json.loads(response)
                
                if data["type"] == "response.audio.delta":
                    if "delta" in data:
                        try:
                            audio_data = data["delta"].replace(" ", "").replace("\n", "")
                            padding = len(audio_data) % 4
                            if padding:
                                audio_data += "=" * padding
                            
                            audio_bytes = base64.b64decode(audio_data)
                            audio = np.frombuffer(audio_bytes, dtype=np.int16)
                            self.output_stream.write(audio)
                            print(".", end="", flush=True)
                            
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            
                elif data["type"] == "response.done":
                    break
                    
        finally:
            self.is_speaking = False

async def main():
    system = ConversationSystem()
    await system.setup_audio()
    await system.start_conversation()

if __name__ == "__main__":
    print("Starting conversation system...")
    asyncio.run(main())