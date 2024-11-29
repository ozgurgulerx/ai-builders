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
        self.input_stream = None
        self.output_stream = None
        self.is_speaking = False
        self.input_buffer = []
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.url = (
            f"wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&api-key={self.api_key}"
        )
        
    async def setup_audio(self):
        self.output_stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        self.input_stream = sd.InputStream(samplerate=24000, channels=1, dtype=np.int16,
                                         callback=self.audio_callback)
        self.output_stream.start()
        self.input_stream.start()
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Input stream error: {status}")
        if not self.is_speaking:
            self.input_buffer.extend(indata.tobytes())
            
    async def process_turn(self):
        try:
            async with websockets.connect(self.url) as ws:
                # Session setup
                await self.setup_session(ws)
                
                while True:
                    # Wait for input
                    if len(self.input_buffer) > 48000:  # 2 seconds of audio
                        audio_data = bytes(self.input_buffer)
                        self.input_buffer.clear()
                        
                        # Send audio to API
                        await self.send_audio(ws, audio_data)
                        
                        # Get and play response
                        await self.handle_response(ws)
                        
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"Error in conversation: {e}")
            
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
        
    async def send_audio(self, ws, audio_data):
        # Convert and send audio data
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
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
                                audio_data += "=" * (4 - padding)
                            
                            audio_bytes = base64.b64decode(audio_data)
                            audio = np.frombuffer(audio_bytes, dtype=np.int16)
                            self.output_stream.write(audio)
                            
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            
                elif data["type"] == "response.done":
                    break
                    
        finally:
            self.is_speaking = False

async def main():
    system = ConversationSystem()
    await system.setup_audio()
    await system.process_turn()

if __name__ == "__main__":
    asyncio.run(main())