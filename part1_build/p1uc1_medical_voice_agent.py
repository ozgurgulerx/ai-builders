import asyncio
import os
import base64
import json
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
import re
from datetime import datetime

class AudioProcessor:
    """Handles real-time audio processing and voice activity detection"""
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.vad_threshold = 0.015  # Voice activity detection threshold
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_duration = int(0.3 * sample_rate)
        self.max_silence_duration = int(0.8 * sample_rate)
        self.buffer = []
        self.is_speaking = False
        self.speech_detected = False
        print("Audio processor initialized")

    def process_audio(self, indata):
        """Process incoming audio data and detect voice activity"""
        if self.is_speaking:
            return

        audio_level = np.abs(indata).mean() / 32768.0
        
        if audio_level > self.vad_threshold:
            self.speech_detected = True
            self.speech_frames += len(indata)
            self.silence_frames = 0
            self.buffer.extend(indata.tobytes())
        elif self.speech_detected:
            self.silence_frames += len(indata)
            if self.silence_frames < self.max_silence_duration:
                self.buffer.extend(indata.tobytes())

    def should_process(self):
        """Determine if we have enough speech to process"""
        return (self.speech_detected and 
                self.speech_frames >= self.min_speech_duration and 
                self.silence_frames >= self.max_silence_duration)

    def reset(self):
        """Reset the audio buffer and counters"""
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_detected = False
        audio_data = bytes(self.buffer)
        self.buffer.clear()
        return audio_data

class InsuranceConversationState:
    """Manages the state and flow of insurance-related conversations"""
    def __init__(self):
        self.current_state = "greeting"  # Initial state
        self.customer_id = None  # Stores customer ID when provided
        self.customer_query = None  # Stores the current query
        self.policy_checked = False  # Tracks if policy has been checked
        print("Conversation state initialized")

    def update_state(self, new_state):
        """Update the conversation state and log the transition"""
        print(f"Conversation state changing from {self.current_state} to {new_state}")
        self.current_state = new_state

class InsuranceConversationSystem:
    """Main system for handling insurance-related voice conversations"""
    def __init__(self):
        print("Initializing Insurance Conversation System...")
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")
            
        self.url = (
            "wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
            f"api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&"
            f"api-key={self.api_key}"
        )
        
        self.audio_processor = AudioProcessor()
        self.streams = {'input': None, 'output': None}
        self.conversation_state = InsuranceConversationState()
        print("System initialization complete")

    def audio_callback(self, indata, frames, time, status):
        """Handle incoming audio data"""
        if status:
            print(f"Audio error: {status}")
            return
        self.audio_processor.process_audio(indata)

    async def setup_audio(self):
        """Initialize audio input and output streams"""
        print("Setting up audio streams...")
        try:
            self.streams['output'] = sd.OutputStream(
                samplerate=24000, channels=1, dtype=np.int16)
            self.streams['input'] = sd.InputStream(
                samplerate=24000, channels=1, dtype=np.int16,
                callback=self.audio_callback, blocksize=4800)
                
            for stream in self.streams.values():
                stream.start()
            print("Audio streams started successfully")
        except Exception as e:
            print(f"Error setting up audio: {e}")
            raise

    async def setup_websocket_session(self, websocket):
        """Set up the websocket session with insurance-specific configuration"""
        print("Configuring insurance agent session...")
        session_config = {
            "type": "session.update",
            "session": {
                "voice": "alloy",
                "instructions": """You are Alex, a professional customer service representative 
                for AtlasMedical Insurance. Start with: 'Hello, this is Alex from AtlasMedical 
                Insurance. How may I assist you with your insurance coverage today?' 
                Always ask for customer ID (5-digit number) before providing policy information. 
                Keep responses professional but warm.""",
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 150,
                    "silence_duration_ms": 600
                }
            }
        }
        
        await websocket.send(json.dumps(session_config))
        
        while True:
            response = json.loads(await websocket.recv())
            if response["type"] == "session.created":
                print("Agent session created successfully")
                break
            if response["type"] == "error":
                raise Exception(f"Session setup failed: {response}")

    async def send_audio(self, websocket, audio_data):
        """Send audio data to Azure API"""
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }))
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await websocket.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]}
        }))

    async def handle_response(self, websocket):
        """Process responses and manage conversation flow"""
        self.audio_processor.is_speaking = True
        
        try:
            while True:
                response = json.loads(await websocket.recv())
                
                if response["type"] == "response.text":
                    # Process transcribed text
                    customer_text = response.get('text', '').lower()
                    print(f"\nCustomer: {customer_text}")
                    
                    # Generate appropriate response based on conversation state
                    agent_response = await self._process_insurance_query(customer_text)
                    print(f"Agent: {agent_response}")
                    
                    # Send response back
                    await websocket.send(json.dumps({
                        "type": "text.generate",
                        "text": agent_response
                    }))
                
                elif response["type"] == "response.audio.delta":
                    if "delta" in response:
                        try:
                            audio_data = response["delta"].strip()
                            padding = -len(audio_data) % 4
                            if padding:
                                audio_data += "=" * padding
                            
                            audio = np.frombuffer(
                                base64.b64decode(audio_data), 
                                dtype=np.int16
                            )
                            self.streams['output'].write(audio)
                            
                        except Exception as e:
                            print(f"Audio processing error: {e}")
                            
                elif response["type"] == "response.done":
                    break
                    
        finally:
            self.audio_processor.is_speaking = False

    async def _process_insurance_query(self, customer_text: str) -> str:
        """Process customer input based on current conversation state"""
        state = self.conversation_state
        
        # Look for customer ID in text
        id_match = re.search(r'\b\d{5}\b', customer_text)
        
        if state.current_state == "greeting":
            if any(word in customer_text for word in ["coverage", "policy", "insurance", "check"]):
                state.update_state("need_id")
                return ("I'll be happy to help you check your coverage. Could you please provide "
                       "your customer ID number? It's the 5-digit number on your insurance card.")
            return ("Hello, this is Alex from AtlasMedical Insurance. How may I assist you "
                   "with your insurance coverage today?")
        
        elif state.current_state == "need_id":
            if id_match:
                state.customer_id = id_match.group(0)
                state.update_state("have_id")
                return ("Thank you for providing your ID number. What specific coverage "
                       "information would you like to check? For example, you can ask about "
                       "specialist visits or specific procedures.")
            return ("I apologize, but I need your 5-digit customer ID number to check your "
                   "coverage details. Could you please provide that?")
        
        elif state.current_state == "have_id":
            # Here we would check the policy document
            policy_path = f"policy_documents/insurance_policy_{state.customer_id}.txt"
            try:
                with open(policy_path, 'r') as file:
                    policy_content = file.read()
                    if "cardiologist" in customer_text.lower():
                        return ("Based on your policy, I can see your cardiologist visits are "
                               "covered. Let me check the specific details... [Policy details "
                               "would be provided here]")
                    return ("I can help you with that coverage question. What specific aspect "
                           "would you like to know about?")
            except FileNotFoundError:
                return ("I'm having trouble accessing your policy information. Could you "
                       "please verify your customer ID?")

    async def run(self):
        """Main execution loop"""
        try:
            print("\nStarting AtlasMedical Insurance Assistant...")
            await self.setup_audio()
            
            print("Connecting to service...")
            async with websockets.connect(self.url) as ws:
                await self.setup_websocket_session(ws)
                print("\n=== AtlasMedical Insurance Assistant Ready ===")
                
                while True:
                    if self.audio_processor.should_process():
                        audio_data = self.audio_processor.reset()
                        await self.send_audio(ws, audio_data)
                        await self.handle_response(ws)
                    await asyncio.sleep(0.05)
                    
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"\nError in main loop: {e}")
        finally:
            for stream in self.streams.values():
                if stream:
                    stream.stop()
                    stream.close()

if __name__ == "__main__":
    print("\n=== Initializing AtlasMedical Insurance Voice Assistant ===")
    system = InsuranceConversationSystem()
    asyncio.run(system.run())