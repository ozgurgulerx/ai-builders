import asyncio
import os
import base64
import json
from dotenv import load_dotenv
import websockets
import numpy as np
import sounddevice as sd

# Load environment variables
load_dotenv()

async def main():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: AZURE_OPENAI_API_KEY is not set.")
        return

    url = (
        f"wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?"
        "api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&api-key="
        f"{api_key}"
    )

    # Set up audio output stream
    try:
        stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
        stream.start()
        print("Audio stream started")
    except Exception as e:
        print(f"Error initializing audio stream: {e}")
        return

    try:
        async with websockets.connect(url) as ws:
            print("Connected to the API WebSocket")

            # Step 1: Send session update
            session_payload = {
                "type": "session.update",
                "session": {
                    "voice": "alloy",
                    "instructions": "Tell a brief, engaging story about a curious robot lost in a busy city. Keep it under 30 seconds.",
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
            print("Session update sent")

            # Wait for session.created
            while True:
                response = await ws.recv()
                data = json.loads(response)
                print(f"Session response: {json.dumps(data, indent=2)}")
                if data.get("type") == "session.created":
                    break
                elif data.get("type") == "error":
                    print("Error creating session")
                    return

            # Step 2: Send user message
            message_payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Speak now."}]
                }
            }
            await ws.send(json.dumps(message_payload))
            print("User message sent")

            # Wait for conversation.item.created
            while True:
                response = await ws.recv()
                data = json.loads(response)
                print(f"Message response: {json.dumps(data, indent=2)}")
                if data.get("type") == "conversation.item.created":
                    break
                elif data.get("type") == "error":
                    print("Error creating message")
                    return

            # Step 3: Request response
            response_payload = {
                "type": "response.create",
                "response": {"modalities": ["audio", "text"]}
            }
            await ws.send(json.dumps(response_payload))
            print("Response requested")

            # Step 4: Stream audio response
            print("Streaming audio...")
            audio_buffer = []
            while True:
                try:
                    response = await ws.recv()
                    data = json.loads(response)
                    
                    if data["type"] == "response.audio.delta":
                        audio_data = data.get("delta", "")
                        if audio_data:
                            try:
                                # Remove any non-base64 characters and pad if necessary
                                audio_data = audio_data.replace(" ", "").replace("\n", "")
                                padding = 4 - (len(audio_data) % 4)
                                if padding != 4:
                                    audio_data += "=" * padding
                                
                                # Decode and play audio
                                audio_bytes = base64.b64decode(audio_data)
                                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                                stream.write(audio)
                                print(".", end="", flush=True)
                            except Exception as decode_error:
                                print(f"\nError decoding audio: {decode_error}")
                        
                    elif data["type"] == "response.done":
                        print("\nAudio streaming completed")
                        break
                    elif data["type"] == "error":
                        print(f"Error response received: {json.dumps(data, indent=2)}")
                        break

                except Exception as e:
                    print(f"Error during audio streaming: {e}")
                    break

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"WebSocket connection failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up audio stream
        await asyncio.sleep(1)  # Allow final audio to play
        stream.stop()
        stream.close()
        print("Audio stream closed")

if __name__ == "__main__":
    print("Starting real-time API test...")
    asyncio.run(main())