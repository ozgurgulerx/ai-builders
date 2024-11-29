import websockets
import asyncio
import os
import base64
import json
from dotenv import load_dotenv
import numpy as np
import soundfile as sf  # Change to soundfile for better audio handling

# Load environment variables
load_dotenv()

# Retrieve API key and WebSocket URL
api_key = os.getenv("AZURE_OPENAI_API_KEY")
wss_url = f"wss://aoai-ep-swedencentral02.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview&api-key={api_key}"

if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY not found. Ensure it is set in the .env file.")

# Folder to save audio responses
AUDIO_OUTPUT_FOLDER = "audio_responses"
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)

def save_audio_response(audio_data, filename="response.wav", sample_rate=24000):
    filepath = os.path.join(AUDIO_OUTPUT_FOLDER, filename)
    try:
        # Convert to numpy array and ensure int16 dtype
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sf.write(filepath, audio_array, sample_rate, subtype='PCM_16')
        
        file_size = os.path.getsize(filepath)
        print(f"Audio saved to {filepath} (size: {file_size} bytes)")
        return True
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return False

async def interact_with_api():
    audio_chunks = []  # Changed to list for collecting chunks
    complete_text = ""
    
    try:
        async with websockets.connect(wss_url) as websocket:
            print("Connected to the realtime endpoint")

            # Step 1: Initialize session
            session_update = {
                "type": "session.update",
                "session": {
                    "voice": "alloy",
                    "instructions": "Respond briefly and to the point. Provide a concise and engaging response.",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {"type": "none"},
                    "temperature": 0.7,
                    "max_response_output_tokens": 50,
                }
            }
            await websocket.send(json.dumps(session_update))
            print("Session update sent")

            # Handle session initialization
            session_created = False
            while not session_created:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Session response: {data}")
                
                if data.get("type") == "session.created":
                    session_created = True
                    session_id = data.get("session", {}).get("id")
                    print(f"Session created: {session_id}")
                elif data.get("type") == "error":
                    raise Exception(f"Session error: {data}")

            # Step 2: Create conversation item
            conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Tell me an interesting fact about space!"
                        }
                    ],
                }
            }
            await websocket.send(json.dumps(conversation_item))
            print("Conversation item sent")

            # Wait for conversation item creation confirmation
            item_created = False
            while not item_created:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Conversation response: {data}")
                if data.get("type") == "conversation.item.created":
                    item_created = True
                    print("Conversation item created")
                elif data.get("type") == "error":
                    raise Exception(f"Conversation error: {data}")

            # Step 3: Request response
            response_create = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"]
                }
            }
            await websocket.send(json.dumps(response_create))
            print("Response requested")

            # Process response stream
            response_started = False
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    print(f"Stream message: {data.get('type')}")
                    
                    if data.get("type") == "response.created":
                        response_started = True
                        print("Response creation confirmed")
                        continue

                    if not response_started:
                        continue

                    if data.get("type") == "response.audio.delta":
                        raw_data = data.get("data", "")
                        print(f"Received audio data chunk of length: {len(raw_data)}")
                        if raw_data:
                            audio_data = base64.b64decode(raw_data)
                            audio_chunks.append(audio_data)  # Append chunks instead of extending
                            print(f"Processed audio chunk: {len(audio_data)} bytes")
                    
                    elif data.get("type") == "response.text.delta":
                        text_delta = data.get("data", "")
                        complete_text += text_delta
                        print(f"Text received: {text_delta}", end="", flush=True)
                    
                    elif data.get("type") in ["response.audio.done", "response.done"]:
                        print("\nResponse complete")
                        break
                    
                    elif data.get("type") == "error":
                        print(f"Error in stream: {data}")
                        break

                except Exception as e:
                    print(f"Error processing stream message: {e}")
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if audio_chunks:
            # Combine all chunks into one audio stream
            complete_audio = b''.join(audio_chunks)
            if len(complete_audio) > 0:
                print(f"\nSaving audio response ({len(complete_audio)} bytes)")
                save_audio_response(complete_audio)
            print(f"Complete text response: {complete_text}")
        else:
            print("No audio data received")
            print(f"Final text response: {complete_text}")

if __name__ == "__main__":
    asyncio.run(interact_with_api())