import numpy as np
import sounddevice as sd

# Generate a simple sine wave
duration = 1.0  # seconds
frequency = 440  # Hz
samples = np.arange(duration * 44100) / 44100.0
audio_data = np.sin(2 * np.pi * frequency * samples)
audio_int16 = (audio_data * 32767).astype(np.int16)

print("Available audio devices:")
print(sd.query_devices())
print("\nDefault device:", sd.default.device)

print("\nPlaying test tone...")
sd.play(audio_int16, 44100)
sd.wait()  # Wait until file is done playing
print("Done!")
