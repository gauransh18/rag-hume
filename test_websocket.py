import asyncio
import websockets
import sounddevice as sd
import numpy as np
import wave

async def test_websocket():
    uri = "ws://localhost:8000/ws/stream"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            print("\nRecording... (speak for 5 seconds)")
            print("Ask a question about your documents!")
            
            # Record audio
            duration = 5
            sample_rate = 44100
            channels = 1
            
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype=np.int16
            )
            sd.wait()
            print("Recording complete!")
            
            # Ensure audio is mono
            if channels > 1:
                recording = recording.mean(axis=1)
            
            # Send audio
            audio_bytes = recording.tobytes()
            print(f"Sending {len(audio_bytes)} bytes of audio data")
            await websocket.send(audio_bytes)
            print("Waiting for response...")
            
            # Handle response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                
                if isinstance(response, bytes):
                    print("Received audio response, playing...")
                    audio_data = np.frombuffer(response, dtype=np.int16)
                    audio_data = np.int16(audio_data * 0.5)  # Reduce volume
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    print("Playback complete!")
                else:
                    print(f"Received text response: {response}")
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            
            print("Closing connection...")
            await asyncio.sleep(1)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting WebSocket test client...")
    print("This will record 5 seconds of audio and send it to the server.")
    print("Press Enter to begin...")
    input()
    asyncio.run(test_websocket())