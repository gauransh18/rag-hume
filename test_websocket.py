import asyncio
import websockets
import sounddevice as sd
import numpy as np
from hume import MicrophoneInterface, Stream

async def test_websocket():
    uri = "ws://localhost:8000/ws/stream"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Configure audio parameters
            duration = 5
            sample_rate = 44100
            channels = 1
            
            # Record audio with proper configuration
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype=np.int16,
                blocking=True  # Wait for recording to complete
            )
            print("Recording complete!")
            
            # Convert to mono if needed
            if channels > 1:
                recording = recording.mean(axis=1).astype(np.int16)
            
            # Send audio
            audio_bytes = recording.tobytes()
            await websocket.send(audio_bytes)
            print("Sent audio, waiting for response...")
            # Create stream for audio playback
            byte_stream = Stream.new()
            playback_task = None
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                
                if isinstance(response, bytes):
                    print("Received audio response, playing...")
                    # Configure audio playback
                    sd.default.samplerate = sample_rate
                    sd.default.channels = channels
                    
                    # Start playback interface
                    playback_task = asyncio.create_task(
                        MicrophoneInterface.play_audio(
                            byte_stream=byte_stream
                        )
                    )
                    
                    # Feed audio data to stream
                    await byte_stream.put(response)
                    await byte_stream.complete()
                    
                    # Wait for playback to complete
                    if playback_task:
                        await playback_task
                else:
                    print(f"Received text response: {response}")
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            finally:
                if playback_task and not playback_task.done():
                    playback_task.cancel()
                    
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

