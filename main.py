from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
from dotenv import load_dotenv
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions, ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice.types import UserInput
from hume import MicrophoneInterface, Stream
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import base64
from starlette.websockets import WebSocketDisconnect
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import whisper
import tempfile
import wave
import numpy as np
from hume import MicrophoneInterface, Stream

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add secret key check
HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

if not all([HUME_API_KEY, HUME_SECRET_KEY, HUGGINGFACE_API_TOKEN]):
    raise ValueError("Missing required environment variables. Please check .env file.")

# Initialize Hume client with both keys
client = AsyncHumeClient(api_key=HUME_API_KEY)

CHROMA_PATH = "chroma"

# Initialize embeddings - using the same config from create_database.py
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'mps'}
)

# Initialize the DB - using the same config from query_data.py
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Check for HuggingFace token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Missing HUGGINGFACE_API_TOKEN in environment variables")


llm = HuggingFaceHub(
    repo_id="google/flan-t5-base", 
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    model_kwargs={
        "temperature": 0.7,
        "max_length": 769,
        "top_p": 0.9
    }
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class TranscriptionRequest(BaseModel):
    audio_data: str
    query: Optional[str] = None

# class WebSocketHandler:
#     def __init__(self):
#         self.socket = None
#         self.byte_strs = Stream.new()
#         # self.audio_complete = asyncio.Event()
#         # self.collected_audio = bytearray()

#     def set_socket(self, socket: ChatWebsocketConnection):
#         self.socket = socket
    
#     async def on_open(self):
#         print("WebSocket connection opened")

#     async def on_message(self, message: SubscribeEvent):
#         print(f"Received message: {message}")
#         if message.type == "audio_output":
#             print("Received audio output")
#             message_str: str = message.data
#             message_bytes = base64.b64decode(message_str.encode("utf-8"))
#             await self.byte_strs.put(message_bytes)
#             return
        
#         # elif message.type == "complete":
#         #     self.audio_complete.set()
#         #     await self.byte_strs.complete()
#         # return

#     async def on_close(self):
#         print("WebSocket connection closed")

#     async def on_error(self, error: Exception):
#         print(f"Error occurred: {str(error)}")

# class WebSocketHandler:
#     def __init__(self):
#         """Construct the WebSocketHandler, initially assigning the socket to None and the byte stream to a new Stream object."""
#         self.socket = None
#         self.byte_strs = Stream.new()

#     def set_socket(self, socket: ChatWebsocketConnection):
#         """Set the socket."""
#         self.socket = socket

#     async def on_message(self, message: SubscribeEvent):
#         """Callback function to handle a WebSocket message event."""
#         print(f"Received message: {message}")
#         if message.type == "audio_output":
#             message_str: str = message.data
#             message_bytes = base64.b64decode(message_str.encode("utf-8"))
#             await self.byte_strs.put(message_bytes)
#             return


# class WebSocketHandler:
#     def __init__(self):
#         self.socket = None
#         self.byte_strs = Stream.new()

#     def set_socket(self, socket: ChatWebsocketConnection):
#         self.socket = socket
    
#     async def on_open(self):
#         print("WebSocket connection opened")

#     async def on_message(self, message: SubscribeEvent):
#         if message.type == "audio_output":
#             message_str: str = message.data
#             message_bytes = base64.b64decode(message_str.encode("utf-8"))
#             await self.byte_strs.put(message_bytes)
#             return

#     async def on_close(self):
#         print("WebSocket connection closed")

#     async def on_error(self, error):
#         print(f"Error: {error}")

class WebSocketHandler:
    def __init__(self):
        """Construct the WebSocketHandler, initially assigning the socket to None and the byte stream to a new Stream object."""
        self.socket = None
        self.byte_strs = Stream.new()

    def set_socket(self, socket: ChatWebsocketConnection):
        """Set the socket."""
        self.socket = socket

    async def on_open(self):
        """Logic invoked when the WebSocket connection is opened."""
        print("WebSocket connection opened.")

    async def on_message(self, message: SubscribeEvent):
        """Callback function to handle a WebSocket message event."""
        if message.type == "audio_output":
            message_str: str = message.data
            message_bytes = base64.b64decode(message_str.encode("utf-8"))
            await self.byte_strs.put(message_bytes)
            return

    async def on_close(self):
        """Logic invoked when the WebSocket connection is closed."""
        print("WebSocket connection closed.")

    async def on_error(self, error):
        """Logic invoked when an error occurs in the WebSocket connection."""
        print(f"Error: {error}")

# Initialize Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

@app.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    
    # Initialize conversation memory and RAG chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are a helpful assistant that provides accurate information based solely on the context provided. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know. Don't halucinate or give false information.

                Context: {context}

                Question: {question}

                Helpful Answer: """,
                # template
                input_variables=["context", "question"]
            )
        }
    )
    
    try:
        while True:
            try:
                # Receive audio data from client
                audio_data = await websocket.receive_bytes()
                
                # Convert bytes to numpy array (ensure proper format)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Normalize sample rate and channels
                sample_rate = 44100
                channels = 1
                
                # Save as WAV file with proper parameters
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    with wave.open(temp_audio.name, 'wb') as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(2)  # 16-bit audio
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_np.tobytes())
                    temp_audio_path = temp_audio.name
                
                print(f"Saved audio to temporary file: {temp_audio_path}")
                
                # Convert speech to text using Whisper
                try:
                    result = whisper_model.transcribe(temp_audio_path)
                    user_text = result["text"]
                    print(f"Transcribed text: {user_text}")
                except Exception as e:
                    print(f"Whisper transcription error: {e}")
                    continue
                
                # Delete temporary audio file
                os.unlink(temp_audio_path)
                
                # Get response from RAG system
                response = await asyncio.to_thread(
                    lambda: qa_chain({"question": user_text})
                )
                
                if isinstance(response, dict) and 'answer' in response:
                    answer = response['answer']
                    print(f"RAG response: {answer}")
                else:
                    answer = "I apologize, but I couldn't generate a proper response."
                    print(f"Unexpected response format: {response}")
                
                # Convert text response to speech using Hume
                try:
                    websocket_handler = WebSocketHandler()
                    options = ChatConnectOptions(config_id=HUME_CONFIG_ID, secret_key=HUME_SECRET_KEY)
                    
                    async with client.empathic_voice.chat.connect_with_callbacks(
                        options=options,
                        on_open=websocket_handler.on_open,
                        on_message=websocket_handler.on_message,
                        on_close=websocket_handler.on_close,
                        on_error=websocket_handler.on_error
                    ) as socket:
                        websocket_handler.set_socket(socket)
                        
                        # Send text to be converted to speech
                        user_input = UserInput(text=answer)
                        await socket.send_user_input(user_input)
                        
                        # Create microphone task for audio handling
                        microphone_task = asyncio.create_task(
                            MicrophoneInterface.start(
                                socket,
                                byte_stream=websocket_handler.byte_strs
                            )
                        )
                        
                        # Wait for audio processing to complete
                        await microphone_task

                except Exception as e:
                    print(f"Error in text-to-speech: {e}")
                    # Fallback: send text response if audio fails
                    await websocket.send_text(answer)
                
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Error processing request: {e}")
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass

@app.on_event("startup")
async def startup_event():
    # Verify the database exists and is populated
    collection = db._collection
    print(f"Total documents in database: {collection.count()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
