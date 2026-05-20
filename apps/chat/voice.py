import uuid
import keyboard
import io
import numpy as np
import soundfile as sf
import sounddevice as sd
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage
from .chatbot.agent import create_chat_agent

client = OpenAI()
fs = 16000

def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    chatbot = create_chat_agent()
    
    recording = []
    is_recording = False

    def callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())

    print("Press and hold SPACE to record, release to transcribe and send. Press 'q' to quit.")
    
    with sd.InputStream(callback=callback, samplerate=fs, channels=1):
        while True:
            if keyboard.is_pressed("q"):
                break
            
            if keyboard.is_pressed("space") and not is_recording:
                print("Recording...")
                is_recording = True
                recording.clear()
            elif not keyboard.is_pressed("space") and is_recording:
                is_recording = False
                if recording:
                    print("Processing audio...")
                    audio = np.concatenate(recording, axis=0)
                    audio_bytes_io = io.BytesIO()
                    sf.write(audio_bytes_io, audio, fs, format="wav")
                    audio_bytes_io.seek(0)
                    audio_bytes_io.name = "audio.wav"  # Give the BytesIO object a name with extension
                    
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_bytes_io
                    )
                    
                    user_text = transcript.text
                    print(f"You said: {user_text}")
                    
                    if user_text.strip():
                        print("Blossom: ", end="", flush=True)
                        for chunk in chatbot.stream(
                            {"messages": [HumanMessage(content=user_text)]}, 
                            config=config,
                            stream_mode="updates"
                        ):
                            if "chatbot" in chunk:
                                messages = chunk["chatbot"].get("messages", [])
                                for msg in messages:
                                    if isinstance(msg, AIMessage):
                                        ai_response = msg.content
                                        print(ai_response, end="\n", flush=True)


if __name__ == "__main__":
    main()