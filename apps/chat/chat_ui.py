import uuid
import io
import os
import threading
import queue
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

import numpy as np
import soundfile as sf
import sounddevice as sd
from openai import OpenAI

from langchain_core.messages import AIMessage, HumanMessage

from .chatbot.agent import create_chat_agent
from .emotion_agent import ConversationEmotionDetector
from apps.shared.emotion_state import save_conversation_emotion


FS = 16000


def get_project_root() -> Path:
    """
    Current file:
        HRIBlossom/apps/chat/chat_ui.py

    Project root:
        HRIBlossom/
    """
    return Path(__file__).resolve().parents[2]


def load_project_env():
    """
    Minimal .env loader so this works even if python-dotenv is not installed.
    """
    env_path = get_project_root() / ".env"

    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value

    except Exception:
        pass


def has_openai_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def create_openai_client():
    load_project_env()

    if not has_openai_api_key():
        return None

    try:
        return OpenAI()
    except Exception:
        return None


class BlossomChatApp:
    def __init__(self, root):
        load_project_env()

        self.root = root
        self.root.title("Blossom Chat")
        self.root.geometry("850x620")

        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

        self.client = create_openai_client()

        self.chatbot = create_chat_agent()
        self.emotion_detector = ConversationEmotionDetector(prefer_autogen=True)

        self.message_queue = queue.Queue()

        self.recording = []
        self.is_recording = False
        self.is_processing = False

        self.chat_visible = False
        self.last_detected_emotion = None

        self.setup_ui()
        self.setup_audio_stream()
        self.setup_key_bindings()

        self.root.after(100, self.process_ui_queue)

        self.append_system_message("Hold SPACE to talk.")
        self.append_system_message("Press ENTER to open the chat box.")
        self.append_system_message("Press ESC to close the chat box.")
        self.append_system_message("When chat is open, type and press ENTER to send.")
        self.append_system_message("User emotion is detected as happy, sad, or angry.")
        self.append_system_message("Detected conversation emotion is shared with combined recognition.")
        self.append_system_message("Emotion detection uses AutoGen when available.")
        self.append_system_message("This UI does not play TTS audio directly.")
        self.append_system_message("Press Q to quit when not typing.")

        if has_openai_api_key():
            self.append_system_message("OPENAI_API_KEY loaded.")
        else:
            self.append_system_message(
                "OPENAI_API_KEY not found. Voice transcription and OpenAI-based emotion detection will not work."
            )

        if self.emotion_detector.autogen_available:
            self.append_system_message("AutoGen emotion detector loaded.")
        else:
            self.append_system_message("AutoGen not available. Using OpenAI/keyword fallback.")

    # -----------------------------
    # UI setup
    # -----------------------------

    def setup_ui(self):
        self.status_label = tk.Label(
            self.root,
            text="Ready. Hold SPACE to talk. Press ENTER to open chat.",
            font=("Arial", 14),
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.emotion_label = tk.Label(
            self.root,
            text="Detected emotion: none",
            font=("Arial", 14),
            anchor="w",
        )
        self.emotion_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.main_label = tk.Label(
            self.root,
            text="Blossom is listening when SPACE is held.",
            font=("Arial", 18),
            wraplength=790,
            justify="center",
        )
        self.main_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.chat_frame = tk.Frame(self.root)

        self.chat_history = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            height=18,
            state=tk.DISABLED,
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        self.input_frame = tk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.chat_input = tk.Entry(
            self.input_frame,
            font=("Arial", 12),
        )
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            command=self.send_typed_message,
        )
        self.send_button.pack(side=tk.RIGHT, padx=(8, 0))

        self.chat_frame.pack_forget()

    # -----------------------------
    # Audio recording setup
    # -----------------------------

    def setup_audio_stream(self):
        self.audio_stream = sd.InputStream(
            callback=self.audio_callback,
            samplerate=FS,
            channels=1,
        )
        self.audio_stream.start()

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.recording.append(indata.copy())

    # -----------------------------
    # Keyboard setup
    # -----------------------------

    def setup_key_bindings(self):
        self.root.bind("<KeyPress-space>", self.on_space_press)
        self.root.bind("<KeyRelease-space>", self.on_space_release)
        self.root.bind("<Return>", self.on_enter_press)
        self.root.bind("<Escape>", self.on_escape_press)
        self.root.bind("<KeyPress-q>", self.on_q_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.chat_input.bind("<Return>", self.on_chat_input_enter)

    def on_space_press(self, event=None):
        focused_widget = self.root.focus_get()

        if focused_widget == self.chat_input:
            return None

        if self.is_processing:
            return "break"

        if not self.is_recording:
            self.is_recording = True
            self.recording.clear()
            self.set_status("Recording... release SPACE to send.")
            self.main_label.config(text="Recording...")

        return "break"

    def on_space_release(self, event=None):
        focused_widget = self.root.focus_get()

        if focused_widget == self.chat_input:
            return None

        if self.is_recording:
            self.is_recording = False

            if self.recording:
                self.set_status("Processing audio...")
                self.main_label.config(text="Processing audio...")

                worker = threading.Thread(
                    target=self.process_voice_message,
                    daemon=True,
                )
                worker.start()

        return "break"

    def on_enter_press(self, event=None):
        focused_widget = self.root.focus_get()

        if focused_widget == self.chat_input:
            self.send_typed_message()
            return "break"

        self.show_chat_box()
        return "break"

    def on_chat_input_enter(self, event=None):
        self.send_typed_message()
        return "break"

    def on_escape_press(self, event=None):
        self.hide_chat_box()
        return "break"

    def on_q_press(self, event=None):
        focused_widget = self.root.focus_get()

        if focused_widget == self.chat_input:
            return None

        self.on_close()
        return "break"

    # -----------------------------
    # Chat box visibility
    # -----------------------------

    def show_chat_box(self):
        if self.chat_visible:
            self.chat_input.focus_set()
            return

        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_visible = True
        self.chat_input.focus_set()
        self.set_status("Chat open. Type and press ENTER to send. Press ESC to close.")

    def hide_chat_box(self):
        if not self.chat_visible:
            return

        self.chat_frame.pack_forget()
        self.chat_visible = False
        self.root.focus_set()
        self.set_status("Chat hidden. Hold SPACE to talk. Press ENTER to open chat.")

    # -----------------------------
    # Typed messages
    # -----------------------------

    def send_typed_message(self):
        user_text = self.chat_input.get().strip()

        if not user_text:
            return

        if self.is_processing:
            self.append_system_message("Still processing previous message.")
            return

        self.chat_input.delete(0, tk.END)

        worker = threading.Thread(
            target=self.process_text_message,
            args=(user_text,),
            daemon=True,
        )
        worker.start()

    def process_text_message(self, user_text):
        if self.is_processing:
            self.queue_system_message("Still processing previous message.")
            return

        self.is_processing = True
        self.queue_status("Detecting emotion...")
        self.queue_main_text("Detecting emotion...")

        try:
            detected_emotion = self.emotion_detector.detect(user_text)

            save_conversation_emotion(
                emotion=detected_emotion,
                source="typed",
                text=user_text,
                confidence=1.0,
            )

            self.queue_user_message(
                text=user_text,
                source="typed",
                emotion=detected_emotion,
            )

            self.queue_detected_emotion(detected_emotion)

            self.queue_status("Thinking...")
            self.queue_main_text(
                f"Detected user emotion: {detected_emotion}\nThinking..."
            )

            ai_response = self.get_chatbot_response(user_text)

            self.queue_ai_message(ai_response, source="typed")
            self.queue_status("Ready.")
            self.queue_main_text("Ready. Hold SPACE to talk or type in chat.")

        except Exception as e:
            self.queue_system_message(f"Text chat error: {e}")
            self.queue_status("Error.")
            self.queue_main_text("Text chat error.")

        finally:
            self.is_processing = False

    # -----------------------------
    # Voice messages
    # -----------------------------

    def process_voice_message(self):
        if self.is_processing:
            self.queue_system_message("Still processing previous message.")
            return

        if self.client is None:
            self.queue_system_message(
                "Voice transcription requires OPENAI_API_KEY. Check your .env file."
            )
            self.queue_status("Missing OPENAI_API_KEY.")
            self.queue_main_text("Missing OPENAI_API_KEY.")
            return

        self.is_processing = True

        try:
            audio = np.concatenate(self.recording, axis=0)

            audio_bytes_io = io.BytesIO()
            sf.write(audio_bytes_io, audio, FS, format="wav")
            audio_bytes_io.seek(0)
            audio_bytes_io.name = "audio.wav"

            self.queue_status("Transcribing...")
            self.queue_main_text("Transcribing...")

            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_bytes_io,
            )

            user_text = transcript.text.strip()

            if not user_text:
                self.queue_status("No speech detected.")
                self.queue_main_text("No speech detected.")
                return

            self.queue_status("Detecting emotion...")
            self.queue_main_text(f"You said: {user_text}\nDetecting emotion...")

            detected_emotion = self.emotion_detector.detect(user_text)

            save_conversation_emotion(
                emotion=detected_emotion,
                source="spoken",
                text=user_text,
                confidence=1.0,
            )

            self.queue_user_message(
                text=user_text,
                source="spoken",
                emotion=detected_emotion,
            )

            self.queue_detected_emotion(detected_emotion)

            self.queue_status("Thinking...")
            self.queue_main_text(
                f"You said: {user_text}\n"
                f"Detected user emotion: {detected_emotion}\n"
                "Thinking..."
            )

            ai_response = self.get_chatbot_response(user_text)

            self.queue_ai_message(ai_response, source="voice")

            self.queue_status("Ready.")
            self.queue_main_text("Ready. Hold SPACE to talk.")

        except Exception as e:
            self.queue_system_message(f"Voice chat error: {e}")
            self.queue_status("Error.")
            self.queue_main_text("Voice chat error.")

        finally:
            self.is_processing = False

    # -----------------------------
    # Chatbot response
    # -----------------------------

    def get_chatbot_response(self, user_text):
        ai_response = ""

        for chunk in self.chatbot.stream(
            {"messages": [HumanMessage(content=user_text)]},
            config=self.config,
            stream_mode="updates",
        ):
            if "chatbot" in chunk:
                messages = chunk["chatbot"].get("messages", [])

                for msg in messages:
                    if isinstance(msg, AIMessage):
                        ai_response = msg.content

        return ai_response

    # -----------------------------
    # Thread-safe UI queue helpers
    # -----------------------------

    def queue_status(self, text):
        self.message_queue.put(("status", text))

    def queue_main_text(self, text):
        self.message_queue.put(("main", text))

    def queue_user_message(self, text, source, emotion):
        self.message_queue.put(("user", text, source, emotion))

    def queue_ai_message(self, text, source):
        self.message_queue.put(("ai", text, source))

    def queue_system_message(self, text):
        self.message_queue.put(("system", text))

    def queue_detected_emotion(self, emotion):
        self.message_queue.put(("emotion", emotion))

    def process_ui_queue(self):
        while not self.message_queue.empty():
            item = self.message_queue.get()
            event_type = item[0]

            if event_type == "status":
                self.set_status(item[1])

            elif event_type == "main":
                self.main_label.config(text=item[1])

            elif event_type == "emotion":
                self.set_detected_emotion(item[1])

            elif event_type == "user":
                _, text, source, emotion = item
                self.append_user_message(text, source=source, emotion=emotion)

            elif event_type == "ai":
                _, text, source = item
                self.append_ai_message(text, source=source)

            elif event_type == "system":
                self.append_system_message(item[1])

        self.root.after(100, self.process_ui_queue)

    # -----------------------------
    # Direct UI helpers
    # -----------------------------

    def set_status(self, text):
        self.status_label.config(text=text)

    def set_detected_emotion(self, emotion):
        self.last_detected_emotion = emotion
        self.emotion_label.config(text=f"Detected emotion: {emotion}")

    def append_to_chat(self, text):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, text + "\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def append_user_message(self, text, source, emotion):
        if source == "spoken":
            label = "You spoke"
        else:
            label = "You typed"

        self.append_to_chat(f"{label} [{emotion}]: {text}")

    def append_ai_message(self, text, source):
        if source == "voice":
            label = "Blossom response"
        else:
            label = "Blossom"

        self.append_to_chat(f"{label}: {text}")
        self.append_to_chat("")

    def append_system_message(self, text):
        self.append_to_chat(f"[System] {text}")

    # -----------------------------
    # Shutdown
    # -----------------------------

    def on_close(self):
        try:
            if hasattr(self, "audio_stream"):
                self.audio_stream.stop()
                self.audio_stream.close()
        except Exception:
            pass

        try:
            sd.stop()
        except Exception:
            pass

        self.root.destroy()


def main():
    root = tk.Tk()
    BlossomChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()