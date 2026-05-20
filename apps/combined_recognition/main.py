"""
Single-file combined Blossom emotion system.

This app combines:

1. Facial emotion recognition
2. Gesture recognition
3. Conversation emotion detection from voice or typed chat

Controls:
    Hold SPACE       -> record voice
    Release SPACE    -> transcribe, send to chatbot, detect conversation emotion
    Press ENTER      -> open text chat popup
    Press q          -> quit camera window

The camera window stays visible the whole time.

The chat popup:
    - Opens when ENTER is pressed
    - Lets the user type messages
    - Can be closed
    - Can be reopened
    - Stores typed messages, spoken messages, and Blossom responses
    - Shares conversation emotion with the camera fusion system

Final emotion fusion works with any available sources:
    face only
    hand only
    conversation only
    face + hand
    face + conversation
    hand + conversation
    face + hand + conversation
"""

import cv2
import io
import os
import queue
import threading
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext

import keyboard
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from langchain_core.messages import AIMessage, HumanMessage

from apps.chat.chatbot.agent import create_chat_agent
from apps.chat.emotion_agent import ConversationEmotionDetector
from apps.shared.emotion_fusion import EmotionFusionClassifier
from apps.shared.emotion_state import save_conversation_emotion
from apps.shared.utils.sequence import get_sequence_by_name
from apps.shared.constants import get_blossom_robot


# -----------------------------
# General config
# -----------------------------

CAMERA_INDEX = 0
FS = 16000

CONVERSATION_EMOTION_MAX_AGE_SECONDS = 60.0
FINAL_CONFIDENCE_THRESHOLD = 0.70
ROBOT_SEQUENCE_COOLDOWN_SECONDS = 5.0

SOURCE_WEIGHTS = {
    "facial": 1.0,
    "gesture": 1.0,
    "conversation": 1.0,
}

FINAL_EMOTION_TO_SEQUENCE = {
    "happy": "happy_1",
    "angry": "anger",
    "sad": "sad_1",
}


# -----------------------------
# MediaPipe setup
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# -----------------------------
# Environment / OpenAI helpers
# -----------------------------

def get_project_root() -> Path:
    """
    Current file:
        HRIBlossom/apps/combined_recognition/main.py

    Project root:
        HRIBlossom/
    """
    return Path(__file__).resolve().parents[2]


def load_project_env():
    """
    Minimal .env loader.

    Expected:
        HRIBlossom/.env

    Example:
        OPENAI_API_KEY=sk-...
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


# -----------------------------
# Shared app state
# -----------------------------

class SharedAppState:
    def __init__(self):
        self.lock = threading.RLock()

        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

        self.openai_client = create_openai_client()
        self.chatbot = create_chat_agent()
        self.emotion_detector = ConversationEmotionDetector(prefer_autogen=False)

        self.chat_history = []

        self.latest_conversation_emotion = None
        self.latest_conversation_text = ""
        self.latest_conversation_source = None
        self.latest_conversation_confidence = 1.0
        self.latest_conversation_timestamp = 0.0

        self.last_user_text = ""
        self.last_ai_response = ""

        self.is_chat_processing = False
        self.is_recording = False
        self.recording = []

        self.popup_visible = False
        self.popup = None

        self.status_text = "Ready."
        self.voice_status = "Hold SPACE to talk. Press ENTER for text chat."

    def add_chat_message(self, role: str, text: str, emotion: str = None):
        with self.lock:
            self.chat_history.append(
                {
                    "role": role,
                    "text": text,
                    "emotion": emotion,
                    "timestamp": time.time(),
                }
            )

    def get_chat_history_copy(self):
        with self.lock:
            return list(self.chat_history)

    def set_popup(self, popup):
        with self.lock:
            self.popup = popup
            self.popup_visible = popup is not None

    def get_popup(self):
        with self.lock:
            return self.popup

    def is_popup_visible(self):
        with self.lock:
            return self.popup_visible

    def set_status(self, text):
        with self.lock:
            self.status_text = text

    def get_status(self):
        with self.lock:
            return self.status_text

    def set_voice_status(self, text):
        with self.lock:
            self.voice_status = text

    def get_voice_status(self):
        with self.lock:
            return self.voice_status

    def set_latest_conversation_emotion(
        self,
        emotion: str,
        text: str,
        source: str,
        confidence: float = 1.0,
    ):
        now = time.time()

        with self.lock:
            self.latest_conversation_emotion = emotion
            self.latest_conversation_text = text
            self.latest_conversation_source = source
            self.latest_conversation_confidence = confidence
            self.latest_conversation_timestamp = now

        save_conversation_emotion(
            emotion=emotion,
            source=source,
            text=text,
            confidence=confidence,
        )

    def get_recent_conversation_state(self):
        with self.lock:
            if self.latest_conversation_emotion is None:
                return None

            age = time.time() - self.latest_conversation_timestamp

            if age > CONVERSATION_EMOTION_MAX_AGE_SECONDS:
                return None

            return {
                "emotion": self.latest_conversation_emotion,
                "text": self.latest_conversation_text,
                "source": self.latest_conversation_source,
                "confidence": self.latest_conversation_confidence,
                "timestamp": self.latest_conversation_timestamp,
                "age_seconds": age,
            }

    def set_last_user_text(self, text):
        with self.lock:
            self.last_user_text = text

    def set_last_ai_response(self, text):
        with self.lock:
            self.last_ai_response = text

    def get_last_user_text(self):
        with self.lock:
            return self.last_user_text

    def get_last_ai_response(self):
        with self.lock:
            return self.last_ai_response

    def try_start_processing(self):
        with self.lock:
            if self.is_chat_processing:
                return False

            self.is_chat_processing = True
            return True

    def stop_processing(self):
        with self.lock:
            self.is_chat_processing = False

    def is_processing(self):
        with self.lock:
            return self.is_chat_processing


# -----------------------------
# Text chat popup
# -----------------------------

class ChatPopup:
    def __init__(self, tk_root, shared_state: SharedAppState):
        self.tk_root = tk_root
        self.shared_state = shared_state
        self.ui_queue = queue.Queue()

        self.window = tk.Toplevel(self.tk_root)
        self.window.title("Blossom Text Chat")
        self.window.geometry("700x500")

        self.status_label = tk.Label(
            self.window,
            text="Type a message and press ENTER.",
            font=("Arial", 12),
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.emotion_label = tk.Label(
            self.window,
            text="Detected emotion: none",
            font=("Arial", 12),
            anchor="w",
        )
        self.emotion_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.chat_history = scrolledtext.ScrolledText(
            self.window,
            wrap=tk.WORD,
            font=("Arial", 11),
            state=tk.DISABLED,
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.input_frame = tk.Frame(self.window)
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

        self.chat_input.bind("<Return>", self.on_enter)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_existing_history()

        self.shared_state.set_popup(self)

        self.window.lift()
        self.window.focus_force()
        self.chat_input.focus_set()

        self.window.attributes("-topmost", True)
        self.window.after(300, lambda: self.window.attributes("-topmost", False))

        self.window.after(100, self.process_ui_queue)

    def load_existing_history(self):
        history = self.shared_state.get_chat_history_copy()

        for message in history:
            role = message.get("role", "")
            text = message.get("text", "")
            emotion = message.get("emotion")

            if role == "user_typed":
                self.append_to_chat(f"You typed [{emotion}]: {text}")

            elif role == "user_spoken":
                self.append_to_chat(f"You spoke [{emotion}]: {text}")

            elif role == "assistant":
                self.append_to_chat(f"Blossom: {text}")
                self.append_to_chat("")

            elif role == "system":
                self.append_to_chat(f"[System] {text}")

    def on_enter(self, event=None):
        self.send_typed_message()
        return "break"

    def send_typed_message(self):
        user_text = self.chat_input.get().strip()

        if not user_text:
            return

        self.chat_input.delete(0, tk.END)

        worker = threading.Thread(
            target=process_user_message,
            args=(self.shared_state, user_text, "typed"),
            daemon=True,
        )
        worker.start()

    def append_to_chat(self, text):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, text + "\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def queue_status(self, text):
        self.ui_queue.put(("status", text))

    def queue_emotion(self, emotion):
        self.ui_queue.put(("emotion", emotion))

    def queue_user_message(self, text, source, emotion):
        self.ui_queue.put(("user", text, source, emotion))

    def queue_ai_message(self, text):
        self.ui_queue.put(("ai", text))

    def queue_system_message(self, text):
        self.ui_queue.put(("system", text))

    def process_ui_queue(self):
        while not self.ui_queue.empty():
            item = self.ui_queue.get()
            event_type = item[0]

            if event_type == "status":
                self.status_label.config(text=item[1])

            elif event_type == "emotion":
                self.emotion_label.config(text=f"Detected emotion: {item[1]}")

            elif event_type == "user":
                _, text, source, emotion = item

                if source == "typed":
                    self.append_to_chat(f"You typed [{emotion}]: {text}")
                else:
                    self.append_to_chat(f"You spoke [{emotion}]: {text}")

            elif event_type == "ai":
                _, text = item
                self.append_to_chat(f"Blossom: {text}")
                self.append_to_chat("")

            elif event_type == "system":
                self.append_to_chat(f"[System] {item[1]}")

        if self.shared_state.get_popup() is self:
            self.window.after(100, self.process_ui_queue)

    def on_close(self):
        self.shared_state.set_popup(None)
        self.window.destroy()


def open_chat_popup(tk_root, shared_state: SharedAppState):
    popup = shared_state.get_popup()

    if popup is not None:
        try:
            popup.window.lift()
            popup.window.focus_force()
            popup.chat_input.focus_set()
        except Exception:
            pass

        return

    ChatPopup(tk_root, shared_state)


def notify_popup_status(shared_state: SharedAppState, text: str):
    popup = shared_state.get_popup()

    if popup is not None:
        popup.queue_status(text)


def notify_popup_emotion(shared_state: SharedAppState, emotion: str):
    popup = shared_state.get_popup()

    if popup is not None:
        popup.queue_emotion(emotion)


def notify_popup_user_message(
    shared_state: SharedAppState,
    text: str,
    source: str,
    emotion: str,
):
    popup = shared_state.get_popup()

    if popup is not None:
        popup.queue_user_message(text, source, emotion)


def notify_popup_ai_message(shared_state: SharedAppState, text: str):
    popup = shared_state.get_popup()

    if popup is not None:
        popup.queue_ai_message(text)


def notify_popup_system_message(shared_state: SharedAppState, text: str):
    popup = shared_state.get_popup()

    if popup is not None:
        popup.queue_system_message(text)


# -----------------------------
# Conversation processing
# -----------------------------

def get_chatbot_response(shared_state: SharedAppState, user_text: str):
    ai_response = ""

    for chunk in shared_state.chatbot.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config=shared_state.config,
        stream_mode="updates",
    ):
        if "chatbot" in chunk:
            messages = chunk["chatbot"].get("messages", [])

            for msg in messages:
                if isinstance(msg, AIMessage):
                    ai_response = msg.content

    return ai_response


def process_user_message(
    shared_state: SharedAppState,
    user_text: str,
    source: str,
):
    """
    Process one user message from either:
        source="typed"
        source="spoken"
    """
    if not shared_state.try_start_processing():
        notify_popup_system_message(shared_state, "Still processing previous message.")
        shared_state.set_status("Still processing previous message.")
        return

    try:
        shared_state.set_status("Detecting conversation emotion...")
        notify_popup_status(shared_state, "Detecting emotion...")

        detected_emotion = shared_state.emotion_detector.detect(user_text)

        shared_state.set_latest_conversation_emotion(
            emotion=detected_emotion,
            text=user_text,
            source=source,
            confidence=1.0,
        )

        if source == "typed":
            role = "user_typed"
        else:
            role = "user_spoken"

        shared_state.add_chat_message(
            role=role,
            text=user_text,
            emotion=detected_emotion,
        )

        shared_state.set_last_user_text(user_text)

        notify_popup_user_message(
            shared_state,
            text=user_text,
            source=source,
            emotion=detected_emotion,
        )

        notify_popup_emotion(shared_state, detected_emotion)

        shared_state.set_status(f"Conversation emotion: {detected_emotion}. Thinking...")
        notify_popup_status(shared_state, "Thinking...")

        ai_response = get_chatbot_response(shared_state, user_text)

        shared_state.add_chat_message(
            role="assistant",
            text=ai_response,
        )

        shared_state.set_last_ai_response(ai_response)

        notify_popup_ai_message(shared_state, ai_response)

        shared_state.set_status("Ready.")
        notify_popup_status(shared_state, "Ready.")

    except Exception as e:
        error_text = f"Chat processing error: {e}"
        shared_state.add_chat_message("system", error_text)
        shared_state.set_status("Chat processing error.")
        notify_popup_system_message(shared_state, error_text)
        notify_popup_status(shared_state, "Error.")

    finally:
        shared_state.stop_processing()


# -----------------------------
# Voice input
# -----------------------------

def audio_callback_factory(shared_state: SharedAppState):
    def audio_callback(indata, frames, callback_time, status):
        with shared_state.lock:
            if shared_state.is_recording:
                shared_state.recording.append(indata.copy())

    return audio_callback


def start_voice_recording(shared_state: SharedAppState):
    with shared_state.lock:
        if shared_state.is_chat_processing:
            return

        if shared_state.popup_visible:
            return

        if not shared_state.is_recording:
            shared_state.is_recording = True
            shared_state.recording.clear()
            shared_state.voice_status = "Recording... release SPACE to send."
            shared_state.status_text = "Recording voice..."


def stop_voice_recording(shared_state: SharedAppState):
    with shared_state.lock:
        if not shared_state.is_recording:
            return

        shared_state.is_recording = False

        if not shared_state.recording:
            shared_state.voice_status = "No audio recorded."
            return

    worker = threading.Thread(
        target=process_voice_recording,
        args=(shared_state,),
        daemon=True,
    )
    worker.start()


def process_voice_recording(shared_state: SharedAppState):
    if shared_state.openai_client is None:
        shared_state.add_chat_message(
            "system",
            "Voice transcription requires OPENAI_API_KEY.",
        )
        shared_state.set_status("Missing OPENAI_API_KEY.")
        shared_state.set_voice_status("Missing OPENAI_API_KEY.")
        return

    with shared_state.lock:
        recording_copy = list(shared_state.recording)

    if not recording_copy:
        shared_state.set_status("No audio recorded.")
        shared_state.set_voice_status("No audio recorded.")
        return

    try:
        shared_state.set_status("Transcribing voice...")
        shared_state.set_voice_status("Transcribing...")

        audio = np.concatenate(recording_copy, axis=0)

        audio_bytes_io = io.BytesIO()
        sf.write(audio_bytes_io, audio, FS, format="wav")
        audio_bytes_io.seek(0)
        audio_bytes_io.name = "audio.wav"

        transcript = shared_state.openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes_io,
        )

        user_text = transcript.text.strip()

        if not user_text:
            shared_state.set_status("No speech detected.")
            shared_state.set_voice_status("No speech detected.")
            return

        shared_state.set_voice_status(f"You said: {user_text}")

        process_user_message(
            shared_state=shared_state,
            user_text=user_text,
            source="spoken",
        )

        shared_state.set_voice_status("Ready. Hold SPACE to talk.")

    except Exception as e:
        error_text = f"Voice processing error: {e}"
        shared_state.add_chat_message("system", error_text)
        shared_state.set_status("Voice processing error.")
        shared_state.set_voice_status("Voice processing error.")
        notify_popup_system_message(shared_state, error_text)


# -----------------------------
# Robot sequence playback
# -----------------------------

current_sequence_lock = threading.Lock()
current_sequence_name = None
sequence_thread = None
last_sequence_time = 0.0


def play_sequence_async(sequence_name: str):
    global current_sequence_name

    try:
        sequence = get_sequence_by_name(sequence_name)

        if sequence is None:
            print(f"Warning: Sequence '{sequence_name}' not found")
            return

        print(f"Playing sequence: {sequence_name}")

        sequence.start()
        sequence.wait_to_stop()

        print(f"Finished playing sequence: {sequence_name}")

    except Exception as e:
        print(f"Error playing sequence: {e}")

    finally:
        with current_sequence_lock:
            current_sequence_name = None


def trigger_emotion_sequence(final_emotion: str):
    global current_sequence_name, sequence_thread, last_sequence_time

    sequence_name = FINAL_EMOTION_TO_SEQUENCE.get(final_emotion)

    if sequence_name is None:
        return False

    now = time.time()

    if now - last_sequence_time < ROBOT_SEQUENCE_COOLDOWN_SECONDS:
        return False

    with current_sequence_lock:
        if current_sequence_name is not None:
            return False

        current_sequence_name = sequence_name
        last_sequence_time = now

    sequence_thread = threading.Thread(
        target=play_sequence_async,
        args=(sequence_name,),
        daemon=True,
    )
    sequence_thread.start()

    return True


# -----------------------------
# Landmark helpers
# -----------------------------

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = np.array(landmark_list, dtype=np.float32)

    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    temp_landmark_list[:, 0] = temp_landmark_list[:, 0] - base_x
    temp_landmark_list[:, 1] = temp_landmark_list[:, 1] - base_y

    temp_landmark_list = temp_landmark_list.flatten()

    max_value = max(list(map(abs, temp_landmark_list)))

    if max_value == 0:
        return temp_landmark_list.tolist()

    temp_landmark_list = temp_landmark_list / max_value

    return temp_landmark_list.tolist()


def draw_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return x, y, x + w, y + h


def draw_label(image, brect, text, color=(0, 0, 0)):
    x1, y1, x2, _ = brect
    top_y = max(0, y1 - 24)

    cv2.rectangle(
        image,
        (x1, top_y),
        (x2, y1),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x1 + 5, max(15, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image


# -----------------------------
# Drawing helpers
# -----------------------------

def draw_text_with_outline(
    image,
    text,
    position,
    font_scale=0.58,
    thickness=1,
    text_color=(255, 255, 255),
    outline_color=(0, 0, 0),
):
    x, y = position

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_info_panel(
    image,
    shared_state: SharedAppState,
    result=None,
    waiting_text=None,
    conversation_state=None,
):
    """
    Draw status text directly on the camera feed without a black background.
    """
    lines = []

    if waiting_text is not None:
        lines.append(waiting_text)

    elif result is not None:
        final_emotion = result["final_emotion"]
        final_probs = result["final_probs"]
        mode = result["mode"]
        weights = result["weights"]

        lines.extend(
            [
                f"Mode: {mode}",
                (
                    "Weights -> "
                    f"Face: {weights.get('facial', 0.0):.2f}, "
                    f"Hand: {weights.get('gesture', 0.0):.2f}, "
                    f"Chat: {weights.get('conversation', 0.0):.2f}"
                ),
                f"Final Emotion: {final_emotion}",
                f"happy: {final_probs.get('happy', 0.0):.2f}",
                f"angry: {final_probs.get('angry', 0.0):.2f}",
                f"sad: {final_probs.get('sad', 0.0):.2f}",
            ]
        )

        top_facial = result.get("top_facial_emotion")
        top_gesture = result.get("top_gesture")
        conversation_emotion = result.get("conversation_emotion")

        if top_facial is not None:
            lines.append(f"Top facial class: {top_facial}")

        if top_gesture is not None:
            lines.append(f"Top gesture class: {top_gesture}")

        if conversation_emotion is not None:
            lines.append(f"Conversation emotion: {conversation_emotion}")

        if conversation_state is not None:
            age = conversation_state.get("age_seconds", 0.0)
            source = conversation_state.get("source", "conversation")
            lines.append(f"Conversation source: {source}, age: {age:.1f}s")

    status_text = shared_state.get_status()
    voice_status = shared_state.get_voice_status()
    last_user_text = shared_state.get_last_user_text()
    last_ai_response = shared_state.get_last_ai_response()

    if status_text:
        lines.append(f"Status: {status_text}")

    if voice_status:
        lines.append(f"Voice: {voice_status}")

    if last_user_text:
        shortened_user = last_user_text[:75]
        lines.append(f"Last user: {shortened_user}")

    if last_ai_response:
        shortened_ai = last_ai_response[:75]
        lines.append(f"Blossom: {shortened_ai}")

    x = 20
    y = 35
    line_height = 24

    for line in lines[:14]:
        draw_text_with_outline(
            image=image,
            text=line,
            position=(x, y),
            font_scale=0.58,
            thickness=1,
        )

        y += line_height

    return image


def draw_playing_text(image):
    with current_sequence_lock:
        if current_sequence_name is None:
            return image

        playing_text = f"Playing: {current_sequence_name}"

    text_size = cv2.getTextSize(
        playing_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        2,
    )[0]

    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = 50

    cv2.rectangle(
        image,
        (text_x - 10, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        image,
        playing_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return image


# -----------------------------
# Keyboard handling
# -----------------------------

def safe_is_pressed(key_name):
    try:
        return keyboard.is_pressed(key_name)
    except Exception:
        return False


def handle_keyboard_controls(shared_state: SharedAppState):
    """
    Uses the keyboard package so SPACE can behave as hold-to-talk while
    the OpenCV camera window stays visible.

    Text input is disabled for voice mode while the chat popup is open.
    """
    if shared_state.is_popup_visible():
        return

    if safe_is_pressed("space"):
        start_voice_recording(shared_state)
    else:
        stop_voice_recording(shared_state)


# -----------------------------
# Main app
# -----------------------------

def main():
    load_project_env()

    print("Starting single-file Blossom emotion system...")
    print("")
    print("Controls:")
    print("  Hold SPACE     -> voice input")
    print("  Release SPACE  -> transcribe and chat")
    print("  ENTER          -> open text chat popup")
    print("  q              -> quit camera window")
    print("")
    print("Fusion sources:")
    print("  face, gesture, conversation")
    print("  any one, any two, or all three can produce the final emotion")
    print("")

    tk_root = tk.Tk()
    tk_root.withdraw()

    shared_state = SharedAppState()

    if has_openai_api_key():
        shared_state.add_chat_message("system", "OPENAI_API_KEY loaded.")
        print("OPENAI_API_KEY loaded.")
    else:
        shared_state.add_chat_message(
            "system",
            "OPENAI_API_KEY not found. Voice transcription will not work.",
        )
        print("Warning: OPENAI_API_KEY not found.")

    try:
        robot = get_blossom_robot()
        print("Robot initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize robot: {e}")
        print("Running in simulation mode. Sequences will not play.")
        robot = None

    try:
        fusion_classifier = EmotionFusionClassifier()
        print("Fusion classifier loaded successfully!")
    except Exception as e:
        print(f"Error loading fusion classifier: {e}")
        print("Make sure these files exist:")
        print("  models/emotion_classifier.tflite")
        print("  models/gesture_classifier.tflite")
        tk_root.destroy()
        return

    try:
        audio_stream = sd.InputStream(
            callback=audio_callback_factory(shared_state),
            samplerate=FS,
            channels=1,
        )
        audio_stream.start()
        print("Audio input stream started.")
    except Exception as e:
        audio_stream = None
        print(f"Warning: Could not start audio stream: {e}")

    vid = cv2.VideoCapture(CAMERA_INDEX)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not vid.isOpened():
        print("Error: Could not open camera")

        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()

        tk_root.destroy()
        return

    print("Camera opened successfully.")
    print("Press q in the camera window to quit.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            try:
                tk_root.update_idletasks()
                tk_root.update()
            except tk.TclError:
                break

            handle_keyboard_controls(shared_state)

            ret, frame = vid.read()

            if not ret:
                print("Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            facial_features = None
            gesture_features = None

            # ---------- FACE ----------
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                face_brect = draw_bounding_rect(frame, face_landmarks)
                face_landmark_list = calc_landmark_list(frame, face_landmarks)
                facial_features = pre_process_landmark(face_landmark_list)

                frame = draw_label(
                    frame,
                    face_brect,
                    "Face detected",
                    color=(0, 0, 0),
                )

            # ---------- HAND ----------
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                hand_brect = draw_bounding_rect(frame, hand_landmarks)
                hand_landmark_list = calc_landmark_list(frame, hand_landmarks)
                gesture_features = pre_process_landmark(hand_landmark_list)

                frame = draw_label(
                    frame,
                    hand_brect,
                    "Hand detected",
                    color=(0, 0, 0),
                )

            # ---------- CONVERSATION ----------
            conversation_state = shared_state.get_recent_conversation_state()

            conversation_emotion = None
            conversation_confidence = 1.0

            if conversation_state is not None:
                conversation_emotion = conversation_state.get("emotion")
                conversation_confidence = conversation_state.get("confidence", 1.0)

            # ---------- FUSION ----------
            if (
                facial_features is not None
                or gesture_features is not None
                or conversation_emotion is not None
            ):
                try:
                    result = fusion_classifier.predict(
                        facial_features=facial_features,
                        gesture_features=gesture_features,
                        conversation_emotion=conversation_emotion,
                        conversation_confidence=conversation_confidence,
                        source_weights=SOURCE_WEIGHTS,
                    )

                    frame = draw_info_panel(
                        image=frame,
                        shared_state=shared_state,
                        result=result,
                        conversation_state=conversation_state,
                    )

                    final_emotion = result["final_emotion"]
                    final_confidence = result["final_probs"].get(final_emotion, 0.0)

                    if (
                        robot is not None
                        and final_confidence >= FINAL_CONFIDENCE_THRESHOLD
                    ):
                        trigger_emotion_sequence(final_emotion)

                except Exception as e:
                    print(f"Fusion classification error: {e}")

                    frame = draw_info_panel(
                        image=frame,
                        shared_state=shared_state,
                        waiting_text=f"Fusion error: {str(e)[:55]}",
                    )

            else:
                frame = draw_info_panel(
                    image=frame,
                    shared_state=shared_state,
                    waiting_text="Waiting for face, hand, or conversation emotion...",
                )

            frame = draw_playing_text(frame)

            draw_text_with_outline(
                image=frame,
                text="Hold SPACE: talk | ENTER: text chat | q: quit",
                position=(10, frame.shape[0] - 15),
                font_scale=0.6,
                thickness=1,
            )

            cv2.imshow("Blossom Combined Emotion Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in (10, 13):
                open_chat_popup(tk_root, shared_state)

    vid.release()
    cv2.destroyAllWindows()

    try:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
    except Exception:
        pass

    try:
        sd.stop()
    except Exception:
        pass

    try:
        tk_root.destroy()
    except Exception:
        pass

    print("Blossom combined emotion system stopped.")

if __name__ == "__main__":
    main()