"""
Shared conversation emotion state.

This file lets the chat UI and the combined recognition app share the latest
conversation-based emotion.

The chat UI writes:
    happy / sad / angry

The combined recognition app reads it and fuses it with:
    facial emotion
    gesture emotion

State is stored in:
    HRIBlossom/models/conversation_emotion_state.json
"""

import json
import time
from pathlib import Path


TARGET_EMOTIONS = {"happy", "sad", "angry"}


def get_project_root() -> Path:
    """
    Current file:
        HRIBlossom/apps/shared/emotion_state.py

    Project root:
        HRIBlossom/
    """
    return Path(__file__).resolve().parents[2]


def get_state_file_path() -> Path:
    return get_project_root() / "models" / "conversation_emotion_state.json"


def normalize_emotion(emotion: str):
    if emotion is None:
        return None

    emotion = emotion.strip().lower()

    if emotion == "anger":
        return "angry"

    if emotion in TARGET_EMOTIONS:
        return emotion

    return None


def save_conversation_emotion(
    emotion: str,
    source: str = "chat",
    text: str = "",
    confidence: float = 1.0,
):
    """
    Save the most recent conversation emotion.

    Args:
        emotion:
            happy, sad, or angry
        source:
            typed, spoken, chat, etc.
        text:
            optional user message that produced the emotion
        confidence:
            confidence value from 0.0 to 1.0
    """
    normalized_emotion = normalize_emotion(emotion)

    if normalized_emotion is None:
        return False

    state_path = get_state_file_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "emotion": normalized_emotion,
        "source": source,
        "text": text,
        "confidence": float(confidence),
        "timestamp": time.time(),
    }

    try:
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return True

    except Exception:
        return False


def load_conversation_emotion():
    """
    Load the latest conversation emotion state.

    Returns:
        dict or None
    """
    state_path = get_state_file_path()

    if not state_path.exists():
        return None

    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        emotion = normalize_emotion(state.get("emotion"))

        if emotion is None:
            return None

        state["emotion"] = emotion
        state["confidence"] = float(state.get("confidence", 1.0))
        state["timestamp"] = float(state.get("timestamp", 0.0))

        return state

    except Exception:
        return None


def load_recent_conversation_emotion(max_age_seconds: float = 60.0):
    """
    Load the latest conversation emotion only if it is recent enough.

    Args:
        max_age_seconds:
            If the saved emotion is older than this, ignore it.

    Returns:
        dict or None
    """
    state = load_conversation_emotion()

    if state is None:
        return None

    age = time.time() - state["timestamp"]

    if age > max_age_seconds:
        return None

    state["age_seconds"] = age

    return state


def clear_conversation_emotion():
    """
    Delete the saved conversation emotion state.
    """
    state_path = get_state_file_path()

    try:
        if state_path.exists():
            state_path.unlink()

        return True

    except Exception:
        return False


if __name__ == "__main__":
    save_conversation_emotion(
        emotion="happy",
        source="test",
        text="This is a test message.",
        confidence=1.0,
    )

    print(load_conversation_emotion())