"""
Conversation emotion detection agent.

Classifies the user's conversation text into exactly one of:

    happy
    sad
    angry

Primary path:
    AutoGen AgentChat AssistantAgent

Fallback path:
    Direct OpenAI chat completion

Final fallback:
    Local keyword rules

This keeps emotion detection separate from the UI, so later we can fuse:
    facial emotion + gesture emotion + conversation emotion
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI


TARGET_EMOTIONS = ["happy", "sad", "angry"]

DEFAULT_AUTOGEN_MODEL = "gpt-4o-mini"
DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"


def get_project_root() -> Path:
    """
    Current file:
        HRIBlossom/apps/chat/emotion_agent.py

    Project root:
        HRIBlossom/
    """
    return Path(__file__).resolve().parents[2]


def load_project_env():
    """
    Minimal .env loader so this works even if python-dotenv is not installed.

    Expected file:
        HRIBlossom/.env

    Expected line:
        OPENAI_API_KEY=your_key_here
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


class ConversationEmotionDetector:
    def __init__(
        self,
        model: str = DEFAULT_AUTOGEN_MODEL,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        prefer_autogen: bool = True,
    ):
        load_project_env()

        self.model = model
        self.fallback_model = fallback_model
        self.prefer_autogen = prefer_autogen

        self.openai_client = None
        self.openai_client_error = None

        self.autogen_available = False
        self.autogen_import_error = None

        self.AssistantAgent = None
        self.OpenAIChatCompletionClient = None

        if has_openai_api_key():
            try:
                self.openai_client = OpenAI()
            except Exception as e:
                self.openai_client_error = e
                self.openai_client = None

        if self.prefer_autogen:
            self._try_import_autogen()

    # -----------------------------
    # Public API
    # -----------------------------

    def detect(self, user_text: str) -> str:
        """
        Detect the emotion of the user text.

        Always returns one of:
            happy
            sad
            angry
        """
        user_text = (user_text or "").strip()

        if not user_text:
            return "sad"

        if self.autogen_available and has_openai_api_key():
            try:
                emotion = self._detect_with_autogen(user_text)
                emotion = self.normalize_emotion_label(emotion)

                if emotion in TARGET_EMOTIONS:
                    return emotion

            except Exception:
                pass

        if self.openai_client is not None:
            try:
                emotion = self._detect_with_openai(user_text)
                emotion = self.normalize_emotion_label(emotion)

                if emotion in TARGET_EMOTIONS:
                    return emotion

            except Exception:
                pass

        return self.detect_with_keywords(user_text)

    # -----------------------------
    # AutoGen detection
    # -----------------------------

    def _try_import_autogen(self):
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            self.AssistantAgent = AssistantAgent
            self.OpenAIChatCompletionClient = OpenAIChatCompletionClient
            self.autogen_available = True

        except Exception as e:
            self.autogen_import_error = e
            self.autogen_available = False

    def _detect_with_autogen(self, user_text: str) -> str:
        """
        Runs the AutoGen emotion classifier.
        """
        try:
            return asyncio.run(self._detect_with_autogen_async(user_text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    self._detect_with_autogen_async(user_text)
                )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

    async def _detect_with_autogen_async(self, user_text: str) -> str:
        model_client = self.OpenAIChatCompletionClient(
            model=self.model
        )

        agent = self.AssistantAgent(
            name="emotion_detector",
            model_client=model_client,
            system_message=(
                "You are an emotion detection agent for a social robot. "
                "Your only job is to classify the user's current emotional state "
                "from their message. "
                "Allowed labels are exactly: happy, sad, angry. "
                "Even if the message is neutral, choose the closest of those three. "
                "Return only the label. Do not explain."
            ),
        )

        task = (
            "Classify this user message into exactly one label: "
            "happy, sad, or angry.\n\n"
            f"User message: {user_text}"
        )

        try:
            result = await agent.run(task=task)

            if not result.messages:
                return self.detect_with_keywords(user_text)

            last_message = result.messages[-1]
            content = getattr(last_message, "content", "")

            if isinstance(content, str):
                return content.strip()

            return str(content).strip()

        finally:
            try:
                await model_client.close()
            except Exception:
                pass

    # -----------------------------
    # Direct OpenAI fallback
    # -----------------------------

    def _detect_with_openai(self, user_text: str) -> str:
        if self.openai_client is None:
            return self.detect_with_keywords(user_text)

        response = self.openai_client.chat.completions.create(
            model=self.fallback_model,
            temperature=0,
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emotion classifier. "
                        "Classify the user's emotional state using exactly one label: "
                        "happy, sad, or angry. "
                        "Even if the message is neutral, choose the closest of the three. "
                        "Return only the label and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
        )

        return response.choices[0].message.content.strip()

    # -----------------------------
    # Local fallback
    # -----------------------------

    def detect_with_keywords(self, user_text: str) -> str:
        text = user_text.lower()

        angry_words = [
            "angry",
            "mad",
            "furious",
            "annoyed",
            "irritated",
            "frustrated",
            "hate",
            "upset with",
            "pissed",
            "rage",
            "unfair",
            "stupid",
            "terrible",
            "awful",
            "ridiculous",
            "sick of",
            "done with",
            "can't stand",
        ]

        sad_words = [
            "sad",
            "unhappy",
            "depressed",
            "lonely",
            "cry",
            "crying",
            "hurt",
            "heartbroken",
            "tired",
            "hopeless",
            "miss",
            "lost",
            "bad day",
            "down",
            "miserable",
            "worried",
            "scared",
            "anxious",
            "overwhelmed",
        ]

        happy_words = [
            "happy",
            "glad",
            "excited",
            "great",
            "good",
            "awesome",
            "amazing",
            "love",
            "wonderful",
            "fun",
            "nice",
            "proud",
            "yay",
            "lol",
            "haha",
            "thanks",
            "thank you",
            "cool",
        ]

        angry_score = sum(1 for word in angry_words if word in text)
        sad_score = sum(1 for word in sad_words if word in text)
        happy_score = sum(1 for word in happy_words if word in text)

        scores = {
            "happy": happy_score,
            "sad": sad_score,
            "angry": angry_score,
        }

        best_emotion = max(scores, key=scores.get)

        if scores[best_emotion] == 0:
            return "happy"

        return best_emotion

    # -----------------------------
    # Label normalization
    # -----------------------------

    def normalize_emotion_label(self, emotion_text: Optional[str]) -> str:
        emotion_text = (emotion_text or "").strip().lower()

        if "happy" in emotion_text:
            return "happy"

        if "joy" in emotion_text:
            return "happy"

        if "excited" in emotion_text:
            return "happy"

        if "sad" in emotion_text:
            return "sad"

        if "depressed" in emotion_text:
            return "sad"

        if "upset" in emotion_text:
            return "sad"

        if "angry" in emotion_text:
            return "angry"

        if "anger" in emotion_text:
            return "angry"

        if "mad" in emotion_text:
            return "angry"

        if emotion_text in TARGET_EMOTIONS:
            return emotion_text

        return "happy"


if __name__ == "__main__":
    load_project_env()

    detector = ConversationEmotionDetector()

    examples = [
        "I am really excited about this!",
        "I feel awful today.",
        "This is so frustrating and unfair.",
        "Okay, I guess that works.",
    ]

    print("OPENAI_API_KEY found:", has_openai_api_key())
    print("AutoGen available:", detector.autogen_available)

    if detector.autogen_import_error is not None:
        print("AutoGen import error:", detector.autogen_import_error)

    if detector.openai_client_error is not None:
        print("OpenAI client error:", detector.openai_client_error)

    for example in examples:
        emotion = detector.detect(example)
        print(f"{example} -> {emotion}")