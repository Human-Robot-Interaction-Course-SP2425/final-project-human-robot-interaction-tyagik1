"""
Fusion logic for combining:

1. Facial emotion recognition
2. Gesture recognition
3. Conversation emotion detection

The final target emotions are:

    happy
    angry
    sad

The fusion works even when only one or two sources are available.

Examples:
    face only                  -> 100% face
    gesture only               -> 100% gesture
    conversation only          -> 100% conversation
    face + gesture             -> 50% face, 50% gesture
    face + conversation        -> 50% face, 50% conversation
    gesture + conversation     -> 50% gesture, 50% conversation
    face + gesture + text      -> 33.3% each
"""

import os
import numpy as np
import tensorflow as tf


FINAL_EMOTIONS = ["happy", "angry", "sad"]

# Must match the order used by your current facial model.
# Your current facial model was trained with:
#   ["happy", "sad", "neutral", "surprised"]
FACIAL_EMOTIONS = ["happy", "sad", "neutral", "surprised"]

# Must match the order used by your current gesture model.
GESTURES = ["thumbs_up", "peace_sign", "closed_fist", "open_palm"]


# Temporary gesture-to-emotion mapping.
# Adjust this if your gesture meanings change.
GESTURE_TO_EMOTION = {
    "thumbs_up": "happy",
    "peace_sign": "happy",
    "closed_fist": "angry",
    "open_palm": "sad",
}


# Temporary facial-to-final-emotion mapping.
# Your current face model does not have angry yet.
# If you retrain the facial model with ["happy", "angry", "sad"],
# update FACIAL_EMOTIONS and this mapping.
FACIAL_TO_EMOTION = {
    "happy": "happy",
    "sad": "sad",
    "neutral": None,
    "surprised": "happy",
}


def get_project_root():
    """
    Current file:
        HRIBlossom/apps/shared/emotion_fusion.py

    Project root:
        HRIBlossom/
    """
    current_file = os.path.abspath(__file__)
    shared_dir = os.path.dirname(current_file)
    apps_dir = os.path.dirname(shared_dir)
    project_root = os.path.dirname(apps_dir)
    return project_root


def get_default_facial_model_path():
    return os.path.join(get_project_root(), "models", "emotion_classifier.tflite")


def get_default_gesture_model_path():
    return os.path.join(get_project_root(), "models", "gesture_classifier.tflite")


def normalize_label(label):
    if label is None:
        return None

    label = str(label).strip().lower()

    if label == "anger":
        return "angry"

    if label in FINAL_EMOTIONS:
        return label

    return None


class TFLiteClassifier:
    """
    Small wrapper around a TensorFlow Lite classifier.
    """

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model file: {model_path}")

        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, features):
        """
        Returns the raw probability vector from the TFLite model.
        """
        features = np.array(features, dtype=np.float32)

        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)

        expected_shape = self.input_details[0]["shape"]

        if features.shape[1] != expected_shape[1]:
            raise ValueError(
                f"Incorrect input size for model {self.model_path}. "
                f"Expected {expected_shape[1]} features, got {features.shape[1]}."
            )

        self.interpreter.set_tensor(self.input_details[0]["index"], features)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output[0]


def normalize(probabilities):
    probabilities = np.array(probabilities, dtype=np.float32)
    total = np.sum(probabilities)

    if total <= 0:
        return probabilities

    return probabilities / total


def zero_final_probs():
    return np.zeros(len(FINAL_EMOTIONS), dtype=np.float32)


def zero_facial_raw_probs():
    return np.zeros(len(FACIAL_EMOTIONS), dtype=np.float32)


def zero_gesture_raw_probs():
    return np.zeros(len(GESTURES), dtype=np.float32)


def probs_to_dict(labels, probs):
    return {
        label: float(prob)
        for label, prob in zip(labels, probs)
    }


def get_top_label(labels, probs):
    if probs is None or len(probs) == 0:
        return None

    return labels[int(np.argmax(probs))]


def facial_probs_to_final_emotions(facial_probs):
    """
    Convert facial model probabilities into final emotion probabilities.

    Current facial model outputs:
        happy, sad, neutral, surprised

    Final system wants:
        happy, angry, sad

    Current limitation:
        angry cannot come from the facial model until the facial model is
        retrained with an angry class.
    """
    final_probs = zero_final_probs()

    for i, facial_emotion in enumerate(FACIAL_EMOTIONS):
        mapped_emotion = FACIAL_TO_EMOTION.get(facial_emotion)

        if mapped_emotion in FINAL_EMOTIONS:
            final_index = FINAL_EMOTIONS.index(mapped_emotion)
            final_probs[final_index] += facial_probs[i]

    final_probs = normalize(final_probs)

    # If everything was ignored because the face model predicted neutral only,
    # fall back to the closest target emotion.
    if np.sum(final_probs) <= 0:
        final_probs[FINAL_EMOTIONS.index("happy")] = 1.0

    return final_probs


def gesture_probs_to_final_emotions(gesture_probs):
    """
    Convert gesture model probabilities into final emotion probabilities.
    """
    final_probs = zero_final_probs()

    for i, gesture in enumerate(GESTURES):
        mapped_emotion = GESTURE_TO_EMOTION.get(gesture)

        if mapped_emotion in FINAL_EMOTIONS:
            final_index = FINAL_EMOTIONS.index(mapped_emotion)
            final_probs[final_index] += gesture_probs[i]

    final_probs = normalize(final_probs)

    if np.sum(final_probs) <= 0:
        final_probs[FINAL_EMOTIONS.index("happy")] = 1.0

    return final_probs


def conversation_emotion_to_final_probs(
    conversation_emotion,
    conversation_confidence=1.0,
):
    """
    Convert conversation emotion into final emotion probabilities.

    Example:
        conversation_emotion = "sad"

    Output:
        happy: 0
        angry: 0
        sad: 1
    """
    final_probs = zero_final_probs()

    emotion = normalize_label(conversation_emotion)

    if emotion is None:
        return final_probs

    confidence = float(conversation_confidence)

    if confidence < 0:
        confidence = 0.0

    if confidence > 1:
        confidence = 1.0

    final_index = FINAL_EMOTIONS.index(emotion)
    final_probs[final_index] = confidence

    # Put the remaining uncertainty evenly across the other classes.
    remaining = 1.0 - confidence

    if remaining > 0:
        other_indices = [
            i for i in range(len(FINAL_EMOTIONS))
            if i != final_index
        ]

        for i in other_indices:
            final_probs[i] = remaining / len(other_indices)

    return normalize(final_probs)


class EmotionFusionClassifier:
    """
    Main fusion classifier.

    Use this when you have facial/gesture landmark features and optionally
    a conversation emotion.

    Example:
        classifier = EmotionFusionClassifier()

        result = classifier.predict(
            facial_features=face_features,
            gesture_features=hand_features,
            conversation_emotion="sad"
        )
    """

    def __init__(self, facial_model_path=None, gesture_model_path=None):
        if facial_model_path is None:
            facial_model_path = get_default_facial_model_path()

        if gesture_model_path is None:
            gesture_model_path = get_default_gesture_model_path()

        self.facial_classifier = TFLiteClassifier(facial_model_path)
        self.gesture_classifier = TFLiteClassifier(gesture_model_path)

    def predict(
        self,
        facial_features=None,
        gesture_features=None,
        conversation_emotion=None,
        conversation_confidence=1.0,
        source_weights=None,
    ):
        """
        Predict final emotion using any available source.

        Args:
            facial_features:
                Preprocessed facial landmarks, or None.
            gesture_features:
                Preprocessed hand landmarks, or None.
            conversation_emotion:
                happy, sad, angry, anger, or None.
            conversation_confidence:
                Confidence for conversation emotion.
            source_weights:
                Optional dictionary:
                    {
                        "facial": 1.0,
                        "gesture": 1.0,
                        "conversation": 1.0
                    }

                Weights are automatically normalized over available sources.

        Returns:
            dictionary with final emotion, probabilities, source probabilities,
            and normalized weights.
        """
        conversation_emotion = normalize_label(conversation_emotion)

        has_face = facial_features is not None
        has_hand = gesture_features is not None
        has_conversation = conversation_emotion is not None

        if not has_face and not has_hand and not has_conversation:
            raise ValueError(
                "At least one source must be provided: "
                "facial_features, gesture_features, or conversation_emotion."
            )

        facial_raw_probs = zero_facial_raw_probs()
        gesture_raw_probs = zero_gesture_raw_probs()

        facial_emotion_probs = zero_final_probs()
        gesture_emotion_probs = zero_final_probs()
        conversation_emotion_probs = zero_final_probs()

        if has_face:
            facial_raw_probs = self.facial_classifier.predict(facial_features)
            facial_emotion_probs = facial_probs_to_final_emotions(facial_raw_probs)

        if has_hand:
            gesture_raw_probs = self.gesture_classifier.predict(gesture_features)
            gesture_emotion_probs = gesture_probs_to_final_emotions(gesture_raw_probs)

        if has_conversation:
            conversation_emotion_probs = conversation_emotion_to_final_probs(
                conversation_emotion=conversation_emotion,
                conversation_confidence=conversation_confidence,
            )

        if source_weights is None:
            source_weights = {
                "facial": 1.0,
                "gesture": 1.0,
                "conversation": 1.0,
            }

        available_weights = {}

        if has_face:
            available_weights["facial"] = float(source_weights.get("facial", 1.0))

        if has_hand:
            available_weights["gesture"] = float(source_weights.get("gesture", 1.0))

        if has_conversation:
            available_weights["conversation"] = float(
                source_weights.get("conversation", 1.0)
            )

        # Remove negative weights.
        for key in list(available_weights.keys()):
            if available_weights[key] < 0:
                available_weights[key] = 0.0

        total_weight = sum(available_weights.values())

        if total_weight <= 0:
            # If user passes all zero weights, fall back to equal weights.
            for key in available_weights:
                available_weights[key] = 1.0

            total_weight = sum(available_weights.values())

        normalized_weights = {
            key: value / total_weight
            for key, value in available_weights.items()
        }

        final_probs = zero_final_probs()

        if has_face:
            final_probs += normalized_weights["facial"] * facial_emotion_probs

        if has_hand:
            final_probs += normalized_weights["gesture"] * gesture_emotion_probs

        if has_conversation:
            final_probs += (
                normalized_weights["conversation"] * conversation_emotion_probs
            )

        final_probs = normalize(final_probs)

        final_index = int(np.argmax(final_probs))
        final_emotion = FINAL_EMOTIONS[final_index]

        available_sources = []

        if has_face:
            available_sources.append("facial")

        if has_hand:
            available_sources.append("gesture")

        if has_conversation:
            available_sources.append("conversation")

        mode = "+".join(available_sources)

        return {
            "mode": mode,
            "available_sources": available_sources,
            "weights": {
                "facial": float(normalized_weights.get("facial", 0.0)),
                "gesture": float(normalized_weights.get("gesture", 0.0)),
                "conversation": float(normalized_weights.get("conversation", 0.0)),
            },
            "final_emotion": final_emotion,
            "final_probs": probs_to_dict(FINAL_EMOTIONS, final_probs),
            "facial_emotion_probs": probs_to_dict(
                FINAL_EMOTIONS,
                facial_emotion_probs,
            ),
            "gesture_emotion_probs": probs_to_dict(
                FINAL_EMOTIONS,
                gesture_emotion_probs,
            ),
            "conversation_emotion_probs": probs_to_dict(
                FINAL_EMOTIONS,
                conversation_emotion_probs,
            ),
            "raw_facial_probs": probs_to_dict(FACIAL_EMOTIONS, facial_raw_probs),
            "raw_gesture_probs": probs_to_dict(GESTURES, gesture_raw_probs),
            "top_facial_emotion": (
                get_top_label(FACIAL_EMOTIONS, facial_raw_probs)
                if has_face
                else None
            ),
            "top_gesture": (
                get_top_label(GESTURES, gesture_raw_probs)
                if has_hand
                else None
            ),
            "conversation_emotion": (
                conversation_emotion
                if has_conversation
                else None
            ),
        }


def classify_combined_emotion(
    facial_features=None,
    gesture_features=None,
    conversation_emotion=None,
    conversation_confidence=1.0,
    facial_model_path=None,
    gesture_model_path=None,
    source_weights=None,
):
    """
    Convenience function.

    For real-time apps, prefer EmotionFusionClassifier so models are loaded once.
    """
    classifier = EmotionFusionClassifier(
        facial_model_path=facial_model_path,
        gesture_model_path=gesture_model_path,
    )

    return classifier.predict(
        facial_features=facial_features,
        gesture_features=gesture_features,
        conversation_emotion=conversation_emotion,
        conversation_confidence=conversation_confidence,
        source_weights=source_weights,
    )


if __name__ == "__main__":
    print("Testing emotion_fusion.py import and model loading...")

    facial_model_path = get_default_facial_model_path()
    gesture_model_path = get_default_gesture_model_path()

    print(f"Facial model path: {facial_model_path}")
    print(f"Gesture model path: {gesture_model_path}")

    if os.path.exists(facial_model_path):
        print("Facial model found.")
    else:
        print("Facial model NOT found.")

    if os.path.exists(gesture_model_path):
        print("Gesture model found.")
    else:
        print("Gesture model NOT found.")

    print("emotion_fusion.py loaded successfully.")