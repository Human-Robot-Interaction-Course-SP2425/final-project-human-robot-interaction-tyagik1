from apps.shared.utils.sequence import get_all_sequences, get_all_sequences_str, get_sequence_by_name
import dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import io


dotenv.load_dotenv()


def get_available_sequences() -> str:
    sequences = get_all_sequences()
    if not sequences:
        return "No sequences available."
    
    happy = [s.animation for s in sequences if s.animation.startswith("happy_")]
    sad = [s.animation for s in sequences if s.animation.startswith("sad_")]
    anger = [s.animation for s in sequences if s.animation.startswith("anger_")]
    fear = [s.animation for s in sequences if s.animation.startswith("fear_")]
    other = [s.animation for s in sequences if not any(s.animation.startswith(prefix) for prefix in ["happy_", "sad_", "anger_", "fear_"])]
    
    result = "Available sequences by emotion:\n\n"
    
    if happy:
        result += f"HAPPY ({len(happy)} sequences): {', '.join(happy[:5])}\n"
    if sad:
        result += f"SAD ({len(sad)} sequences): {', '.join(sad[:5])}\n"
    if anger:
        result += f"ANGER ({len(anger)} sequences): {', '.join(anger[:5])}\n"
    if fear:
        result += f"FEAR ({len(fear)} sequences): {', '.join(fear[:5])}\n"
    if other:
        result += f"OTHER ({len(other)} sequences): {', '.join(other)}\n"
    
    result += "\nTo play a sequence, use the play_sequence tool with the exact sequence name."
    
    return result


def play_sequence(sequence_name: str) -> bool:
    sequence = get_sequence_by_name(sequence_name)
    if sequence is None:
        return f"Sequence '{sequence_name}' not found. Available sequences:\n{get_all_sequences_str()}"
    
    try:
        sequence.start()
        sequence.wait_to_stop()

        reset_sequence = get_sequence_by_name("reset")
        reset_sequence.start()
        return True
    except (ValueError, RuntimeError, AttributeError) as e:
        return False


def speak(text: str):
    try:
        client = OpenAI()
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        
        audio_bytes = io.BytesIO(response.content)
        
        data, samplerate = sf.read(audio_bytes)
        
        sd.play(data, samplerate)
        sd.wait()
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
    

