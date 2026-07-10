"""TTS voice for the booth (M5 stretch).

OpenAI TTS with a distinct voice per speaker; the delivery instructions are
derived from the Director's register intensity — the same "tone is derived,
not user-picked" contract as the text side.

Returns None when there's no API key or synthesis fails, so the broadcast
degrades to text-only (offline-first contract).
"""
import config


def _instructions(intensity: float) -> str:
    if intensity >= 0.8:
        return ("你是情緒沸騰的體育賽事主播，剛目睹關鍵時刻：語速快、"
                "音量與音調明顯上揚、充滿爆發力與緊張感。")
    if intensity >= 0.5:
        return "你是專業棋賽球評：語調有起伏、帶著明顯的興趣，但保持專業節制。"
    return "你是平穩沉著的棋賽旁白：語速從容、語調低調平和。"


def speak(text: str, speaker: str, intensity: float = 0.2):
    """Synthesize one commentary turn. Returns MP3 bytes, or None if unavailable."""
    if not config.USE_LLM or not text:
        return None
    try:
        from openai import OpenAI
        resp = OpenAI().audio.speech.create(
            model=config.TTS_MODEL,
            voice=config.TTS_VOICES.get(speaker, "alloy"),
            input=text,
            instructions=_instructions(intensity),
        )
        return resp.content
    except Exception:
        return None
