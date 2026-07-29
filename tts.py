"""TTS voice for the booth (M5). Fish Audio is the primary provider — a
distinct Taiwan-Mandarin voice PAIR per persona (calm/excited), not
just per speaker role; speed/temperature follow both the persona and the
Director's register intensity — same "tone is derived, not user-picked"
contract as the text side. Falls back to edge-tts (local/free, no key) when
FISHAUDIO_API_KEY isn't set or a call fails, so the broadcast degrades
gracefully rather than going silent (offline-first contract).

speak() returns (audio_bytes, format) — format is "mp3" from either
provider — or None if both are unavailable.
"""
import asyncio
import io
import logging

import config


# ---------- Fish Audio (primary, needs FISHAUDIO_API_KEY) ----------
# Persona-level multipliers on top of the intensity-based prosody below.
# Keyed by persona, or by (persona, speaker) for a tweak to just one of the
# two voices — e.g. 沉穩's male (play_by_play) speaks a touch faster than its
# female analyst, both otherwise sharing "calm"'s baseline.
_PERSONA_PROSODY = {
    "calm": (1.0, 1.0),
    ("calm", "play_by_play"): (1.3, 1.0),
    "excited": (1.0, 1.0),
}

# Temperature ceiling per persona — Fish Audio regenerates from the
# reference on every call rather than replaying a fixed recording, and high
# temperature makes that run-to-run drift more audible (the same voice can
# sound noticeably different call to call at critical's 0.9). Both personas
# get capped now — "excited" highest since it's meant to stay lively.
_TEMP_CAP = {"calm": 0.65, "excited": 0.75}

# Extra speed on top of the persona multiplier, for specific routes.
# 真人語料的開場白與閒聊都跑在最低的 intensity 檔（0.4 / 0.25 → speed 0.9），
# 而 realcast 沒有 calm 那份 play_by_play 加速，講起來就慢半拍、開場拖很長。
# 只提這兩條路徑，實際講棋（light/deep/recap/closing）的節奏不動。
_ROUTE_SPEED = {("realcast", "opening"): 1.2, ("realcast", "filler"): 1.2}


def _fish_prosody(intensity: float, persona: str, speaker: str,
                  route: str | None = None) -> tuple[float, float]:
    """(speed, temperature) — mirrors the three register tiers used for text
    and the edge-tts fallback, then scaled by persona (optionally per
    speaker too) and by route. speed range is [0.5, 2.0] per Fish's API;
    temperature [0, 1] controls expressiveness/randomness."""
    if intensity >= 0.8:
        speed, temp = 1.3, 0.9    # 情緒沸騰
    elif intensity >= 0.5:
        speed, temp = 1.05, 0.7   # 專業起伏
    else:
        speed, temp = 0.9, 0.5   # 平穩沉著
    speed_mul, temp_mul = _PERSONA_PROSODY.get(
        (persona, speaker), _PERSONA_PROSODY.get(persona, (1.0, 1.0)))
    speed_mul *= _ROUTE_SPEED.get((persona, route), 1.0)
    temp = min(temp * temp_mul, _TEMP_CAP.get(persona, 1.0))
    return max(0.5, min(2.0, speed * speed_mul)), max(0.0, min(1.0, temp))


def _fish_speak(text: str, speaker: str, intensity: float, persona: str,
                route: str | None = None):
    import requests
    voices = config.FISHAUDIO_VOICES.get(persona, config.FISHAUDIO_VOICES["calm"])
    reference_id = voices.get(speaker, voices["play_by_play"])
    speed, temperature = _fish_prosody(intensity, persona, speaker, route)
    try:
        resp = requests.post(
            "https://api.fish.audio/v1/tts",
            headers={"Authorization": f"Bearer {config.FISHAUDIO_API_KEY}",
                     "Content-Type": "application/json",
                     "model": "s1"},
            json={
                "text": text,
                "reference_id": reference_id,
                "format": "mp3",
                "temperature": temperature,
                "prosody": {"speed": speed},
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.content or None
    except Exception as e:
        logging.warning("Fish Audio TTS failed: %s", e)
        return None


# ---------- edge-tts (fallback — local/free, no key) ----------
def _edge_prosody(intensity: float, speed_mul: float = 1.0) -> tuple[str, str]:
    """(rate, pitch) offsets — same three register tiers, edge-tts's own syntax.
    speed_mul carries the same per-persona/route speed-up the Fish path applies,
    so a persona doesn't change pace just because it fell back to edge-tts."""
    if intensity >= 0.8:
        rate, pitch = 10, "+15Hz"
    elif intensity >= 0.5:
        rate, pitch = 0, "+5Hz"
    else:
        rate, pitch = -10, "+0Hz"
    return f"{round((1 + rate / 100) * speed_mul * 100 - 100):+d}%", pitch


async def _edge_synthesize(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _edge_speak(text: str, speaker: str, intensity: float,
                persona: str = "calm", route: str | None = None):
    """One retry: edge-tts occasionally drops a connection under concurrent
    load (several turns fire at once), and a retry is far cheaper than
    losing audio for that line."""
    voice = config.TTS_VOICES.get(speaker, config.TTS_VOICES["play_by_play"])
    rate, pitch = _edge_prosody(intensity, _ROUTE_SPEED.get((persona, route), 1.0))
    for attempt in range(2):
        try:
            data = asyncio.run(_edge_synthesize(text, voice, rate, pitch))
            if data:
                return data
        except Exception as e:
            logging.warning("edge-tts failed (attempt %d): %s", attempt + 1, e)
    return None


def speak(text: str, speaker: str, intensity: float = 0.2, persona: str = "calm",
          route: str | None = None):
    """Synthesize one commentary turn. `route` is the booth route this line came
    from ("opening"/"filler"/"light"/…) — only used for per-route pacing tweaks.
    Returns (audio_bytes, format) — format is always "mp3" here — or None if
    both providers are unavailable. Safe to call from any thread."""
    if not text:
        return None
    if config.FISHAUDIO_API_KEY:
        audio = _fish_speak(text, speaker, intensity, persona, route)
        if audio:
            return audio, "mp3"
        logging.warning("Fish Audio TTS unavailable for this line, falling back to edge-tts")
    data = _edge_speak(text, speaker, intensity, persona, route)
    return (data, "mp3") if data else None
