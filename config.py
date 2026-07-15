"""Central config. Tune SEVERITY thresholds here — they drive the whole router."""
import os
from dotenv import load_dotenv
load_dotenv()

# --- Stockfish ---
# Point this at your local Stockfish binary. Download: https://stockfishchess.org/download/
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
SHALLOW_DEPTH = 12          # baseline eval, every move (fast/cheap)
DEEP_DEPTH = 20             # deep path only (critical moves)
MULTIPV = 3                 # top-N lines

# --- Severity thresholds (centipawns). 100cp ~= one pawn. TUNE THESE. ---
DELTA_NOTABLE = 80          # |delta| >= this -> at least notable (depth-12 noise is ±50-70)
DELTA_CRITICAL = 150        # |delta| >= this -> critical
BLUNDER_DROP = 200          # eval drop against mover -> blunder (critical)
BRILLIANT_GAP = 300         # only-move gap to 2nd best -> brilliant (critical)
SAC_MIN_MATERIAL = 3        # >= a minor piece given up -> candidate sacrifice

# --- LLM (plug in when ready). Leave commentary offline-templated if unset. ---
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))
LIGHT_MODEL = "gpt-4o-mini"   # routine moves: cheap/fast
DEEP_MODEL = "gpt-4o"         # critical moves: strong

# --- Register intensity by severity label (tone is derived, not user-picked) ---
INTENSITY = {"routine": 0.2, "notable": 0.55, "critical": 0.9}

# --- TTS (M5). Fish Audio is primary (set FISHAUDIO_API_KEY) — a distinct
# voice PAIR per persona (not just per speaker role), from the community
# voice library; falls back to edge-tts (local/free, no key) when unset or a
# call fails. Only 3 confirmed-Taiwan-accent voices found with a clear
# distinct character (2 male, 1 female) — "文藝" reuses the energetic female
# with much slower/calmer prosody rather than an unverified 4th voice; swap
# in a better reference_id if you find one you like more (fish.audio voice
# library, reference_id is the hex in the /m/<hex>/ URL). ---
FISHAUDIO_API_KEY = os.environ.get("FISHAUDIO_API_KEY")
FISHAUDIO_VOICES = {
    "calm": {"play_by_play": "d4494d578101483795e7e6b9d6b4810e",   # 台灣年輕高個男子：clear/measured/warm, confirmed Taiwan accent (user-picked)
             "analyst": "2d16cee8017e4ff2b2977790344b5abd"},       # 台灣女生：calm/clear/professional, confirmed Taiwan accent (user-picked)
    "excited": {"play_by_play": "b70857799ddc4a1981c616e631c8e225",  # 台湾女：energetic/announcer
                "analyst": "502da0830fef4ba9ab3c203ffe929819"},     # 台：high-energy sports/gaming announcer (user-picked)
    "literary": {"play_by_play": "f6d2bc657fc74343bac02327e7f5e1cd",  # 台灣男：warm/professional (moved here from 熱血)
                 "analyst": "c82f963136f34bccab7108e341061acc"},     # 台湾女：sweet/gentle, confirmed Taiwan accent (user-picked)
}
TTS_VOICES = {"play_by_play": "zh-TW-YunJheNeural", "analyst": "zh-TW-HsiaoChenNeural"}
