"""Central config. Tune SEVERITY thresholds here — they drive the whole router."""
import os

# --- Stockfish ---
# Point this at your local Stockfish binary. Download: https://stockfishchess.org/download/
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
SHALLOW_DEPTH = 12          # baseline eval, every move (fast/cheap)
DEEP_DEPTH = 20             # deep path only (critical moves)
MULTIPV = 3                 # top-N lines

# --- Severity thresholds (centipawns). 100cp ~= one pawn. TUNE THESE. ---
DELTA_NOTABLE = 50          # |delta| >= this -> at least notable
DELTA_CRITICAL = 150        # |delta| >= this -> critical
BLUNDER_DROP = 200          # eval drop against mover -> blunder (critical)
BRILLIANT_GAP = 150         # only-move gap to 2nd best -> brilliant (critical)
SAC_MIN_MATERIAL = 3        # >= a minor piece given up -> candidate sacrifice

# --- LLM (plug in when ready). Leave commentary offline-templated if unset. ---
USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))
LIGHT_MODEL = "claude-haiku-4-5-20251001"   # routine moves: cheap/fast
DEEP_MODEL = "claude-sonnet-4-6"            # critical moves: strong

# --- Register intensity by severity label (tone is derived, not user-picked) ---
INTENSITY = {"routine": 0.2, "notable": 0.55, "critical": 0.9}
