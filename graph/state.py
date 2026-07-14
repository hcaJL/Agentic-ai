"""Shared data structures. Pure dicts so they drop into LangGraph state unchanged."""
from typing import TypedDict, Literal, Optional
import chess


class StockfishAnalysis(TypedDict):
    score_cp: int               # current eval (centipawns, + = White better)
    score_before_cp: int        # eval before the move was played
    delta_cp: int               # change from mover's perspective (negative = worse for mover)
    mate_in: Optional[int]
    best_line_san: list          # principal variation, SAN
    top_moves: list              # multipv: [{"san":..., "score_cp":...}, ...]
    depth: int


class EventInfo(TypedDict):
    types: list                 # e.g. ["sacrifice", "check"]
    motifs: list                # e.g. ["fork"]
    phase: Literal["opening", "middlegame", "endgame"]
    severity_score: float       # 0.0–1.0
    severity_label: Literal["routine", "notable", "critical"]


class FactsPacket(TypedDict):
    """The ONLY thing commentary agents may use. Grounding contract lives here."""
    move: dict
    board: dict
    eval: dict
    event: dict
    memory: dict                # {"callbacks": [...], "theory": [...]} — deep path only
    register: dict              # {"intensity": float, "persona": str}


class BroadcastState(TypedDict, total=False):
    board: chess.Board
    move_history: list
    last_move: Optional[chess.Move]
    analysis: StockfishAnalysis
    event: EventInfo
    route: Literal["light", "deep"]
    facts: FactsPacket
    retrieved_memory: dict
    commentary: list            # [{"speaker":..., "text":...}, ...]
    said_so_far: list           # short-term memory: one-line summaries of past key moments
    persona: str
    register_intensity: float
    # internal scratch keys written by nodes — must be declared so LangGraph
    # keeps them as channels between steps
    _phase: str
    _material: int
    _prev_phase: str
    _light_streak: int
    skip_generation: bool       # async worker: catch up without LLM/TTS this move
