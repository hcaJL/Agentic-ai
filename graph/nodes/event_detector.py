"""Event detection + severity scoring. This drives the whole router — tune in config."""
import chess
import config
from engine.chess_utils import captured_value, detects_fork, is_en_prise, PIECE_VALUE


def event_detector_node(state: dict) -> dict:
    board: chess.Board = state["board"]          # AFTER the move
    move: chess.Move = state["last_move"]
    a = state["analysis"]
    phase = state["_phase"]

    # reconstruct board BEFORE move to inspect the capture victim
    before = board.copy(); before.pop()

    types, motifs = [], []

    # --- forcing / structural ---
    if board.is_checkmate():
        types.append("checkmate")
    elif board.is_check():
        types.append("check")
    if move.promotion:
        types.append("promotion")

    # --- material ---
    cap = captured_value(before, move)
    if cap >= PIECE_VALUE[chess.ROOK]:
        types.append("major_capture")
    elif cap > 0:
        types.append("capture")

    # sacrifice: piece left hanging (>= a minor) but eval held/improved
    hanging, pv = is_en_prise(board, move)
    if hanging and pv >= config.SAC_MIN_MATERIAL and cap < pv and a["delta_cp"] >= -30:
        types.append("sacrifice")

    # --- eval-based ---
    if a["delta_cp"] <= -config.BLUNDER_DROP:
        types.append("blunder")
    if len(a["top_moves"]) >= 2:
        gap = abs(a["top_moves"][0]["score_cp"] - a["top_moves"][1]["score_cp"])
        if gap >= config.BRILLIANT_GAP and a["delta_cp"] >= -30:
            types.append("brilliant")
    if a["mate_in"] is not None:
        types.append("mate_sequence")

    # --- phase ---
    if phase == "endgame" and state.get("_prev_phase") != "endgame":
        types.append("entering_endgame")

    # --- tactical motif (MVP: fork only) ---
    if detects_fork(board, move):
        motifs.append("fork")

    score = _severity_score(a["delta_cp"], types, motifs)
    label = "critical" if score >= 0.8 else "notable" if score >= 0.4 else "routine"

    state["event"] = {"types": types, "motifs": motifs, "phase": phase,
                      "severity_score": round(score, 2), "severity_label": label}
    state["_prev_phase"] = phase
    return state


CRITICAL_EVENTS = {"checkmate", "blunder", "sacrifice", "promotion",
                   "entering_endgame", "brilliant", "mate_sequence"}
NOTABLE_EVENTS = {"check", "major_capture"}


def _severity_score(delta_cp: int, types: list, motifs: list) -> float:
    score = min(abs(delta_cp) / 300.0, 1.0)
    if any(t in CRITICAL_EVENTS for t in types):
        score = max(score, 0.9)
    if any(t in NOTABLE_EVENTS for t in types) or motifs:
        score = max(score, 0.55)
    return score
