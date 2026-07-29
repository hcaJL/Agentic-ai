"""Per-move driver. Prefers the LangGraph app (graph/build_graph.py); falls back
to running the same nodes sequentially when langgraph isn't installed, so the
project stays runnable with minimal deps.
"""
import math
import re

import chess
import chess.pgn
from graph.nodes.perception import perception_node
from graph.nodes.event_detector import event_detector_node
from graph.nodes.director import director_node
from graph.nodes.memory import retrieve_memory_node, update_memory_node
from graph.nodes.booth import light_commentary_node, booth_node


def make_stepper(engine, persona: str = "calm"):
    """Return (step_fn, mode). step_fn(state) runs one move through the pipeline.
    mode is "langgraph", or "sequential" when langgraph isn't installed."""
    try:
        from graph.build_graph import build_graph
        app = build_graph(engine, persona)
        return (lambda state: app.invoke(state)), "langgraph"
    except ImportError:
        return (lambda state: process_move(state, engine,
                                           state.get("persona", persona))), "sequential"


def process_move(state: dict, engine, persona: str = "calm") -> dict:
    """Sequential fallback: run one move through the same nodes LangGraph wires up.
    `state['board']` must be the position BEFORE the move, with `state['last_move']` set."""
    state = perception_node(state, engine)        # board is now AFTER the move
    state = event_detector_node(state)
    state = director_node(state)

    if state["route"] == "deep":
        state = retrieve_memory_node(state)
        state = booth_node(state, persona)
    else:
        state = light_commentary_node(state, persona)

    state = update_memory_node(state)
    return state


def run_game(moves, engine, persona: str = "calm", start_fen: str = None):
    """Generator: yields (move, state) after each move so a UI can render live."""
    step, _ = make_stepper(engine, persona)
    board = chess.Board(start_fen) if start_fen else chess.Board()
    state = {"board": board, "move_history": [], "said_so_far": [], "persona": persona}
    for mv in moves:
        state["last_move"] = mv
        state = step(state)
        state["move_history"].append(mv)
        yield mv, state


def moves_from_pgn(path: str):
    with open(path) as f:
        game = chess.pgn.read_game(f)
    return list(game.mainline_moves())


# ---------- real per-move timing, recovered from [%clk] annotations ----------
def _time_control(tc: str) -> tuple[float, float]:
    """(base seconds, increment seconds) from a PGN TimeControl header."""
    m = re.match(r"(\d+)(?:\+(\d+))?", tc or "")
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2) or 0)


def timed_moves_from_pgn(path: str) -> list[tuple[chess.Move, float | None]]:
    """[(move, seconds the mover spent on it)] — seconds is None when the PGN
    carries no usable clock for that move.

    Clocks are the remaining time AFTER the move, so a move's cost is
    (my clock last move) - (my clock now) + (increment I just earned).

    Each side's FIRST move has no previous clock to subtract from, and the
    TimeControl header can't stand in for it — some PGNs start stamping clocks
    at a value that doesn't match the advertised base time (Polgar/Rudolf's
    first %clk is well under its "6600", which would invent a 61-minute think).
    Those two moves simply report None.

    Implausible values are dropped rather than trusted for the same reason: a
    simul scoresheet may stamp a clock only once, and the arithmetic then
    attributes the entire session to a single move. Anything past 80% of the
    base time is treated as a recording artefact, not a think.
    """
    with open(path) as f:
        game = chess.pgn.read_game(f)
    base, inc = _time_control(game.headers.get("TimeControl", ""))
    ceiling = base * 0.8 if base else None

    out: list[tuple[chess.Move, float | None]] = []
    prev: dict[bool, float] = {}          # color -> clock after that side's previous move
    for node in game.mainline():
        color = not node.board().turn     # the side that just moved
        clk = node.clock()
        spent = None
        if clk is not None:
            before = prev.get(color)
            if before is not None:
                spent = max(0.0, before - clk + inc)
                if ceiling is not None and spent > ceiling:
                    spent = None          # recording artefact — don't pretend it's a think
            prev[color] = clk
        out.append((node.move, spent))
    return out


# Real seconds -> replay pacing. sqrt keeps the *relative* feel of a long think
# without actually sitting through it, and the clamp keeps a whole game inside a
# watchable window: 1s -> 1.5s, 5s -> 3.4s, 26s -> 7.6s, 55s -> 11.1s, 469s -> 12s.
PACE_K, PACE_MIN_S, PACE_MAX_S = 1.5, 1.5, 12.0


def pace_seconds(spent: float | None) -> float:
    """How long the replay should linger before playing a move that really took
    `spent` seconds. The booth still gets to finish its line either way — the
    client advances on max(this, commentary finished)."""
    if not spent or spent <= 0:
        return PACE_MIN_S
    return max(PACE_MIN_S, min(PACE_MAX_S, PACE_K * math.sqrt(spent)))
