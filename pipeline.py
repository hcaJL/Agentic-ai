"""Sequential driver for the initial version. Same nodes drop into LangGraph later
(see graph/build_graph.py). Keeps the initial version runnable with minimal deps.
"""
import chess
import chess.pgn
from graph.nodes.perception import perception_node
from graph.nodes.event_detector import event_detector_node
from graph.nodes.director import director_node
from graph.nodes.memory import retrieve_memory_node, update_memory_node
from graph.nodes.booth import light_commentary_node, booth_node


def process_move(state: dict, engine, persona: str = "calm") -> dict:
    """Run one move through the full pipeline. `state['board']` must be the position
    BEFORE the move, with `state['last_move']` set."""
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
    board = chess.Board(start_fen) if start_fen else chess.Board()
    state = {"board": board, "move_history": [], "said_so_far": []}
    for mv in moves:
        state["last_move"] = mv
        state = process_move(state, engine, persona)
        state["move_history"].append(mv)
        yield mv, state


def moves_from_pgn(path: str):
    with open(path) as f:
        game = chess.pgn.read_game(f)
    return list(game.mainline_moves())
