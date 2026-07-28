"""Perception: update board, shallow-eval before & after, compute delta_cp."""
import chess
import config
from engine.chess_utils import material_balance, game_phase


def perception_node(state: dict, engine) -> dict:
    """Expects state with `board` (position BEFORE the move) and `last_move`.
    Pushes the move and analyses the resulting position."""
    board: chess.Board = state["board"]
    move: chess.Move = state["last_move"]

    before = engine.analyse(board, config.SHALLOW_DEPTH, multipv=config.MULTIPV)
    score_before = before["score_cp"]
    played_san = board.san(move)

    board.push(move)  # board is now AFTER the move
    after = engine.analyse(board, config.SHALLOW_DEPTH, multipv=config.MULTIPV)

    # delta from the mover's perspective (mover = side who just moved = not board.turn)
    mover_white = not board.turn
    raw_delta = after["score_cp"] - score_before
    delta_cp = raw_delta if mover_white else -raw_delta

    analysis = {
        "score_cp": after["score_cp"],
        "score_before_cp": score_before,
        "delta_cp": delta_cp,
        "mate_in": after["mate_in"],
        "best_line_san": after["best_line_san"],
        "top_moves": after["top_moves"],
        "top_moves_before": before["top_moves"],  # mover's own candidates
        "played_san": played_san,
        "depth": after["depth"],
    }
    state["analysis"] = analysis
    state["_phase"] = game_phase(board)
    state["_material"] = material_balance(board)
    return state
