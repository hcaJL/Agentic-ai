"""Layered memory.

Layer 1 (short-term game memory, for call-backs) lives in state["said_so_far"].
Layer 2 (chess theory) is vector retrieval from ChromaDB — see memory/chroma_store.
Layer 3 (player profile) is future work.
"""
import chess


def _theory_query(state: dict) -> str:
    """Describe the current moment in the pipeline's English event vocabulary —
    the same vocabulary the theory corpus keywords are written in."""
    ev = state["event"]
    board: chess.Board = state["board"]           # AFTER the move
    move: chess.Move = state["last_move"]
    piece = board.piece_at(move.to_square)
    bits = ev["types"] + ev["motifs"] + [ev["phase"]]
    if piece:
        bits.append(chess.piece_name(piece.piece_type))
    bits.append(chess.square_name(move.to_square))
    if ev["phase"] == "opening":
        # early SANs help match named-opening entries (e4 e5 Nf3 ...)
        b = chess.Board()
        for mv in state.get("move_history", [])[:8]:
            bits.append(b.san(mv))
            b.push(mv)
    return " ".join(bits)


def retrieve_memory_node(state: dict) -> dict:
    said = state.get("said_so_far", [])
    callbacks = said[-3:]                      # recent key moments for call-backs

    from memory.chroma_store import query_theory
    theory = query_theory(_theory_query(state), k=2)

    state["retrieved_memory"] = {"callbacks": callbacks, "theory": theory}
    return state


def update_memory_node(state: dict) -> dict:
    """Write this move's key moment back to short-term memory (for consistency + call-backs)."""
    ev = state["event"]
    if ev["severity_label"] != "routine":
        a = state["analysis"]
        mv = state["facts"]["move"]["san"] if state.get("facts") else state["last_move"].uci()
        summary = f"#{state['board'].fullmove_number} {mv}: {'/'.join(ev['types']) or 'notable'} ({a['score_cp']/100:+.1f})"
        state.setdefault("said_so_far", []).append(summary)
    return state
