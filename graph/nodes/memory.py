"""Layered memory.

MVP ships layer 1 (short-term game memory) working, and a stub for layer 2
(chess theory via ChromaDB) with a clear integration point. Layer 3 (player
profile) is future work.
"""


def retrieve_memory_node(state: dict) -> dict:
    said = state.get("said_so_far", [])
    callbacks = said[-3:]                      # recent key moments for call-backs

    # TODO(layer 2): query ChromaDB with current FEN/position embedding for theory.
    #   from memory.chroma_store import query_theory
    #   theory = query_theory(state["board"].fen(), k=2)
    theory = []

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
