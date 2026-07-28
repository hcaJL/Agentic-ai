"""Console runner. Plays a PGN (or the built-in demo game) and prints commentary.

    python demo_cli.py                  # built-in short game, calm persona
    python demo_cli.py games/demo.pgn excited
"""
import sys
from engine.stockfish_client import StockfishClient
from pipeline import run_game, moves_from_pgn
import chess


DEMO_SAN = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5",
            "exd5", "Nxd5", "Nxf7", "Kxf7", "Qf3+", "Ke6", "Nc3"]


def san_to_moves(san_list):
    b, moves = chess.Board(), []
    for s in san_list:
        m = b.parse_san(s); moves.append(m); b.push(m)
    return moves


def main():
    persona = "calm"
    if len(sys.argv) >= 2:
        moves = moves_from_pgn(sys.argv[1])
        if len(sys.argv) >= 3:
            persona = sys.argv[2]
    else:
        moves = san_to_moves(DEMO_SAN)

    engine = StockfishClient()
    mode = "MOCK engine (set STOCKFISH_PATH for real eval)" if engine.mock else "Stockfish"
    try:
        import langgraph  # noqa: F401
        flow = "LangGraph"
    except ImportError:
        flow = "sequential fallback"
    print(f"=== Chess Broadcaster — {mode}, {flow}, persona={persona} ===\n")

    for mv, state in run_game(moves, engine, persona):
        ev = state["event"]
        tag = f"[{ev['severity_label']:8}|{state['route']:5}]"
        if state["commentary"]:
            for turn in state["commentary"]:
                print(f"{tag} ({turn['speaker']}) {turn['text']}")
        else:
            print(f"{tag} … (silent)")
    engine.close()


if __name__ == "__main__":
    main()
