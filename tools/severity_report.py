"""Severity/routing validation harness — runs games through perception →
event_detector → director ONLY (no LLM, no booth) and reports how the
thresholds in config.py classify every move. Use it after touching config
thresholds or event_detector logic.

Usage:
    python tools/severity_report.py games/*.pgn
    python tools/severity_report.py            # defaults to all games/*.pgn
"""
import glob
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.pgn
from engine.stockfish_client import StockfishClient
from graph.nodes.perception import perception_node
from graph.nodes.event_detector import event_detector_node
from graph.nodes.director import director_node


def analyse_game(path: str, engine) -> list:
    with open(path) as f:
        game = chess.pgn.read_game(f)
    board = chess.Board()
    state = {"board": board, "move_history": [], "said_so_far": []}
    rows = []
    for mv in game.mainline_moves():
        san = state["board"].san(mv)
        state["last_move"] = mv
        state = perception_node(state, engine)
        state = event_detector_node(state)
        state = director_node(state)
        a, ev = state["analysis"], state["event"]
        ply = len(state["move_history"]) + 1
        rows.append({
            "ply": ply,
            "move_no": f"{(ply + 1) // 2}{'.' if ply % 2 else '...'}",
            "san": san,
            "delta_cp": a["delta_cp"],
            "score_cp": a["score_cp"],
            "types": ",".join(ev["types"]) or "-",
            "motifs": ",".join(ev["motifs"]) or "-",
            "severity": ev["severity_score"],
            "label": ev["severity_label"],
            "route": state["route"],
        })
        state["move_history"].append(mv)
    return rows


def report(path: str, rows: list):
    print(f"\n=== {os.path.basename(path)} ({len(rows)} plies) ===")
    print(f"{'move':>8} {'san':<8} {'Δcp':>6} {'eval':>6}  {'label':<8} {'route':<5} events")
    for r in rows:
        mark = "  <<" if r["label"] != "routine" else ""
        print(f"{r['move_no']:>8} {r['san']:<8} {r['delta_cp']:>6} {r['score_cp']:>6}"
              f"  {r['label']:<8} {r['route']:<5} {r['types']}"
              f"{' | ' + r['motifs'] if r['motifs'] != '-' else ''}{mark}")
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    deep = sum(1 for r in rows if r["route"] == "deep")
    dist = "  ".join(f"{k}:{v} ({v / len(rows):.0%})" for k, v in
                     sorted(counts.items(), key=lambda kv: ["routine", "notable", "critical"].index(kv[0])))
    print(f"  -> {dist}  |  deep 路由: {deep}/{len(rows)} ({deep / len(rows):.0%})")


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "games", "*.pgn")))
    engine = StockfishClient()
    if engine.mock:
        print("警告：Stockfish 未接上，正在用 mock 引擎 — 這份報告沒有意義。")
    try:
        for p in paths:
            report(p, analyse_game(p, engine))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
