"""Thin wrapper around a local Stockfish via python-chess UCI.

Falls back to a deterministic MOCK engine when Stockfish isn't available, so the
rest of the pipeline runs offline. Replace nothing — just set STOCKFISH_PATH.
"""
import chess
import chess.engine
import config


def _to_cp(score: chess.engine.PovScore) -> tuple:
    """Return (score_cp_white_pov, mate_in or None)."""
    white = score.white()
    if white.is_mate():
        m = white.mate()
        return (10000 if m > 0 else -10000, m)
    return (white.score(), None)


class StockfishClient:
    def __init__(self, path: str = None):
        self.path = path or config.STOCKFISH_PATH
        self._engine = None
        self.mock = False
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        except Exception:
            self.mock = True  # no binary -> mock mode

    def analyse(self, board: chess.Board, depth: int, multipv: int = 1) -> dict:
        if self.mock:
            return self._mock_analyse(board, depth, multipv)
        info = self._engine.analyse(
            board, chess.engine.Limit(depth=depth), multipv=multipv
        )
        lines = info if isinstance(info, list) else [info]
        best = lines[0]
        score_cp, mate_in = _to_cp(best["score"])
        pv = best.get("pv", [])
        best_line_san = self._pv_to_san(board, pv[:6])
        top = []
        for ln in lines:
            cp, _ = _to_cp(ln["score"])
            pv0 = ln.get("pv", [])
            top.append({"san": board.san(pv0[0]) if pv0 else "", "score_cp": cp})
        return {"score_cp": score_cp, "mate_in": mate_in,
                "best_line_san": best_line_san, "top_moves": top, "depth": depth}

    def _pv_to_san(self, board: chess.Board, pv: list) -> list:
        out, b = [], board.copy()
        for mv in pv:
            try:
                out.append(b.san(mv)); b.push(mv)
            except Exception:
                break
        return out

    def _mock_analyse(self, board, depth, multipv):
        # Deterministic stand-in: eval ~ material, so the pipeline is exercisable offline.
        from engine.chess_utils import material_balance
        cp = material_balance(board) * 100
        legal = list(board.legal_moves)
        top = [{"san": board.san(m), "score_cp": cp} for m in legal[:multipv]]
        return {"score_cp": cp, "mate_in": None,
                "best_line_san": [board.san(legal[0])] if legal else [],
                "top_moves": top, "depth": depth}

    def close(self):
        if self._engine:
            self._engine.quit()
