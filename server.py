"""對弈模式 Web 後端 — FastAPI 版，取代 Streamlit 的 play.py。

    uvicorn server:app --reload
    # 然後開 http://localhost:8000

前端（web/index.html）只在走棋時打 API，棋盤由 JS 原地更新 → 零閃爍。
每一步都走完整 LangGraph 管線（感知 → 事件 → Director → 記憶 → 主播台），
解說、嚴重度分級、TTS 跟 Streamlit 版完全一致。
"""
import base64
import threading

import chess
import chess.svg
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

import config
import tts
from engine.stockfish_client import StockfishClient
from pipeline import make_stepper

app = FastAPI(title="Chess Broadcaster")

WEB_DIR = Path(__file__).parent / "web"

# ── Game session (single-user demo: one global game, guarded by a lock) ───────
_lock = threading.Lock()
_engine = StockfishClient()
_step, _flow = make_stepper(_engine)


def _fresh_game(opts: dict) -> dict:
    return {
        "state": {"board": chess.Board(), "move_history": [], "said_so_far": []},
        "feed": [],
        "opts": opts,
    }


GAME = _fresh_game({"vs_engine": True, "human_is_white": True, "depth": 8,
                    "persona": "calm", "tts_on": True, "verbose": True})


# ── Pipeline: play one move, collect commentary + audio ───────────────────────
def _play_one(mv: chess.Move) -> list[dict]:
    """Push one move through the full pipeline. Returns new feed entries."""
    opts = GAME["opts"]
    s = GAME["state"]
    s["last_move"] = mv
    s["persona"] = opts["persona"]
    s = _step(s)
    s["move_history"].append(mv)
    GAME["state"] = s
    ev = s["event"]

    commentary = s["commentary"]
    if not commentary and opts["verbose"]:
        commentary = [{"speaker": "play_by_play",
                       "text": s["analysis"]["played_san"]}]

    entries = []
    if not commentary:
        entries.append({"severity": ev["severity_label"], "route": s["route"],
                        "speaker": None, "text": s["analysis"]["played_san"],
                        "audio": None})
    for turn in commentary:
        audio_b64 = None
        if opts["tts_on"]:
            clip = tts.speak(turn["text"], turn["speaker"],
                             s.get("register_intensity", 0.2))
            if clip:
                audio_b64 = base64.b64encode(clip).decode()
        entries.append({"severity": ev["severity_label"], "route": s["route"],
                        "speaker": turn["speaker"], "text": turn["text"],
                        "audio": audio_b64})
    GAME["feed"].extend(entries)
    return entries


def _engine_color() -> chess.Color | None:
    opts = GAME["opts"]
    if not opts["vs_engine"]:
        return None
    return chess.BLACK if opts["human_is_white"] else chess.WHITE


def _snapshot(new_entries: list[dict] | None = None) -> dict:
    s = GAME["state"]
    board: chess.Board = s["board"]
    a = s.get("analysis")

    # SAN move list for the scoresheet
    tmp = chess.Board()
    sans = []
    for m in s["move_history"]:
        try:
            sans.append(tmp.san(m)); tmp.push(m)
        except Exception:
            break

    return {
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "legal_moves": [m.uci() for m in board.legal_moves],
        "last_move": board.peek().uci() if board.move_stack else None,
        "is_check": board.is_check(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "is_checkmate": board.is_checkmate(),
        "eval": ({"score_cp": a["score_cp"], "delta_cp": a["delta_cp"],
                  "mate_in": a["mate_in"]} if a else None),
        "phase": s.get("event", {}).get("phase") if s.get("event") else None,
        "sans": sans,
        "engine_to_move": (GAME["opts"]["vs_engine"] and not board.is_game_over()
                           and board.turn == _engine_color()),
        "feed": GAME["feed"],
        "new_entries": new_entries or [],
        "mock_engine": _engine.mock,
        "tts_available": config.USE_LLM,
        "flow": _flow,
        "opts": GAME["opts"],
    }


# ── API ────────────────────────────────────────────────────────────────────────
class NewGameReq(BaseModel):
    vs_engine: bool = True
    human_is_white: bool = True
    depth: int = 8
    persona: str = "calm"
    tts_on: bool = True     # 語音一律開啟（沒有 API key 時自動退化成純文字）
    verbose: bool = True    # 每步都播報


class MoveReq(BaseModel):
    uci: str


class OptsReq(BaseModel):
    """Live-tunable options; opponent/color still require a new game."""
    persona: str | None = None
    tts_on: bool | None = None
    verbose: bool | None = None
    depth: int | None = None


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/pieces")
def pieces():
    """SVG for the 12 piece types, rendered once by python-chess."""
    out = {}
    for sym in "PNBRQKpnbrqk":
        piece = chess.Piece.from_symbol(sym)
        out[sym] = chess.svg.piece(piece)
    return out


@app.get("/api/state")
def state():
    with _lock:
        return _snapshot()


@app.post("/api/new")
def new_game(req: NewGameReq):
    global GAME
    with _lock:
        GAME = _fresh_game(req.model_dump())
        entries = []
        # engine plays first when the human took black
        if req.vs_engine and not req.human_is_white:
            mv = _engine.best_move(GAME["state"]["board"], req.depth)
            if mv:
                entries = _play_one(mv)
        return _snapshot(entries)


@app.post("/api/opts")
def set_opts(req: OptsReq):
    with _lock:
        for k, v in req.model_dump().items():
            if v is not None:
                GAME["opts"][k] = v
        return {"opts": GAME["opts"], "tts_available": config.USE_LLM}


@app.post("/api/move")
def play_move(req: MoveReq):
    with _lock:
        board: chess.Board = GAME["state"]["board"]
        try:
            mv = chess.Move.from_uci(req.uci)
        except ValueError:
            return {"error": "bad uci"}
        # auto-queen if the client sent a bare promotion move
        if mv not in board.legal_moves:
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
        if mv not in board.legal_moves:
            return {"error": "illegal move"}

        entries = _play_one(mv)

        b2: chess.Board = GAME["state"]["board"]
        opts = GAME["opts"]
        if opts["vs_engine"] and not b2.is_game_over():
            reply = _engine.best_move(b2, opts["depth"])
            if reply:
                entries = entries + _play_one(reply)
        return _snapshot(entries)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
