"""對弈模式 Web 後端 — FastAPI + 非同步播報。

    uvicorn server:app
    # 開 http://localhost:8000

架構（Phase A/B/C）：
- 棋局時間軸與播報時間軸分離：/api/move 立刻回棋盤，不等 LLM/TTS
- 播報由背景 worker 執行緒跑完整 LangGraph 管線，逐句（utterance）
  經 SSE（/api/stream）推給前端；critical 事件帶 flush=true 讓前端切斷舊語音
- worker 落後時自動跳過過時例行步的生成（真人也會跳過不重要的棋）
- 冷場（前端音訊佇列空 >5s）由前端打 /api/filler 觸發閒聊
"""
import base64
import json
import queue
import threading

import chess
import chess.svg
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

import config
import tts
from engine.stockfish_client import StockfishClient
from graph.nodes.booth import generate_filler
from pipeline import make_stepper

app = FastAPI(title="Chess Broadcaster")

WEB_DIR = Path(__file__).parent / "web"

_engine = StockfishClient()          # python-chess SimpleEngine is thread-safe
_step, _flow = make_stepper(_engine)

# ── SSE hub ────────────────────────────────────────────────────────────────────
_subs: list[queue.Queue] = []
_subs_lock = threading.Lock()


def publish(evt: dict):
    data = json.dumps(evt, ensure_ascii=False)
    with _subs_lock:
        for q in list(_subs):
            try:
                q.put_nowait(data)
            except Exception:
                pass


@app.get("/api/stream")
def stream():
    q: queue.Queue = queue.Queue(maxsize=256)
    with _subs_lock:
        _subs.append(q)

    def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    data = q.get(timeout=15)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _subs_lock:
                if q in _subs:
                    _subs.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── Game (live board — advances immediately, never waits for commentary) ──────
_lock = threading.Lock()


def _default_opts() -> dict:
    return {"vs_engine": True, "human_is_white": True, "depth": 8,
            "persona": "calm", "tts_on": True}


GAME = {"id": 0, "board": chess.Board(), "sans": [], "feed": [],
        "opts": _default_opts()}


def _engine_color() -> chess.Color | None:
    if not GAME["opts"]["vs_engine"]:
        return None
    return chess.BLACK if GAME["opts"]["human_is_white"] else chess.WHITE


def _snapshot() -> dict:
    board: chess.Board = GAME["board"]
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "legal_moves": [m.uci() for m in board.legal_moves],
        "last_move": board.peek().uci() if board.move_stack else None,
        "is_check": board.is_check(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "is_checkmate": board.is_checkmate(),
        "sans": GAME["sans"],
        "engine_to_move": (GAME["opts"]["vs_engine"] and not board.is_game_over()
                           and board.turn == _engine_color()),
        "feed": GAME["feed"],
        "mock_engine": _engine.mock,
        "tts_available": config.USE_LLM,
        "flow": _flow,
        "opts": GAME["opts"],
    }


# ── Commentary worker (owns its own pipeline state; serializes generation) ───
_jobs: queue.Queue = queue.Queue()


def _fresh_pipe_state() -> dict:
    # NOTE: keys must exist in graph.state.BroadcastState — LangGraph only
    # carries declared channels, undeclared keys are dropped by _step()
    return {"board": chess.Board(), "move_history": [], "said_so_far": []}


def _emit(entry: dict):
    """Publish one feed entry over SSE and remember it for page reloads."""
    with _lock:
        GAME["feed"].append(entry)
    publish({"type": "utterance", **entry})


def _speak(text: str, speaker: str, intensity: float, opts: dict):
    if not opts.get("tts_on"):
        return None
    clip = tts.speak(text, speaker, intensity)
    return base64.b64encode(clip).decode() if clip else None


def _worker():
    pipe = _fresh_pipe_state()
    chat_log: list[dict] = []     # broadcast-session memory, NOT pipeline state
    chatting = False              # a chatter session is in progress

    def handle(kind: str, gid: int, payload):
        nonlocal pipe, chat_log, chatting

        if kind == "reset":
            pipe = _fresh_pipe_state()
            chat_log, chatting = [], False
            return

        opts = GAME["opts"]

        if kind == "filler":
            if not _jobs.empty() or not pipe["move_history"]:
                return                        # real work pending / nothing to chat about
            turns = generate_filler(pipe, opts["persona"], chat_log)
            for t in turns:
                if gid != GAME["id"]:
                    return
                audio = _speak(t["text"], t["speaker"], 0.25, opts)
                _emit({"severity": "filler", "route": "filler",
                       "speaker": t["speaker"], "text": t["text"],
                       "audio": audio, "flush": False})
                chat_log.append(t)
            del chat_log[:-16]
            chatting = bool(turns)
            return

        # kind == "move": run the full pipeline on the worker's own board
        mv = chess.Move.from_uci(payload)
        pipe["last_move"] = mv
        pipe["persona"] = opts["persona"]
        # behind schedule? skip generation for this move, keep state coherent
        pipe["skip_generation"] = not _jobs.empty()
        try:
            pipe = _step(pipe)
        except Exception:
            import logging
            logging.exception("pipeline failed for %s", payload)
            pipe["board"].push(mv)            # keep the worker board in sync
            pipe.setdefault("move_history", []).append(mv)
            return
        pipe["move_history"].append(mv)

        ev = pipe["event"]
        a = pipe["analysis"]
        publish({"type": "eval", "score_cp": a["score_cp"],
                 "delta_cp": a["delta_cp"], "mate_in": a["mate_in"],
                 "phase": ev["phase"], "san": a["played_san"]})

        commentary = pipe["commentary"]
        label = ev["severity_label"]

        # routine moves stay silent (muted feed line only) — dead air is
        # covered by the chatter session, which speaks in full sentences
        if not commentary:
            _emit({"severity": label, "route": pipe["route"],
                   "speaker": None, "text": a["played_san"],
                   "audio": None, "flush": False})
            return

        # real commentary ends the chatter session (notable queues politely,
        # critical barges in and cuts the audio)
        chatting = False
        intensity = pipe.get("register_intensity", 0.2)
        flush = label == "critical"
        for i, turn in enumerate(commentary):
            if gid != GAME["id"]:
                return
            audio = _speak(turn["text"], turn["speaker"], intensity, opts)
            _emit({"severity": label, "route": pipe["route"],
                   "speaker": turn["speaker"], "text": turn["text"],
                   "audio": audio, "flush": flush and i == 0})
            chat_log.append(turn)             # chat continues from what was said
        del chat_log[:-16]

    while True:
        kind, gid, payload = _jobs.get()
        if gid != GAME["id"]:
            continue                          # stale job from a previous game
        try:
            handle(kind, gid, payload)
        except Exception:
            import logging
            logging.exception("worker job %s failed", kind)  # never kill the thread


threading.Thread(target=_worker, daemon=True).start()


# ── API ────────────────────────────────────────────────────────────────────────
class NewGameReq(BaseModel):
    vs_engine: bool = True
    human_is_white: bool = True
    depth: int = 8
    persona: str = "calm"
    tts_on: bool = True     # 語音一律開啟（沒有 API key 時自動退化成純文字）


class MoveReq(BaseModel):
    uci: str


class OptsReq(BaseModel):
    persona: str | None = None
    tts_on: bool | None = None
    depth: int | None = None


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/pieces")
def pieces():
    return {sym: chess.svg.piece(chess.Piece.from_symbol(sym))
            for sym in "PNBRQKpnbrqk"}


@app.get("/api/state")
def state():
    with _lock:
        return _snapshot()


@app.post("/api/new")
def new_game(req: NewGameReq):
    with _lock:
        GAME["id"] += 1
        GAME["board"] = chess.Board()
        GAME["sans"] = []
        GAME["feed"] = []
        GAME["opts"] = req.model_dump()
        gid = GAME["id"]
    _jobs.put(("reset", gid, None))
    publish({"type": "reset"})
    # engine plays first when the human took black
    if req.vs_engine and not req.human_is_white:
        _engine_move(gid)
    with _lock:
        return _snapshot()


@app.post("/api/opts")
def set_opts(req: OptsReq):
    with _lock:
        for k, v in req.model_dump().items():
            if v is not None:
                GAME["opts"][k] = v
        return {"opts": GAME["opts"], "tts_available": config.USE_LLM}


def _push_move(mv: chess.Move, gid: int):
    """Advance the live board and queue commentary. Caller holds no lock."""
    with _lock:
        board: chess.Board = GAME["board"]
        GAME["sans"].append(board.san(mv))
        board.push(mv)
    _jobs.put(("move", gid, mv.uci()))


def _engine_move(gid: int):
    with _lock:
        board = GAME["board"].copy()
        depth = GAME["opts"]["depth"]
    mv = _engine.best_move(board, depth)
    if mv:
        _push_move(mv, gid)


@app.post("/api/move")
def play_move(req: MoveReq):
    with _lock:
        gid = GAME["id"]
        board: chess.Board = GAME["board"]
        try:
            mv = chess.Move.from_uci(req.uci)
        except ValueError:
            return {"error": "bad uci"}
        if mv not in board.legal_moves:
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
        if mv not in board.legal_moves:
            return {"error": "illegal move"}
        vs_engine = GAME["opts"]["vs_engine"]

    _push_move(mv, gid)                       # returns immediately, no LLM wait

    with _lock:
        over = GAME["board"].is_game_over()
    if vs_engine and not over:
        _engine_move(gid)                     # opponent "thinking" is game time

    with _lock:
        return _snapshot()


@app.post("/api/filler")
def filler():
    """Client reports dead air; queue one chatter job (worker skips it if busy)."""
    with _lock:
        gid = GAME["id"]
        over = GAME["board"].is_game_over()
        started = bool(GAME["sans"])
    if started and not over and _jobs.empty():
        _jobs.put(("filler", gid, None))
        return {"queued": True}
    return {"queued": False}


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
