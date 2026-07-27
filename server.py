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
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.pgn
import chess.svg
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

import config
import tts
from engine.stockfish_client import StockfishClient
from graph.nodes.booth import (generate_closing, generate_filler,
                               generate_opening, generate_recap)
from pipeline import make_stepper, moves_from_pgn

app = FastAPI(title="Chess Broadcaster")

WEB_DIR = Path(__file__).parent / "web"

_engine = StockfishClient()          # position analysis for the commentary pipeline (worker thread)
_opp_engine = StockfishClient()      # move selection for the AI opponent (request thread)
# Separate engine PROCESSES on purpose: a single shared SimpleEngine serializes
# every analyse() call behind one lock (see stockfish_client.py), so the worker's
# per-move eval and the opponent's move-picking would queue behind each other
# instead of running concurrently — that's what made analysis feel sluggish
# after the lock was added to stop them from cancelling each other.
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
            "persona": "calm", "tts_on": True, "mode": "play"}


GAME = {"id": 0, "board": chess.Board(), "sans": [], "feed": [],
        "opts": _default_opts(), "replay": None}   # replay = {moves, idx} when replaying


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
        "llm_available": config.USE_LLM,   # text generation only — TTS (edge-tts) needs no key
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


def _feedback_ctx(persona: str, intensity: float, facts: dict | None, san=None) -> dict:
    """Context the client echoes back with a 👍/👎 so an approved line lands in
    the right persona bucket with the moment's event tags (see feedback_store)."""
    ev = (facts or {}).get("event", {})
    tags = list(ev.get("types", [])) + list(ev.get("motifs", []))
    if ev.get("severity_label"):
        tags.append(ev["severity_label"])
    return {"persona": persona, "intensity": intensity, "tags": tags, "san": san}


def _speak(text: str, speaker: str, intensity: float, opts: dict):
    """Returns (base64_audio, format) — format is "wav" or "mp3" depending on
    which TTS provider served this line — or (None, None)."""
    if not opts.get("tts_on"):
        return None, None
    result = tts.speak(text, speaker, intensity, opts.get("persona", "calm"))
    if not result:
        return None, None
    clip, fmt = result
    return base64.b64encode(clip).decode(), fmt


def _speak_stream(turns: list[dict], intensity: float, opts: dict):
    """Synthesize every turn's audio concurrently, yielding (turn, audio, fmt)
    in script order as each finishes — not after the whole batch completes.
    All turns start synthesizing immediately, so the first line goes out
    after ~1 TTS round-trip instead of waiting for the slowest line in a
    multi-turn critical script (that wait was the main cause of the booth
    feeling like it starts talking late)."""
    if not turns:
        return
    with ThreadPoolExecutor(max_workers=len(turns)) as pool:
        futures = [pool.submit(_speak, t["text"], t["speaker"], intensity, opts)
                   for t in turns]
        for turn, fut in zip(turns, futures):
            audio, fmt = fut.result()
            yield turn, audio, fmt


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

        if kind == "opening":
            turns = generate_opening(opts, opts.get("persona", "calm"))
            ctx = _feedback_ctx(opts["persona"], 0.4, None)
            for i, (turn, audio, fmt) in enumerate(_speak_stream(turns, 0.4, opts)):
                if gid != GAME["id"]:
                    return
                _emit({"severity": "notable", "route": "opening",
                       "speaker": turn["speaker"], "text": turn["text"], "ctx": ctx,
                       "audio": audio, "audio_format": fmt, "flush": i == 0, "ply": 0})
                chat_log.append(turn)
            del chat_log[:-16]
            return

        if kind == "filler":
            if not _jobs.empty() or not pipe["move_history"]:
                return                        # real work pending / nothing to chat about
            ply = len(pipe["move_history"])
            turns = generate_filler(pipe, opts["persona"], chat_log)
            ctx = _feedback_ctx(opts["persona"], 0.25, None)
            for t, audio, fmt in _speak_stream(turns, 0.25, opts):
                if gid != GAME["id"]:
                    return
                _emit({"severity": "filler", "route": "filler",
                       "speaker": t["speaker"], "text": t["text"], "ctx": ctx,
                       "audio": audio, "audio_format": fmt, "flush": False, "ply": ply})
                chat_log.append(t)
            del chat_log[:-16]
            chatting = bool(turns)
            return

        # kind == "move": drain any backlog into a batch. One move = normal
        # per-move script; several moves = ONE compressed「剛剛…」recap
        # focused on the batch's key move (like real commentators catching up).
        batch = [payload]
        while True:
            try:
                nxt = _jobs.get_nowait()
            except queue.Empty:
                break
            k2, g2, p2 = nxt
            if k2 == "move" and g2 == gid:
                batch.append(p2)
            elif k2 == "filler":
                continue                      # chatter about an old position: drop
            else:
                _jobs.put(nxt)                # reset etc. — handle after the batch
                break

        solo = len(batch) == 1
        infos, key = [], None                 # per-move digest; most important one
        _RANK = {"routine": 0, "notable": 1, "critical": 2}
        for uci in batch:
            mv = chess.Move.from_uci(uci)
            pipe["last_move"] = mv
            pipe["persona"] = opts["persona"]
            pipe["skip_generation"] = not solo
            # "3-5" turns roughly doubles the LLM's generation time over "2-3"
            # (~3.9s vs ~1.8s measured) — that's the single biggest lever on
            # how long it takes before ANY voice comes out after a move, so
            # real-time (queue-empty) events get the short form too now, not
            # just backlogged ones.
            pipe["dialogue_turns"] = "2-3"
            try:
                pipe = _step(pipe)
            except Exception:
                import logging
                logging.exception("pipeline failed for %s", uci)
                pipe["board"].push(mv)        # keep the worker board in sync
                pipe.setdefault("move_history", []).append(mv)
                continue
            pipe["move_history"].append(mv)
            ply = len(pipe["move_history"])

            ev = pipe["event"]
            a = pipe["analysis"]
            publish({"type": "eval", "score_cp": a["score_cp"],
                     "delta_cp": a["delta_cp"], "mate_in": a["mate_in"],
                     "phase": ev["phase"], "san": a["played_san"], "ply": ply})

            info = {"san": a["played_san"],
                    "by": "black" if pipe["board"].turn else "white",
                    "severity": ev["severity_label"], "types": ev["types"],
                    "delta_cp": a["delta_cp"], "ply": ply,
                    "facts": pipe.get("facts"),
                    "intensity": pipe.get("register_intensity", 0.2),
                    "route": pipe.get("route", "light")}
            infos.append(info)
            if key is None or _RANK[info["severity"]] > _RANK[key["severity"]] \
                    or (_RANK[info["severity"]] == _RANK[key["severity"]]
                        and abs(info["delta_cp"]) >= abs(key["delta_cp"])):
                key = info

            # batch moves show as muted lines; solo silent moves too
            if not solo or not pipe["commentary"]:
                _emit({"severity": info["severity"], "route": info["route"],
                       "speaker": None, "text": info["san"],
                       "audio": None, "flush": False, "ply": ply})

        def _announce_closing():
            """Game just ended (checkmate/stalemate/draw) — sign off with a
            short whole-game retrospective, independent of whether the final
            move got its own commentary."""
            if not pipe["board"].is_game_over():
                return
            closing = generate_closing(pipe, opts["persona"])
            if not closing:
                return
            ctx = _feedback_ctx(opts["persona"], 0.7, None)
            for i, (turn, audio, fmt) in enumerate(_speak_stream(closing, 0.7, opts)):
                if gid != GAME["id"]:
                    return
                _emit({"severity": "critical", "route": "closing",
                       "speaker": turn["speaker"], "text": turn["text"], "ctx": ctx,
                       "audio": audio, "audio_format": fmt, "flush": i == 0,
                       "ply": len(pipe["move_history"])})
                chat_log.append(turn)
            del chat_log[:-16]

        if not infos:
            _announce_closing()
            return

        last_ply = infos[-1]["ply"]

        # ── produce speech ──
        if solo:
            commentary = pipe["commentary"]
            label = infos[0]["severity"]
            if not commentary:
                _announce_closing()
                return                        # routine: silence, chatter covers it
        else:
            # recap only worth doing when something non-routine happened
            if _RANK[key["severity"]] == 0 or not key["facts"]:
                _announce_closing()
                return
            digest = []
            for i in infos:
                d = {k: v for k, v in i.items()
                     if k in ("san", "by", "severity", "types", "delta_cp")}
                d["san_spoken"] = (i.get("facts") or {}).get("move", {}).get("san_spoken") or d.get("san")
                digest.append(d)
            # always the short form now — "2-4" measured ~3.6s vs "2-3"'s ~1.8s,
            # and a caught-up queue doesn't make the wait for text+audio to
            # appear together feel any shorter, so there's no reason to use
            # the slower one just because the backlog happens to be clear
            commentary = generate_recap(key["facts"], digest, "2-3")
            label = key["severity"]
            if not commentary:
                _announce_closing()
                return

        chatting = False
        intensity = key["intensity"] if not solo else pipe.get("register_intensity", 0.2)
        facts_for_ctx = (pipe.get("facts") if solo else key.get("facts")) or {}
        ctx = _feedback_ctx(opts["persona"], intensity, facts_for_ctx,
                            (infos[0] if solo else key)["san"])
        flush = label == "critical"
        for i, (turn, audio, fmt) in enumerate(_speak_stream(commentary, intensity, opts)):
            if gid != GAME["id"]:
                return
            _emit({"severity": label, "route": "recap" if not solo else infos[0]["route"],
                   "speaker": turn["speaker"], "text": turn["text"], "ctx": ctx,
                   "audio": audio, "audio_format": fmt, "flush": flush and i == 0, "ply": last_ply})
            chat_log.append(turn)             # chat continues from what was said
        del chat_log[:-16]
        _announce_closing()

    while True:
        kind, gid, payload = _jobs.get()
        if gid != GAME["id"]:
            continue                          # stale job from a previous game
        try:
            handle(kind, gid, payload)
        except Exception:
            import logging
            logging.exception("worker job %s failed", kind)  # never kill the thread
        # signal the client that this move/opening is fully generated + emitted,
        # so a client-paced replay knows it can advance once the audio plays out
        if kind in ("move", "opening") and gid == GAME["id"]:
            publish({"type": "step_done", "kind": kind, "ply": len(GAME["sans"])})


threading.Thread(target=_worker, daemon=True).start()


def _warmup():
    """First call to a fresh API connection (TLS handshake, connection pool,
    provider-side cold start) tends to cost ~1s extra — fire one throwaway
    call to each provider at boot so that cost lands here, not on the first
    real move of the first game."""
    try:
        if config.USE_LLM:
            from langchain_openai import ChatOpenAI
            ChatOpenAI(model=config.LIGHT_MODEL, max_tokens=5).invoke("hi")
    except Exception:
        pass
    try:
        if config.FISHAUDIO_API_KEY:
            tts.speak("嗨", "play_by_play", 0.2, "calm")
    except Exception:
        pass


threading.Thread(target=_warmup, daemon=True).start()


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


class FeedbackReq(BaseModel):
    verdict: str                 # "good" -> keep, "bad" -> remove/don't collect
    persona: str
    speaker: str
    text: str
    intensity: float = 0.2
    tags: list[str] = []
    san: str | None = None


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
        GAME["replay"] = None                 # leave any prior replay mode
        gid = GAME["id"]
    _jobs.put(("reset", gid, None))
    publish({"type": "reset"})
    _jobs.put(("opening", gid, None))         # short welcome before the first move
    # engine plays first when the human took black
    if req.vs_engine and not req.human_is_white:
        _engine_move(gid)
    with _lock:
        return _snapshot()


# ── Replay mode: auto-play a preset real game and let the booth commentate ────
# The client drives the pace — it calls /api/replay/next once the current move's
# commentary has finished playing — so the booth never falls behind, moves are
# never batched/skipped, and there are no long silences from a too-fast timer.
GAMES_DIR = Path(__file__).parent / "games"


class ReplayReq(BaseModel):
    pgn: str                 # a filename inside games/ (not a path)
    persona: str = "calm"
    tts_on: bool = True


@app.get("/api/games")
def games():
    """List the preset real games available to replay (from games/*.pgn)."""
    out = []
    for p in sorted(GAMES_DIR.glob("*.pgn")):
        try:
            with open(p) as f:
                h = chess.pgn.read_headers(f)
        except Exception:
            h = None
        g = dict(h) if h else {}
        out.append({"file": p.name,
                    "white": g.get("White") or "?", "black": g.get("Black") or "?",
                    "event": g.get("Event") or "", "date": g.get("Date") or ""})
    return {"games": out}


@app.post("/api/replay")
def replay(req: ReplayReq):
    """Load a preset game into replay mode. The board doesn't advance yet — the
    client fires /api/replay/next (first move after the opening remark plays,
    then one per move once its commentary finishes)."""
    path = GAMES_DIR / Path(req.pgn).name     # basename only — no path traversal
    if not path.exists():
        return {"error": "game not found"}
    try:
        moves = moves_from_pgn(str(path))
    except Exception:
        return {"error": "could not read pgn"}
    if not moves:
        return {"error": "empty game"}
    with _lock:
        GAME["id"] += 1
        GAME["board"] = chess.Board()
        GAME["sans"] = []
        GAME["feed"] = []
        GAME["opts"] = {"vs_engine": False, "human_is_white": True, "depth": 8,
                        "persona": req.persona, "tts_on": req.tts_on, "mode": "replay"}
        GAME["replay"] = {"moves": moves, "idx": 0}
        gid = GAME["id"]
    _jobs.put(("reset", gid, None))
    publish({"type": "reset"})
    _jobs.put(("opening", gid, None))         # client fires the first move after this plays
    with _lock:
        return _snapshot()


@app.post("/api/replay/next")
def replay_next():
    """Advance the replay by one move — client-paced, called once the current
    move's commentary has finished playing. Returns {"done": bool}."""
    with _lock:
        gid = GAME["id"]
        rp = GAME.get("replay")
        if not rp or rp["idx"] >= len(rp["moves"]):
            return {"done": True}
        mv = rp["moves"][rp["idx"]]
        rp["idx"] += 1
        done = rp["idx"] >= len(rp["moves"])
    _push_move(mv, gid)
    return {"done": done}


@app.post("/api/opts")
def set_opts(req: OptsReq):
    with _lock:
        for k, v in req.model_dump().items():
            if v is not None:
                GAME["opts"][k] = v
        return {"opts": GAME["opts"], "llm_available": config.USE_LLM}


def _push_move(mv: chess.Move, gid: int):
    """Advance the live board and queue commentary. Caller holds no lock.
    Also broadcasts a board update over SSE so viewers who aren't the one
    making the move (replay mode, or the opponent's reply) see the piece move
    live without polling."""
    with _lock:
        board: chess.Board = GAME["board"]
        GAME["sans"].append(board.san(mv))
        board.push(mv)
        snap = {"fen": board.fen(), "sans": list(GAME["sans"]),
                "last_move": mv.uci(), "ply": len(GAME["sans"]),
                "game_over": board.is_game_over()}
    _jobs.put(("move", gid, mv.uci()))
    publish({"type": "board", **snap})


MIN_THINK_S, MAX_THINK_S = 2.0, 3.0   # purely cosmetic pacing — doesn't touch depth/strength


def _engine_move(gid: int) -> float:
    """Pick and push the engine's reply IMMEDIATELY — the worker starts
    analysing/speaking about it right away. Returns elapsed compute time so
    the caller can pad the *response* (visual reveal), not the analysis."""
    with _lock:
        board = GAME["board"].copy()
        depth = GAME["opts"]["depth"]
    t0 = time.monotonic()
    mv = _opp_engine.best_move(board, depth)
    elapsed = time.monotonic() - t0
    if mv:
        _push_move(mv, gid)          # queued for the worker now, not after any pad
    return elapsed


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
        elapsed = _engine_move(gid)
        # pad only the response, i.e. when the move becomes visible — analysis/
        # commentary for it was already queued inside _engine_move() and is
        # running concurrently on the worker thread during this sleep
        target = random.uniform(MIN_THINK_S, MAX_THINK_S)
        remaining = target - elapsed
        if remaining > 0:
            time.sleep(remaining)

    with _lock:
        return _snapshot()


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    """Record a 👍/👎 on one booth line into the user's style profile.
    good -> keep as an approved few-shot; bad -> remove/don't collect."""
    from personas import feedback_store
    count = feedback_store.record(req.verdict, req.persona, req.speaker, req.text,
                                  req.intensity, req.tags, req.san)
    return {"ok": count is not None, "count": count}


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
