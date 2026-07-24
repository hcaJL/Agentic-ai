"""Commentary booth. Builds the FactsPacket and generates grounded commentary.

Runs OFFLINE out of the box (templated grounded text). Set OPENAI_API_KEY
to switch to real LLM commentary.
"""
import chess
import config
from personas.personas import persona_prompt


# ---------- FactsPacket assembly ----------
def build_facts(state: dict, persona: str) -> dict:
    board: chess.Board = state["board"]
    move: chess.Move = state["last_move"]
    a = state["analysis"]
    ev = state["event"]
    before = board.copy(); before.pop()
    piece = before.piece_at(move.from_square)
    if before.is_en_passant(move):
        captured = "pawn"
    else:
        victim = before.piece_at(move.to_square)
        captured = chess.piece_name(victim.piece_type) if victim else None

    return {
        "move": {
            "san": before.san(move),
            "uci": move.uci(),
            "played_by": "black" if board.turn else "white",  # mover = not side_to_move
            "piece": chess.piece_name(piece.piece_type) if piece else "",
            "is_capture": before.is_capture(move),
            "captured": captured,
            "is_check": board.is_check(),
            "is_mate": board.is_checkmate(),
            "is_promotion": move.promotion is not None,
        },
        "board": {
            "fen": board.fen(),
            "move_number": board.fullmove_number,
            "side_to_move": "white" if board.turn else "black",
            "material_balance": state["_material"],
            "phase": ev["phase"],
        },
        # score_cp/delta_cp are sentinel ±10000 when mate_in is set (see
        # stockfish_client._to_cp) — that's not a real centipawn/material value,
        # so it's dropped here to stop the LLM narrating "lost 9000+ material"
        "eval": {
            "score_cp": a["score_cp"] if a["mate_in"] is None else None,
            "delta_cp": a["delta_cp"] if a["mate_in"] is None else None,
            "mate_in": a["mate_in"],
            "mate_note": (f"engine sees forced mate in {abs(a['mate_in'])} move(s) for "
                          f"{'white' if a['mate_in'] > 0 else 'black'} — describe this as "
                          "a forced-mate sequence, NEVER as a material/point count"
                          if a["mate_in"] is not None else None),
            "best_line_san": a["best_line_san"],
            "top_moves": a["top_moves"],
        },
        "event": {"types": ev["types"], "motifs": ev["motifs"],
                  "severity_label": ev["severity_label"]},
        "memory": state.get("retrieved_memory", {"callbacks": [], "theory": []}),
        "register": {"intensity": state.get("register_intensity", 0.2), "persona": persona},
    }


# ---------- generation ----------
def _generate(facts: dict, speaker: str) -> str:
    if config.USE_LLM:
        try:
            return _llm_generate(facts, speaker)
        except Exception as e:
            return f"[LLM error, fell back to template] {_template(facts, speaker)}"
    return _template(facts, speaker)


def _llm_generate(facts: dict, speaker: str) -> str:
    from langchain_openai import ChatOpenAI
    model = config.DEEP_MODEL if speaker == "analyst" else config.LIGHT_MODEL
    llm = ChatOpenAI(model=model, max_tokens=200)
    sys = persona_prompt(facts["register"]["persona"], facts["register"]["intensity"])
    if speaker == "analyst":
        role = ("你是戰略分析師，解釋『為什麼』。"
                "memory.theory 是檢索到的棋理，若與當前局面相關，自然地融入解說（不相關就忽略）。"
                "memory.callbacks 是本局先前的關鍵時刻，適合時回扣它們"
                "（例如「還記得第 N 手的…」），讓解說有整局脈絡。")
    else:
        role = "你是即時主播，描述剛發生的事。"
    msg = (f"{sys}\n{role}\n以下是唯一可用的事實，請用一兩句口語播報：\n{facts}")
    return llm.invoke(msg).content.strip()


def _qual_score(score_cp: float | None) -> str:
    """Qualitative read of the overall (white-POV) evaluation — viewers don't
    need the raw centipawn number, just who's better and by how much."""
    if score_cp is None:
        return "局勢不明朗"
    side = "白方" if score_cp >= 0 else "黑方"
    mag = abs(score_cp)
    if mag < config.DELTA_NOTABLE:
        return "雙方大致均勢"
    if mag < config.DELTA_CRITICAL:
        return f"{side}稍佔優勢"
    if mag < config.BLUNDER_DROP * 2:
        return f"{side}優勢明顯"
    return f"{side}大幅領先"


def _qual_delta(delta_cp: float | None) -> str:
    """Qualitative read of this move's swing (mover's own POV) — direction and
    magnitude only, never the raw number."""
    if delta_cp is None:
        return "局勢變化不大"
    direction = "改善" if delta_cp >= 0 else "轉差"
    mag = abs(delta_cp)
    if mag < config.DELTA_NOTABLE:
        return f"局勢略有{direction}"
    if mag < config.DELTA_CRITICAL:
        return f"局勢明顯{direction}"
    return f"局勢大幅{direction}"


def _template(facts: dict, speaker: str) -> str:
    """Offline grounded fallback — proves the grounding contract without an LLM."""
    m, e, ev = facts["move"], facts["eval"], facts["event"]
    bits = [f"{facts['board']['move_number']}. {m['san']}"]
    if ev["types"]:
        bits.append("（" + "、".join(ev["types"]) + "）")
    if speaker == "analyst":
        if e["best_line_san"]:
            bits.append("引擎建議續法：" + " ".join(e["best_line_san"][:4]))
        if e["mate_in"] is not None:
            bits.append(f"偵測到強制將死（{abs(e['mate_in'])} 步內）")
        else:
            bits.append(_qual_score(e["score_cp"]))
        if facts["memory"]["theory"]:
            bits.append("（理論：" + facts["memory"]["theory"][0] + "）")
    else:
        if m["is_check"]:
            bits.append("將軍！")
        if e["mate_in"] is not None:
            bits.append("已進入強制將死序列")
        else:
            bits.append(_qual_delta(e["delta_cp"]))
    return " ".join(bits)


# ---------- dialogue script (Phase B: two voices in conversation) ----------
def _dialogue_generate(facts: dict, n_turns: str = "3-5") -> list[dict]:
    """One LLM call → a short alternating dialogue script.
    Returns [{"speaker": ..., "text": ...}, ...]; raises on failure."""
    import json as _json
    import re as _re
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=config.DEEP_MODEL, max_tokens=500)
    persona = facts["register"]["persona"]
    intensity = facts["register"]["intensity"]
    sys = persona_prompt(persona, intensity)
    if intensity < 0.8:
        interject = ""
    elif persona == "calm":
        # same "this matters" cue as excited's interjection, but without an
        # exclamation or excited-style wording — otherwise calm gets the
        # excited persona's hook verbatim on every critical move
        interject = "第一句可以用簡短的提示語開場（例如「這裡值得停一下」「注意看這步」），語氣仍維持沉穩，不可用驚嘆詞；"
    else:
        interject = "第一句要用打斷式的驚嘆開場（例如「欸等等——」「哇這步！」），"
    msg = (
        f"{sys}\n"
        "你要寫一段「兩位棋賽播報員的即時對話」，像真人搭檔接話，不是各自獨白。\n"
        "角色：play_by_play（主播，描述發生什麼、拋出鉤子）、"
        "analyst（分析師，接話解釋為什麼、給續法）。\n"
        f"規則：共 {n_turns} 句，兩人交替；每句短（15-40字），口語、可加語助詞；"
        "後一句要接前一句的話尾，可以互相附和或反問；"
        f"{interject}"
        "不要重複同樣的資訊；"
        "memory.callbacks 有本局先前關鍵時刻，適合時回扣；"
        "memory.theory 有棋理，相關才用。\n"
        "只能使用以下事實，不可捏造評估或棋步：\n"
        f"{facts}\n"
        '輸出 JSON 陣列：[{"speaker":"play_by_play","text":"..."},'
        '{"speaker":"analyst","text":"..."}]，不要其他文字。'
    )
    raw = llm.invoke(msg).content.strip()
    m = _re.search(r"\[.*\]", raw, _re.S)
    turns = _json.loads(m.group(0) if m else raw)
    out = []
    for t in turns:
        sp = t.get("speaker")
        tx = (t.get("text") or "").strip()
        if sp in ("play_by_play", "analyst") and tx:
            out.append({"speaker": sp, "text": tx})
    if not out:
        raise ValueError("empty dialogue")
    return out


def generate_recap(key_facts: dict, batch_moves: list[dict],
                   n_turns: str = "2-4") -> list[dict]:
    """Catch-up recap: the booth fell behind several moves — one compressed
    script in a「剛剛…」retrospective voice, focused on the most important
    move of the batch, mentioning the rest in passing.
    batch_moves: [{"san", "by", "severity", "types", "delta_cp"}, ...]
    Returns [] on failure (caller then stays silent for the batch)."""
    if not config.USE_LLM:
        return []
    import json as _json
    import re as _re
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=config.DEEP_MODEL, max_tokens=420)
    intensity = key_facts["register"]["intensity"]
    sys = persona_prompt(key_facts["register"]["persona"], intensity)
    key_san = key_facts["move"]["san"]
    msg = (
        f"{sys}\n"
        "你們是兩位棋賽播報員。剛才棋下得很快，你們來不及逐步講解，"
        "現在要用「回顧補講」的方式一次帶過剛剛的幾步：\n"
        f"剛剛依序發生了這些（最後一步是最新局面）：{batch_moves}\n"
        f"其中最關鍵的一步是 {key_san}，它的完整事實如下（只能引用這些，不可捏造）：\n"
        f"{key_facts}\n"
        f"規則：共 {n_turns} 句，兩人交替接話；用回顧口吻開場"
        "（例如「剛剛這幾步…」「趁現在補一下，剛才那步…」）；"
        f"把重點放在 {key_san}，其他步一句帶過或不提；每句短（15-45字）、口語。\n"
        '輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
    )
    try:
        raw = llm.invoke(msg).content.strip()
        m = _re.search(r"\[.*\]", raw, _re.S)
        turns = _json.loads(m.group(0) if m else raw)
        return [{"speaker": t["speaker"], "text": t["text"].strip()}
                for t in turns
                if t.get("speaker") in ("play_by_play", "analyst")
                and t.get("text", "").strip()]
    except Exception:
        return []


def generate_opening(opts: dict, persona: str) -> list[dict]:
    """Pre-game welcome: sets the scene before the first move is even made —
    persona's style, the matchup, and who has which color. Unlike the other
    generate_* helpers this never returns [] — an opening remark should
    always play, so a templated welcome covers the no-LLM/failure case
    instead of falling silent at kickoff."""
    vs_engine = opts.get("vs_engine", True)
    matchup = f"人類 vs 引擎（深度 {opts.get('depth', 8)}）" if vs_engine else "雙人對弈"
    human_side = ("白方" if opts.get("human_is_white", True) else "黑方") if vs_engine else None
    template = [
        {"speaker": "play_by_play",
         "text": f"歡迎回到棋訊直播間！本局是{matchup}"
                 + (f"，我們這邊執{human_side}" if human_side else "") + "。"},
        {"speaker": "analyst", "text": "準備開始了，一起關注這盤棋的每一步。"},
    ]
    if not config.USE_LLM:
        return template

    import json as _json
    import re as _re
    from langchain_openai import ChatOpenAI
    setup = {"matchup": matchup, "human_side": human_side}
    llm = ChatOpenAI(model=config.DEEP_MODEL, max_tokens=260)
    sys = persona_prompt(persona, 0.4)
    msg = (
        f"{sys}\n"
        "對局即將開始，你們兩位播報員要做個簡短的「開場白」歡迎觀眾、介紹這場對局"
        "（這是第一段話，棋還沒下，不可提到任何棋步、評估或局勢）。\n"
        f"本局設定：{setup}\n"
        "規則：共 2-3 句，兩人交替；用開場歡迎的口吻開場"
        "（例如「歡迎回到棋訊直播間」「今天這一局…」）；"
        "只能引用上面的設定，不可捏造棋步或局勢；每句短（15-40字）、口語。\n"
        '輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
    )
    try:
        raw = llm.invoke(msg).content.strip()
        m = _re.search(r"\[.*\]", raw, _re.S)
        turns = _json.loads(m.group(0) if m else raw)
        out = [{"speaker": t["speaker"], "text": t["text"].strip()}
               for t in turns
               if t.get("speaker") in ("play_by_play", "analyst")
               and t.get("text", "").strip()]
        return out or template
    except Exception:
        return template


def generate_closing(state: dict, persona: str) -> list[dict]:
    """Post-game sign-off: the booth wraps up once the game actually ends
    (checkmate/stalemate/draw) with a short retrospective over the whole
    game, not just the final move. Draws on said_so_far — the running list
    of every non-routine moment recorded across the game.
    Returns [] on failure/no LLM (caller then stays silent)."""
    if not config.USE_LLM:
        return []
    import json as _json
    import re as _re
    from langchain_openai import ChatOpenAI
    board: chess.Board = state["board"]
    result = board.result()
    if board.is_checkmate():
        ending = "將殺"
        winner = "黑方" if board.turn else "白方"   # side to move is the one just mated
    elif board.is_stalemate():
        ending, winner = "逼和", None
    else:
        ending, winner = "和棋", None
    highlights = state.get("said_so_far", [])[-6:]
    intensity = 0.9 if board.is_checkmate() else 0.4
    llm = ChatOpenAI(model=config.DEEP_MODEL, max_tokens=320)
    sys = persona_prompt(persona, intensity)
    msg = (
        f"{sys}\n"
        "對局剛剛結束，你們兩位播報員要做個簡短的「賽後總結」為這場對局收尾"
        "（這是最後一段話，不是在播下一步）。\n"
        f"結果：{result}（{ending}"
        + (f"，{winner}獲勝" if winner else "") + f"）\n"
        f"整場比賽依序的關鍵時刻：{highlights}\n"
        "規則：共 2-3 句，兩人交替；用總結收尾的口吻開場"
        "（例如「這場對局…」「回顧整盤棋…」）；可以點名 1-2 個真正關鍵的時刻；"
        "只能引用上面的事實，不可捏造細節；每句短（15-45字）、口語。\n"
        '輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
    )
    try:
        raw = llm.invoke(msg).content.strip()
        m = _re.search(r"\[.*\]", raw, _re.S)
        turns = _json.loads(m.group(0) if m else raw)
        return [{"speaker": t["speaker"], "text": t["text"].strip()}
                for t in turns
                if t.get("speaker") in ("play_by_play", "analyst")
                and t.get("text", "").strip()]
    except Exception:
        return []


def generate_filler(state: dict, persona: str,
                    chat_history: list[dict] | None = None) -> list[dict]:
    """Phase C: dead-air chatter — an ongoing conversation, not one-shot rounds.
    chat_history carries what was already said so the pair keeps the thread
    going instead of repeating themselves. Returns [] when no LLM / failure."""
    if not config.USE_LLM:
        return []
    import json as _json
    import re as _re
    from langchain_openai import ChatOpenAI
    board: chess.Board = state["board"]
    a = state.get("analysis") or {}
    mem = state.get("retrieved_memory", {}) or {}
    b = chess.Board()
    sans = []
    for mv in state.get("move_history", []):
        try:
            sans.append(b.san(mv)); b.push(mv)
        except Exception:
            break
    material = {
        "fen": board.fen(),
        "recent_moves": sans[-10:],
        "evaluation": _qual_score(a.get("score_cp")),
        "likely_next": [t.get("san") for t in (a.get("top_moves") or [])[:3]],
        "callbacks": (mem.get("callbacks") or state.get("said_so_far", []))[-3:],
    }
    history = ""
    if chat_history:
        lines = "\n".join(f"{t['speaker']}: {t['text']}" for t in chat_history[-8:])
        history = (
            "你們剛才已經聊了這些（接著這段對話聊下去，"
            "不要重複已講過的觀點；可以深入同一話題，也可以自然換新角度）：\n"
            f"{lines}\n"
        )
    llm = ChatOpenAI(model=config.LIGHT_MODEL, max_tokens=260)
    sys = persona_prompt(persona, 0.25)
    msg = (
        f"{sys}\n"
        "棋局暫時沒有新動作，你們兩位播報員在轉播空檔自然閒聊，"
        "像平常聊天一樣有來有往。\n"
        f"{history}"
        "話題方向（挑還沒聊過的）：局面走向、回扣先前關鍵時刻、"
        "猜接下來的著法、子力擺位的觀察、與此局面相關的棋理或趣談。\n"
        "規則：2 句，兩人一來一往；每句短（15-40字）、語氣放鬆口語；"
        "只能引用以下事實，不可捏造：\n"
        f"{material}\n"
        '輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
    )
    try:
        raw = llm.invoke(msg).content.strip()
        m = _re.search(r"\[.*\]", raw, _re.S)
        turns = _json.loads(m.group(0) if m else raw)
        return [{"speaker": t["speaker"], "text": t["text"].strip()}
                for t in turns
                if t.get("speaker") in ("play_by_play", "analyst")
                and t.get("text", "").strip()][:3]
    except Exception:
        return []


# ---------- nodes ----------
def light_commentary_node(state: dict, persona: str) -> dict:
    facts = build_facts(state, persona)
    state["facts"] = facts
    # async worker sets this to catch up when the game has moved on
    # (explicit reset — with LangGraph a missing key keeps the old channel value)
    skip = state.get("skip_generation", False)
    state["skip_generation"] = False
    if skip:
        state["commentary"] = []
        return state
    # routine moves can be silent
    if facts["event"]["severity_label"] == "routine" and not facts["move"]["is_check"]:
        state["commentary"] = []
        return state
    state["commentary"] = [{"speaker": "play_by_play", "text": _generate(facts, "play_by_play")}]
    return state


def booth_node(state: dict, persona: str) -> dict:
    facts = build_facts(state, persona)
    state["facts"] = facts
    # in a batch (worker behind), the caller discards this and speaks via
    # generate_recap instead — so generating a full dialogue here per critical
    # move would be pure waste: one extra LLM round-trip (~2-5s) for EVERY
    # critical move in the batch, which is exactly what was making the booth
    # fall further behind on a tactically sharp sequence. Only pay for it
    # when this is a solo (real-time) move that will actually be spoken.
    skip = state.get("skip_generation", False)
    state["skip_generation"] = False
    if skip:
        state["commentary"] = []
        return state
    if config.USE_LLM:
        try:
            state["commentary"] = _dialogue_generate(
                facts, state.get("dialogue_turns") or "3-5")
            return state
        except Exception:
            pass  # fall back to the two-monologue form
    state["commentary"] = [
        {"speaker": "play_by_play", "text": _generate(facts, "play_by_play")},
        {"speaker": "analyst", "text": _generate(facts, "analyst")},
    ]
    return state
