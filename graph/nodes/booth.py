"""Commentary booth. Builds the FactsPacket and generates grounded commentary.

Runs OFFLINE out of the box (templated grounded text). Set OPENAI_API_KEY
to switch to real LLM commentary.
"""
import chess
import config
from personas.personas import persona_prompt


# ---------- spoken move names (say「皇后到 c4」, never「Qc4」) ----------
_PIECE_ZH = {"king": "王", "queen": "皇后", "rook": "城堡",
             "bishop": "主教", "knight": "馬", "pawn": "兵"}


def _san_spoken(san: str, piece_en: str, to_name: str, is_capture: bool,
                promotion_en: str | None, is_check: bool, is_mate: bool) -> str:
    """Colloquial Chinese for a move: 皇后到 c4 / 馬吃 f7 / 王翼易位, plus a
    將軍/將死 suffix. Booth prompts prefer this over the raw SAN symbol."""
    if san.startswith("O-O-O"):
        s = "后翼易位（長易位）"
    elif san.startswith("O-O"):
        s = "王翼易位（短易位）"
    else:
        p = _PIECE_ZH.get(piece_en, piece_en or "")
        s = f"{p}{'吃' if is_capture else '到'}{to_name}"
        if promotion_en:
            s += f"，升變為{_PIECE_ZH.get(promotion_en, promotion_en)}"
    if is_mate:
        s += "，將死"
    elif is_check:
        s += "，將軍"
    return s


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

    san = before.san(move)
    piece_en = chess.piece_name(piece.piece_type) if piece else ""
    promotion_en = chess.piece_name(move.promotion) if move.promotion else None

    return {
        "move": {
            "san": san,
            "san_spoken": _san_spoken(san, piece_en, chess.square_name(move.to_square),
                                      before.is_capture(move), promotion_en,
                                      board.is_check(), board.is_checkmate()),
            "uci": move.uci(),
            "played_by": "black" if board.turn else "white",  # mover = not side_to_move
            "piece": piece_en,
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
def _make_llm(model: str, max_tokens: int, temperature: float):
    """ChatOpenAI with looser, repetition-penalised sampling (see config).
    The penalties are the real lever against the "every line has the same
    skeleton" feel — they discourage the model reusing words/phrasing."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=config.FREQUENCY_PENALTY,
        presence_penalty=config.PRESENCE_PENALTY,
    )


def _event_tags(facts: dict) -> list[str]:
    """Flatten event types + motifs + severity into few-shot selection tags."""
    ev = facts.get("event", {})
    return list(ev.get("types", [])) + list(ev.get("motifs", [])) + [ev.get("severity_label", "")]


def _generate(facts: dict, speaker: str) -> str:
    if config.USE_LLM:
        try:
            return _llm_generate(facts, speaker)
        except Exception as e:
            return f"[LLM error, fell back to template] {_template(facts, speaker)}"
    return _template(facts, speaker)


def _llm_generate(facts: dict, speaker: str) -> str:
    sys = persona_prompt(facts["register"]["persona"], facts["register"]["intensity"],
                         _event_tags(facts))
    if speaker == "analyst":
        llm = _make_llm(config.DEEP_MODEL, max_tokens=200, temperature=config.LIGHT_TEMPERATURE)
        role = ("你是負責解釋『為什麼』的分析師搭檔。"
                "memory.theory 若跟眼前局面對得上就自然帶進來，對不上就別提；"
                "memory.callbacks 是本局先前的關鍵時刻，聊到相關的地方可以回扣一下，"
                "讓解說有整局的來龍去脈。")
        guide = "用一兩句口語講，講到的棋步要照下面的事實："
        f = facts
    else:
        # light path (single quick call): keep it to ONE short line — just say what
        # happened, don't analyse or give continuations (that's the analyst's job,
        # on the deep path). Drop best_line/top_moves from facts so it can't spill
        # engine continuations into a "light" blurb.
        llm = _make_llm(config.LIGHT_MODEL, max_tokens=90, temperature=config.LIGHT_TEMPERATURE)
        role = ("你是即時主播，只用一句話簡短講出剛剛這步發生了什麼、帶點臨場反應就好。"
                "不要分析為什麼、不要給後續著法或續法建議，那是分析師的事。")
        guide = "只用一句話、簡短口語，照下面的事實講剛發生的棋步（不要念評估、不要給續法）："
        f = {**facts, "eval": {k: v for k, v in facts["eval"].items()
                               if k not in ("best_line_san", "top_moves")}}
    msg = f"{sys}\n{role}\n{guide}\n{f}"
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
    llm = _make_llm(config.DEEP_MODEL, max_tokens=500,
                    temperature=config.DIALOGUE_TEMPERATURE)
    persona = facts["register"]["persona"]
    intensity = facts["register"]["intensity"]
    sys = persona_prompt(persona, intensity, _event_tags(facts))
    # critical-move opening cue is persona-aware: calm keeps its composure (no
    # exclamation), everyone else gets the barge-in hype — otherwise calm gets
    # the excited persona's hook verbatim on exactly the moves that matter most
    if intensity < 0.8:
        hype = ""
    elif persona == "calm":
        hype = "第一句可以用簡短的提示語開場（例如「這裡值得停一下」「注意看這步」），語氣仍維持沉穩，不可用驚嘆詞。\n"
    else:
        hype = "這步很勁爆，主播可以用被畫面嚇到的反應開場，情緒衝出來。\n"
    msg = (
        f"{sys}\n"
        "接下來是你和搭檔的即時對話——像兩個人真的在轉播台上你一句我一句、"
        "會互相接話、附和、反問，不是各講各的獨白。\n"
        "play_by_play 是主播，先講發生了什麼、把話拋出去；"
        "analyst 是分析師，接著講為什麼、之後可能怎麼走。\n"
        f"{hype}"
        f"大概 {n_turns} 句上下，長短跟著情緒走、該短就短；別把同一件事講兩遍。\n"
        "memory.callbacks 是本局先前的關鍵時刻，聊到相關處可以回扣；"
        "memory.theory 有棋理，對得上再用。\n"
        "講到的棋步只能照下面的事實，別自己編：\n"
        f"{facts}\n"
        '只輸出 JSON 陣列：[{"speaker":"play_by_play","text":"..."},'
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
    intensity = key_facts["register"]["intensity"]
    llm = _make_llm(config.DEEP_MODEL, max_tokens=420,
                    temperature=config.DIALOGUE_TEMPERATURE)
    sys = persona_prompt(key_facts["register"]["persona"], intensity, _event_tags(key_facts))
    key_san = key_facts["move"].get("san_spoken") or key_facts["move"]["san"]
    msg = (
        f"{sys}\n"
        "剛才棋下得太快，你和搭檔沒能一步步講，現在趁空檔用回顧的口氣一次補講剛剛那幾步。\n"
        f"剛剛依序發生了這些（最後一步是最新局面）：{batch_moves}\n"
        f"其中最關鍵的是「{key_san}」，它的完整事實如下（講到就照這個，別編）：\n"
        f"{key_facts}\n"
        f"用回頭補講的口氣開場，大概 {n_turns} 句、兩人接話；"
        f"重點擺在「{key_san}」，其他步一句帶過或不提；口語、該短就短。\n"
        '只輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
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
    llm = _make_llm(config.DEEP_MODEL, max_tokens=320, temperature=config.DIALOGUE_TEMPERATURE)
    sys = persona_prompt(persona, intensity)
    msg = (
        f"{sys}\n"
        "對局剛剛結束了，你和搭檔用一小段話為這場棋收尾——這是今天最後一段話，"
        "不是在播下一步。\n"
        f"結果：{result}（{ending}"
        + (f"，{winner}獲勝" if winner else "") + f"）\n"
        f"整場比賽依序的關鍵時刻：{highlights}\n"
        "用收尾、回望整盤的口氣，兩人接話大概 2-3 句；"
        "可以點名一兩個真正關鍵的時刻；講到的細節照上面的事實、別編；口語、簡短。\n"
        '只輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
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
    llm = _make_llm(config.LIGHT_MODEL, max_tokens=260, temperature=config.DIALOGUE_TEMPERATURE)
    sys = persona_prompt(persona, 0.25)
    msg = (
        f"{sys}\n"
        "棋局暫時沒有新動作，你和搭檔趁空檔隨口聊兩句，像平常聊天一樣有來有往。\n"
        f"{history}"
        "可以聊的方向（挑還沒聊過的）：局面走向、回扣先前關鍵時刻、"
        "猜接下來會怎麼走、子力擺位的觀察、跟這局面有關的棋理或趣談。\n"
        "兩人一來一往，放鬆口語；講到的棋步照下面的事實、別編：\n"
        f"{material}\n"
        '只輸出 JSON 陣列：[{"speaker":"play_by_play"或"analyst","text":"..."}]，不要其他文字。'
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
