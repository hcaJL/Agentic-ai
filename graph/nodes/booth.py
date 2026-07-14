"""Commentary booth. Builds the FactsPacket and generates grounded commentary.

Runs OFFLINE out of the box (templated grounded text). Set ANTHROPIC_API_KEY and
`pip install langchain-anthropic` to switch to real LLM commentary.
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
        "eval": {
            "score_cp": a["score_cp"],
            "delta_cp": a["delta_cp"],
            "mate_in": a["mate_in"],
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


def _template(facts: dict, speaker: str) -> str:
    """Offline grounded fallback — proves the grounding contract without an LLM."""
    m, e, ev = facts["move"], facts["eval"], facts["event"]
    bits = [f"{facts['board']['move_number']}. {m['san']}"]
    if ev["types"]:
        bits.append("（" + "、".join(ev["types"]) + "）")
    if speaker == "analyst":
        if e["best_line_san"]:
            bits.append("引擎建議續法：" + " ".join(e["best_line_san"][:4]))
        bits.append(f"評估 {e['score_cp']/100:+.1f}")
        if facts["memory"]["theory"]:
            bits.append("（理論：" + facts["memory"]["theory"][0] + "）")
    else:
        if m["is_check"]:
            bits.append("將軍！")
        bits.append(f"評估變化 {e['delta_cp']/100:+.1f}")
    return " ".join(bits)


# ---------- dialogue script (Phase B: two voices in conversation) ----------
def _dialogue_generate(facts: dict, n_turns: str = "3-5") -> list[dict]:
    """One LLM call → a short alternating dialogue script.
    Returns [{"speaker": ..., "text": ...}, ...]; raises on failure."""
    import json as _json
    import re as _re
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=config.DEEP_MODEL, max_tokens=500)
    sys = persona_prompt(facts["register"]["persona"], facts["register"]["intensity"])
    intensity = facts["register"]["intensity"]
    interject = ("第一句要用打斷式的驚嘆開場（例如「欸等等——」「哇這步！」），"
                 if intensity >= 0.8 else "")
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
        "eval_cp": a.get("score_cp"),
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
    # deep path = critical moment: always worth commenting, even when the
    # worker is behind — real commentators circle back to the big moves
    state["skip_generation"] = False
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
