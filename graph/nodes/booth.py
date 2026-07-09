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


# ---------- nodes ----------
def light_commentary_node(state: dict, persona: str) -> dict:
    facts = build_facts(state, persona)
    state["facts"] = facts
    # routine moves can be silent
    if facts["event"]["severity_label"] == "routine" and not facts["move"]["is_check"]:
        state["commentary"] = []
        return state
    state["commentary"] = [{"speaker": "play_by_play", "text": _generate(facts, "play_by_play")}]
    return state


def booth_node(state: dict, persona: str) -> dict:
    facts = build_facts(state, persona)
    state["facts"] = facts
    state["commentary"] = [
        {"speaker": "play_by_play", "text": _generate(facts, "play_by_play")},
        {"speaker": "analyst", "text": _generate(facts, "analyst")},
    ]
    return state
