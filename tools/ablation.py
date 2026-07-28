"""M4 — three-arm memory ablation harness (spec §"三層消融").

Same game, same engine analysis, same analyst model; only the memory fed into
FactsPacket.memory differs per arm:

    baseline  no memory        {callbacks: [], theory: []}
    rag       theory only      {callbacks: [], theory: <ChromaDB top-k>}
    full      layered memory   {callbacks: <last key moments>, theory: <top-k>}

For every deep-routed move each arm generates analyst commentary, then we score:
  * auto metrics  — call-back references (earlier move numbers / recall phrases),
                    theory-concept mentions split into grounded (concept was in
                    the retrieved snippets) vs ungrounded (LLM name-dropping).
  * LLM judge     — blind, shuffled A/B/C scoring on insight / game-context /
                    grounding, 1-5 each.

Usage:
    python tools/ablation.py games/demo.pgn [--no-judge]

Writes eval_out/ablation_<game>.md and prints the aggregate table.
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import config
from engine.stockfish_client import StockfishClient
from graph.nodes.perception import perception_node
from graph.nodes.event_detector import event_detector_node
from graph.nodes.director import director_node
from graph.nodes.memory import retrieve_memory_node, update_memory_node
from graph.nodes.booth import build_facts, _generate
from pipeline import moves_from_pgn

ARMS = ["baseline", "rag", "full"]

# Distinctive concept names used by the theory corpus — the measurable trace
# that a commentary actually drew on retrieved theory (vs name-dropping).
CONCEPT_TERMS = [
    "炸雞肝", "義大利開局", "西班牙開局", "西西里", "法蘭西", "卡羅-卡恩",
    "后翼棄兵", "倫敦系統", "超現代", "捉雙", "牽制", "串擊", "閃擊",
    "希臘禮物", "底線", "引離", "超載", "中間步", "孤兵", "疊兵", "落後兵",
    "開放線", "第七橫排", "雙象", "先手", "棄換", "預防性", "通路兵",
    "對王", "zugzwang",
]

CALLBACK_PATTERNS = [r"第\s*\d+\s*手", r"回想", r"還記得", r"稍早", r"先前", r"之前的"]


def _auto_metrics(text: str, theory: list, move_number: int) -> dict:
    callbacks = 0
    for pat in CALLBACK_PATTERNS:
        for m in re.finditer(pat, text):
            # a move-number reference only counts if it's not this very move
            if pat.startswith("第") and re.search(rf"第\s*{move_number}\s*手", m.group()):
                continue
            callbacks += 1
    theory_blob = " ".join(theory)
    grounded, ungrounded = 0, 0
    for term in CONCEPT_TERMS:
        if term in text:
            if term in theory_blob:
                grounded += 1
            else:
                ungrounded += 1
    return {"callbacks": callbacks, "theory_grounded": grounded,
            "theory_ungrounded": ungrounded}


def _judge(facts_summary: str, said_so_far: list, texts: dict) -> dict:
    """Blind LLM judge: shuffled labels, returns {arm: {insight, context, grounding}}."""
    from langchain_openai import ChatOpenAI
    order = list(ARMS)
    random.shuffle(order)
    labels = dict(zip("ABC", order))
    blocks = "\n\n".join(f"【解說 {lab}】{texts[arm]}" for lab, arm in labels.items())
    prompt = (
        "你是西洋棋播報品質評審。以下三段解說出自同一步棋、同一份引擎事實，"
        "請針對每段解說分別評分（1-5 整數）：\n"
        "- insight：洞察深度（是否解釋了『為什麼』，而非複述數字）\n"
        "- context：整局脈絡（是否與先前關鍵時刻連貫、有回扣）\n"
        "- grounding：接地性（內容是否都能由事實與先前時刻支持，不腦補）\n\n"
        f"這步棋的事實：{facts_summary}\n"
        f"本局先前關鍵時刻：{said_so_far or '（無）'}\n\n"
        f"{blocks}\n\n"
        '只輸出 JSON，格式：{"A": {"insight": n, "context": n, "grounding": n}, "B": {...}, "C": {...}}'
    )
    llm = ChatOpenAI(model=config.DEEP_MODEL, max_tokens=200,
                     model_kwargs={"response_format": {"type": "json_object"}})
    scores = json.loads(llm.invoke(prompt).content)
    return {arm: scores[lab] for lab, arm in labels.items()}


def run_ablation(pgn_path: str, use_judge: bool = True) -> list:
    engine = StockfishClient()
    if engine.mock:
        print("警告：mock 引擎——消融數據沒有意義。請設定 STOCKFISH_PATH。")
    state = {"board": chess.Board(), "move_history": [], "said_so_far": []}
    records = []
    try:
        for mv in moves_from_pgn(pgn_path):
            state["last_move"] = mv
            state = perception_node(state, engine)
            state = event_detector_node(state)
            state = director_node(state)

            if state["route"] == "deep":
                state = retrieve_memory_node(state)
                mem = state["retrieved_memory"]
                arm_memory = {
                    "baseline": {"callbacks": [], "theory": []},
                    "rag": {"callbacks": [], "theory": mem["theory"]},
                    "full": mem,
                }
                texts, metrics = {}, {}
                n = state["board"].fullmove_number
                for arm in ARMS:
                    state["retrieved_memory"] = arm_memory[arm]
                    facts = build_facts(state, "calm")
                    texts[arm] = _generate(facts, "analyst")
                    metrics[arm] = _auto_metrics(
                        texts[arm], arm_memory[arm]["theory"], n)
                state["retrieved_memory"] = mem  # full memory is the live arm

                a, ev = state["analysis"], state["event"]
                facts_summary = (f"第 {n} 手 {a['played_san']}，事件 {ev['types']}"
                                 f"{ev['motifs']}，評估 {a['score_cp']/100:+.1f}"
                                 f"（Δ {a['delta_cp']/100:+.1f}）")
                rec = {"move_number": n, "san": a["played_san"],
                       "severity": ev["severity_label"],
                       "facts_summary": facts_summary,
                       "theory": mem["theory"], "callbacks_fed": mem["callbacks"],
                       "texts": texts, "metrics": metrics}
                if use_judge:
                    try:
                        rec["judge"] = _judge(facts_summary,
                                              list(state["said_so_far"]), texts)
                    except Exception as e:
                        rec["judge_error"] = str(e)
                records.append(rec)
                print(f"  deep #{n} {a['played_san']} ✓")

            state["facts"] = build_facts(state, "calm")
            state = update_memory_node(state)
            state["move_history"].append(mv)
    finally:
        engine.close()
    return records


def write_report(pgn_path: str, records: list) -> str:
    name = os.path.splitext(os.path.basename(pgn_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "eval_out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ablation_{name}.md")

    agg = {arm: {"callbacks": 0, "theory_grounded": 0, "theory_ungrounded": 0,
                 "insight": [], "context": [], "grounding": []} for arm in ARMS}
    lines = [f"# 記憶消融報告 — {name}", "",
             f"深度路徑步數：{len(records)}；三組設定：baseline（無記憶）/ "
             f"rag（只棋理）/ full（棋理＋回扣）。同一引擎分析、同一模型、盲評打分。", ""]
    for rec in records:
        lines += [f"## 第 {rec['move_number']} 手 {rec['san']} [{rec['severity']}]", "",
                  f"- 事實：{rec['facts_summary']}",
                  f"- 檢索棋理：{'；'.join(t[:40] + '…' for t in rec['theory']) or '（無）'}",
                  f"- 回扣素材：{rec['callbacks_fed'] or '（無）'}", ""]
        for arm in ARMS:
            m = rec["metrics"][arm]
            j = rec.get("judge", {}).get(arm)
            jtxt = (f"｜盲評 insight {j['insight']} / context {j['context']} / "
                    f"grounding {j['grounding']}") if j else ""
            lines += [f"**{arm}**（回扣 {m['callbacks']}、棋理引用 有據 "
                      f"{m['theory_grounded']} / 無據 {m['theory_ungrounded']}{jtxt}）", "",
                      f"> {rec['texts'][arm]}", ""]
            for k in ("callbacks", "theory_grounded", "theory_ungrounded"):
                agg[arm][k] += m[k]
            if j:
                for k in ("insight", "context", "grounding"):
                    agg[arm][k].append(j[k])

    lines += ["## 總表", "",
              "| 設定 | 回扣次數 | 棋理引用（有據） | 棋理引用（無據） | 盲評 insight | 盲評 context | 盲評 grounding |",
              "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        g = agg[arm]
        avg = lambda xs: f"{sum(xs)/len(xs):.2f}" if xs else "—"
        lines.append(f"| {arm} | {g['callbacks']} | {g['theory_grounded']} | "
                     f"{g['theory_ungrounded']} | {avg(g['insight'])} | "
                     f"{avg(g['context'])} | {avg(g['grounding'])} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-6:]))
    return path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_judge = "--no-judge" not in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    for pgn in args:
        print(f"=== {pgn} ===")
        recs = run_ablation(pgn, use_judge)
        print("報告：", write_report(pgn, recs))
