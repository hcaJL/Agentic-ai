# Agentic 西洋棋即時主播 — 初版

會判斷何時該激動、何時該閉嘴、所有棋理都有引擎背書的多主播台。
完整設計見 `chess_broadcaster_spec.md`。

## 這個初版做到什麼（M0–M2）

- ✅ 盤面感知（python-chess）+ Stockfish 評估 + **落子前後評估差**
- ✅ 事件偵測 + 嚴重度（check / mate / capture / blunder / sacrifice / fork / 進殘局…）
- ✅ **Director 動態路由**（light / deep）+ 由嚴重度推導的情緒 register ← 核心
- ✅ 接地播報（FactsPacket，主播只能複述事實）
- ✅ routine 步**自動靜默**
- ✅ 三種 persona 風格皮
- ✅ Streamlit 介面（棋盤、評估、逐步播報）
- ✅ **離線可跑**：沒有 Stockfish / 沒有 API key 時，用 mock 引擎 + 接地模板代打

## 還沒做（接下來照 spec 補）

- ⬜ 真 LLM 播報（接 `langchain-anthropic`，見 `graph/nodes/booth.py` 的 `_llm_generate`）
- ⬜ ChromaDB 分層記憶第 2 層（棋理檢索，見 `graph/nodes/memory.py` 的 TODO）
- ⬜ LangGraph 化（`graph/build_graph.py` 已備好，跑 `pip install langgraph` 即可切換）
- ⬜ 三層消融 harness、TTS 語音、棋手風格檔

## 快速開始

```bash
pip install -r requirements.txt        # 初版其實只需要 python-chess + streamlit

# 1) 下載 Stockfish，設定路徑（不設就用 mock 引擎，仍可跑）
#    https://stockfishchess.org/download/
export STOCKFISH_PATH=/path/to/stockfish

# 2) 接真 LLM（不設就用接地模板）
export ANTHROPIC_API_KEY=sk-...

# 命令列跑一局
python demo_cli.py
python demo_cli.py games/demo.pgn excited

# 開介面
streamlit run app.py
```

## 結構

```
config.py                  # 門檻、模型、Stockfish 路徑（最常調的就是嚴重度門檻）
pipeline.py                # 初版的循序驅動（process_move / run_game）
demo_cli.py / app.py       # CLI 與 Streamlit 入口
engine/stockfish_client.py # UCI 封裝（含 mock fallback）
engine/chess_utils.py      # material、phase、fork、en-prise
graph/state.py             # 資料結構
graph/nodes/               # perception / event_detector / director / memory / booth
graph/build_graph.py       # LangGraph 版（M2+ 切換用）
personas/personas.py       # 三種風格 + 接地鐵則
```

## 最先該調的東西

`config.py` 的嚴重度門檻（`DELTA_CRITICAL` 等）。太鬆 → 每步都進 deep、變慢；太緊 → 冷場。
先跑 `demo_cli.py` 看判定合不合理，再微調。
