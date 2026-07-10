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

- ✅ **真 LLM 播報**（OpenAI gpt-4o / gpt-4o-mini，見 `graph/nodes/booth.py`）
- ✅ Stockfish 真引擎（設好 `.env` 的 `STOCKFISH_PATH` 即自動切換；門檻已用真實棋局校過，驗證工具見 `tools/severity_report.py`）
- ✅ ChromaDB 分層記憶第 2 層（棋理檢索，語料在 `memory/theory_seed.py`；沒裝 chromadb 時自動退回關鍵字比對）
- ✅ 三層消融 harness（`tools/ablation.py`，baseline / rag / full 對照＋盲評，報告在 `eval_out/`）
- ✅ LangGraph 化（裝了 `langgraph` 自動走圖，沒裝退回 sequential；Director 路由是圖上真正的 conditional edge）
- ⬜ TTS 語音、棋手風格檔

## 架構（LangGraph）

```mermaid
graph TD
    S([每一手]) --> perception --> detect_event --> director
    director -. "light（routine/notable）" .-> light
    director -. "deep（critical）" .-> retrieve_memory --> booth
    light --> update_memory
    booth --> update_memory
    update_memory --> E([下一手])
```

`director` 的分流是 `add_conditional_edges`——這是整個系統的 agentic 核心：例行步走便宜的 `light`（多半靜默），關鍵步才觸發 `retrieve_memory`（ChromaDB 棋理 + 本局回扣）與雙主播 `booth`。

## 快速開始

### 1) 安裝依賴

```bash
pip install -r requirements.txt
# 或是只裝必要的：
pip install python-chess streamlit langchain-openai langchain python-dotenv
```

### 2) 建立 .env 設定檔

在專案根目錄建一個 `.env`（此檔不會被 git 追蹤，請勿 commit）：

```
OPENAI_API_KEY=sk-proj-你的key填這裡

# 下載 Stockfish 後把路徑填進來（不填就用 mock 引擎）
# STOCKFISH_PATH=/path/to/stockfish
```

> **注意**：API key 請從 https://platform.openai.com/api-keys 取得，不要貼在程式碼或 git 裡。

### 3) Phase 0 — 離線跑通（不需要 Stockfish 或 API key）

```bash
# 命令列跑內建 demo（用 mock 引擎，看路由判定是否合理）
python3 demo_cli.py
# 預期輸出：
#   [routine |light] … (silent)
#   [critical|deep ] (play_by_play) 6. Nxf7 （capture、sacrifice）…
#   [notable |light] (play_by_play) 7. Qf3+ 將軍！…

# Streamlit 介面
streamlit run app.py
# 開瀏覽器 http://localhost:8501，按「▶ 下一步」逐步播報
```

### 4) Phase 1 — 接真引擎（讓評估數字有意義）

```bash
# 下載 Stockfish binary：https://stockfishchess.org/download/
# Linux/WSL 範例：
wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64.tar
tar xf stockfish-ubuntu-x86-64.tar

# 把路徑加進 .env
# STOCKFISH_PATH=/path/to/stockfish

python3 demo_cli.py   # 評估數字變正常，開始調 config.py 門檻
```

### 5) Phase 3 — 接真 LLM 播報（已完成）

把 `OPENAI_API_KEY` 填進 `.env` 後直接跑，系統自動偵測並切換：

```bash
python3 demo_cli.py   # critical/deep 步驟自動用 GPT 生成播報
```

> 使用 `gpt-4o-mini`（light 路徑）和 `gpt-4o`（deep 路徑）。
> 模型名稱在 `config.py` 的 `LIGHT_MODEL` / `DEEP_MODEL` 調整。

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
