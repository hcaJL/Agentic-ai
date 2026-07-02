# Agentic 西洋棋即時主播系統 — 技術規格

> 給 AI coding 工具(Claude Code / Cursor 等)的建置參考文件。
> 本文以繁體中文敘述,技術名詞與程式碼一律用英文。請嚴格遵守第 2 節「核心設計原則」與第 12 節「反模式」,這兩節定義了本專案與「玩具版棋評器」的差別。

---

## 1. 專案概述

建立一個**即時西洋棋主播系統**:輸入一場棋局(逐步落子 / FEN / PGN / 引擎對局),系統即時產生**有臨場感、戰略深度、且絕不亂講**的播報。

**一句話定位:** 這不是一個「會換語氣講評的 LLM」,而是一個**會判斷何時該激動、何時該閉嘴、記得住整局脈絡、且所有棋理都有引擎背書的多主播台**。

**為什麼不是玩具:** 「LLM + Stockfish 產生旁白」已有大量現成實作(學術與 GitHub)。本專案的技術重量集中在三件少人做的事:
1. **事件驅動的動態路由**(Director):同一步棋,系統自己決定花多少運算、走哪條路。
2. **工具接地(tool grounding)**:LLM 只能複述引擎與盤面的客觀事實,藉此根除棋局幻覺。
3. **分層記憶與回扣**:跨整局記住關鍵時刻,做出「還記得第 12 手那個弱兵嗎」這種有脈絡的播報。

---

## 2. 核心設計原則(不可違反)

1. **棋理真相只來自工具,不來自 LLM。** 所有評估、最佳線、戰術判斷一律由 Stockfish + python-chess 產生。LLM **不得**自行計算變化、評估局面或宣稱任何不在 `FactsPacket` 裡的事實。LLM 的角色是「把結構化事實講成人話」,不是「下棋」。
2. **語氣是輸出,不是下拉選單。** 主播的情緒強度(register)由**事件嚴重度推導**,不是使用者撥的開關。`persona`(激動 / 冷靜 / 文謅謅)只是套在已接地內容上的「風格皮」,不是專案核心。
3. **不是每步都深析。** 例行步走輕量路徑(便宜、快速、甚至靜默);只有關鍵步才觸發深度路徑(更深引擎搜索 + 強模型 + 記憶)。把算力花在刀口上。
4. **靜默也是播報。** 平淡的步數可以不講話。不要強迫系統對每一步都產生文字。
5. **本地優先,確保 demo 穩定。** 使用本地 Stockfish binary,不依賴任何外部分析網站。關鍵分析結果預先快取作為防呆。

---

## 3. 技術棧

| 層 | 技術 | 角色 |
|---|---|---|
| Orchestration | **LangGraph** | 整張流程圖、Director 的 conditional routing、跨手狀態管理 |
| LLM / 整合 | **LangChain** | LLM 呼叫、prompt 模板、ChromaDB 檢索整合 |
| 棋局 | **python-chess** | 盤面狀態、走法解析、PGN、UCI 引擎介面 |
| 引擎 | **Stockfish 17**(本地 binary) | 局面評估、最佳線、multipv |
| 記憶 | **ChromaDB** | 分層記憶向量庫 |
| UI | **Streamlit** | 互動式棋盤 + 即時播報展示 |
| 語音(選配) | TTS(ElevenLabs API 或本地 pyttsx3) | 語音播報,情緒對齊 register |

LLM 建議:輕量路徑用小/快模型(如 haiku 級),深度路徑用強模型(如 sonnet 級)。模型字串以 config 管理,不要寫死。

---

## 4. 系統架構(資料流)

```
輸入:每一步棋局
        │
        ▼
[盤面感知]  python-chess 更新盤面 + Stockfish 淺層評估(算前後評估差)
        │
        ▼
[事件偵測]  判定事件類型 + 嚴重度分數
        │
        ▼
[Director 路由]  ── 依嚴重度選擇路徑 ──┐
        │                              │
   (例行步)                        (關鍵步)
        ▼                              ▼
[輕量路徑]                       [深度路徑]
 簡述或靜默                   更深 Stockfish 搜索
 便宜模型/模板          → 記憶檢索(ChromaDB)
                        → 主播台(play-by-play + analyst,接地生成)
        │                              │
        └──────────────┬───────────────┘
                       ▼
                   [輸出]  文字 / 語音(Streamlit 顯示)
                       │
                       ▼
              ↻ 回寫短期記憶(供回扣與一致性),回到下一手
```

---

## 5. 核心資料結構

以下為跨節點傳遞的型別,請以 `TypedDict` / `pydantic` 實作。

### 5.1 LangGraph State

```python
from typing import TypedDict, Literal, Optional
import chess

class BroadcastState(TypedDict):
    board: chess.Board                  # 當前盤面(python-chess)
    move_history: list[chess.Move]      # 整局走法
    last_move: Optional[chess.Move]     # 剛落的這一步
    analysis: "StockfishAnalysis"       # 引擎分析結果
    event: "EventInfo"                  # 事件偵測結果
    route: Literal["light", "deep"]     # Director 的決策
    facts: "FactsPacket"                # 餵給主播 agent 的接地事實包
    retrieved_memory: dict              # ChromaDB 撈回的回扣與棋理
    commentary: list["CommentaryTurn"]  # 本步產生的播報(可能多位主播)
    said_so_far: list[str]              # 短期記憶:整局已講過的內容摘要
    persona: str                        # 當前主播風格皮
```

### 5.2 StockfishAnalysis

```python
class StockfishAnalysis(TypedDict):
    score_cp: int            # 當前評估(centipawn,正=白優)
    score_before_cp: int     # 落子前評估
    delta_cp: int            # 評估變化幅度(關鍵訊號)
    mate_in: Optional[int]   # 若為將殺序列,幾步將殺
    best_line_san: list[str] # 最佳線(SAN)
    top_moves: list[dict]    # multipv:[{san, score_cp}, ...]
    depth: int               # 本次搜索深度
```

### 5.3 EventInfo

```python
class EventInfo(TypedDict):
    types: list[str]               # 例:["sacrifice", "check"]
    motifs: list[str]              # 戰術:["fork", "pin", "discovered_attack", ...]
    phase: Literal["opening", "middlegame", "endgame"]
    severity_score: float          # 0.0–1.0
    severity_label: Literal["routine", "notable", "critical"]
```

### 5.4 FactsPacket(接地事實包 — 主播 agent 的唯一資訊來源)

```python
class FactsPacket(TypedDict):
    move: dict       # {san, uci, piece, is_capture, is_check, is_mate, is_promotion}
    board: dict      # {fen, move_number, side_to_move, material_balance, phase}
    eval: dict       # {score_cp, delta_cp, mate_in, best_line_san, top_moves}
    event: dict      # {types, motifs, severity_label}
    memory: dict     # {callbacks: [...], theory: [...]}  深度路徑才填
    register: dict   # {intensity: 0.0–1.0, persona: str}
```

> **接地鐵則:** 主播 agent 的 system prompt 必須明確規定:「你只能陳述 `FactsPacket` 中存在的事實。不得自行評估局面、計算變化或杜撰最佳線。若某項資訊不在 packet 中,就不要宣稱它。」

---

## 6. 元件規格

### 6.1 盤面感知(perception node)

- 用 `python-chess` 維護 `chess.Board`,套用 `last_move`,更新 `move_history`。
- 用 `chess.engine.SimpleEngine.popen_uci()` 驅動本地 Stockfish。
- **每步都算「落子前」與「落子後」兩個評估**,得出 `delta_cp`。落子前評估可沿用上一步快取,避免重算。
- 此處用**淺層深度**(如 depth 12 或 movetime 50–100ms),求快。深度搜索留給深度路徑。
- 計算 `material_balance`、判定 `phase`(opening:在開局理論內或前 ~12 手;endgame:盤面子力 ≤ 某門檻或后已交換且子力少;其餘 middlegame)。

### 6.2 事件偵測(event_detector node)

吃 `board` + `StockfishAnalysis`,輸出 `EventInfo`。判定邏輯見第 7 節嚴重度表。需偵測:

- **評估類:** blunder(評估驟降)、brilliant(唯一守住的好棋且與次佳差距大)。
- **強制類:** check、checkmate、stalemate、promotion。
- **物質類:** capture(吃子,依子力價值分級)、sacrifice(給子但評估守住或變好)。
- **戰術 motif:** fork、pin、skewer、discovered attack(可用 python-chess 的 attack/pin API 偵測,MVP 可先做 fork 與 check,其餘列 future work)。
- **階段類:** entering_endgame、opening_deviation(脫離開局書)。

### 6.3 Director / 路由(director node + conditional edge)⭐ 核心

這是 agentic 核心。輸入 `EventInfo` + 節奏脈絡,輸出 `route` 與 `register`。

```python
def director(state: BroadcastState) -> dict:
    sev = state["event"]["severity_label"]
    # 1. 路由決策
    route = "deep" if sev == "critical" else "light"
    # 2. 節奏修正:太久沒深評,可把一個 notable 升級為 deep(避免冷場)
    # 3. 情緒 register:由嚴重度推導,不是使用者選的
    intensity = {"routine": 0.2, "notable": 0.55, "critical": 0.9}[sev]
    # 4.(可選)決定由誰講:light→play_by_play;deep→play_by_play + analyst
    return {"route": route, "register_intensity": intensity}
```

- LangGraph 用 `add_conditional_edges` 依 `route` 分流到 `light_commentary` 或 `deep_pipeline`。
- **節奏脈絡**很重要:參考 `said_so_far`,若連續 N 步都是 routine,可主動把下一個 notable 升級為 deep,或在 routine 步保持靜默以製造對比。

### 6.4 記憶(memory node,ChromaDB,分層)

三層,**MVP 先做前兩層**:

1. **短期 / 本局記憶(必做):** 存本局已播報的關鍵時刻摘要(move number + 一句話)。用途:(a) 避免重複;(b) 回扣(「第 12 手的弱兵現在要了命」)。可存在 state 的 `said_so_far`,或寫進 ChromaDB 以本局 game_id 命名 collection。
2. **長期棋理知識(必做):** 預先 seed 開局理論、戰術命名、戰略概念進 ChromaDB。深度路徑時依當前盤面 embedding 檢索相關棋理,餵進 `FactsPacket.memory.theory`。用途:接地的戰略解說。
3. **棋手風格檔(future work):** 匯入特定棋士歷史棋譜,抽取開局偏好與慣性戰術。詳見第 9 節。

> 注意:**記憶檢索 ≠ 動態決策。** 撈資料是固定流程;動態決策是 Director 的路由。簡報時別把記憶當成「動態決策」來講。

### 6.5 主播台 / Commentary Booth(booth nodes)

- **play_by_play agent:** 即時、節奏、情緒。負責「剛剛走了什麼、立即威脅」。輕量與深度路徑都會用到。
- **analyst agent:** 戰略深度、「為什麼」。只在深度路徑出場。引用 `eval.best_line_san` 與 `memory.theory`。
- 兩位 agent **都只吃 `FactsPacket`**,受接地鐵則約束。
- 輪替協調:由 Director / graph 決定發言順序(深度路徑:play_by_play 先報事件,再交棒 analyst 深入)。MVP 可先序列呼叫,不必做真正的並行對話。
- **每次生成前檢查 `said_so_far`**,避免重複用語與重複觀點。

### 6.6 Persona / 風格層(personas)

- 提供至少三種 persona prompt 模板:`excited`(激動體育主播)、`calm`(冷靜大師)、`literary`(文謅謅)。
- persona 套在**已接地的內容**上,只改語氣與遣詞,**不得改變或新增事實**。
- `register.intensity`(來自 Director)調整 persona 的強度(同一個 excited persona,intensity 0.9 比 0.2 更激昂)。

### 6.7 輸出(render node + update_memory node)

- `render`:組合本步播報文字,送 Streamlit 顯示(同步棋盤、評估條、最佳走法)。
- 選配 TTS:依 `register` 與 persona 合成語音。
- `update_memory`:把本步重點寫回短期記憶 / `said_so_far`,供後續回扣與一致性。
- 回到下一手(LangGraph loop 或由 Streamlit 驅動逐步呼叫)。

---

## 7. 事件嚴重度判定表(具體起始值)

> 這張表是整條路由的命脈,**一動,系統行為就跟著變**。以下為起始門檻,需在實測中校準。`cp` = centipawn(100cp ≈ 一個兵)。

### 7.1 評估變化 → 基礎嚴重度

| `abs(delta_cp)` | 基礎 label |
|---|---|
| < 50 | routine |
| 50 – 150 | notable |
| ≥ 150 | critical |

### 7.2 無條件升級為 critical 的事件(不管 delta)

- checkmate / 將殺序列出現(`mate_in` 不為 None)
- blunder:`delta_cp` 對行棋方不利且 ≥ 200
- sacrifice:主動給出 ≥ 1 個輕子(minor piece)的子力,但落子後評估守住或變好
- promotion(升變)
- entering_endgame(首次進入殘局)
- brilliant:該步為唯一守住的好棋,且與次佳手評估差距 ≥ 150cp

### 7.3 升級為(至少)notable 的事件

- check(將軍)
- capture ≥ rook 價值的子
- 偵測到 fork / pin / skewer / discovered attack

### 7.4 severity_score(0–1,給 register 用)

```
score = clip(abs(delta_cp) / 300, 0, 1)
若命中 7.2 任一事件:score = max(score, 0.9)
若命中 7.3 任一事件:score = max(score, 0.55)
label 由 score 反推:>=0.8→critical, >=0.4→notable, else routine
```

---

## 8. LangGraph 圖結構

```python
from langgraph.graph import StateGraph, END

g = StateGraph(BroadcastState)

g.add_node("perception", perception_node)
g.add_node("detect_event", event_detector_node)
g.add_node("director", director_node)
g.add_node("light_commentary", light_commentary_node)   # 輕量路徑
g.add_node("deep_analyze", deep_analyze_node)            # 更深 Stockfish
g.add_node("retrieve_memory", memory_node)
g.add_node("booth", booth_node)                         # play_by_play + analyst
g.add_node("render", render_node)
g.add_node("update_memory", update_memory_node)

g.set_entry_point("perception")
g.add_edge("perception", "detect_event")
g.add_edge("detect_event", "director")

# Director 的 conditional routing — 動態決策核心
g.add_conditional_edges(
    "director",
    lambda s: s["route"],
    {"light": "light_commentary", "deep": "deep_analyze"},
)

# 深度路徑串接
g.add_edge("deep_analyze", "retrieve_memory")
g.add_edge("retrieve_memory", "booth")

# 兩條路徑匯流
g.add_edge("light_commentary", "render")
g.add_edge("booth", "render")
g.add_edge("render", "update_memory")
g.add_edge("update_memory", END)   # 單步結束;由外層逐步驅動下一手

graph = g.compile()
```

---

## 9. MVP 範圍 vs Future work

### MVP(競賽 demo 必做)
- 盤面感知 + Stockfish 評估 + delta
- 事件偵測(至少:check / checkmate / capture 分級 / blunder / sacrifice / entering_endgame)
- Director 路由(light / deep)+ register
- 輕量路徑 + 深度路徑
- 分層記憶第 1、2 層(短期 + 棋理)
- 主播台:play_by_play + analyst,接地生成
- 三種 persona 風格皮
- Streamlit 展示(棋盤 + 評估條 + 播報)
- 三層消融(見第 10 節)

### Future work(寫進提案、標明延伸,不綁進 MVP)
- **棋手風格檔:** 匯入特定棋士歷史棋譜,抽取開局偏好與慣性戰術,使播報具備「這正是他一貫的西西里套路」這種選手洞察。
- **語音播報(TTS):** 情緒對齊 register 的合成語音。若 MVP 行有餘力可提前納入(大眾票鉤子)。
- **LangGraph 流程可視化:** 現場逐節點亮起的白箱 demo。
- 進階戰術 motif 偵測(skewer / discovered attack 等)。

---

## 10. 三層消融(eval harness,獨立於 runtime)

> 這是 demo 的鑑別度來源,**不是即時流程的一部分**,需單獨實作於 `eval/ablation.py`。

同一場棋局、同一條流程,把「記憶」模組換三種設定,比較解說品質:

| 版本 | 設定 |
|---|---|
| baseline | 無記憶,僅當步事實 |
| RAG | 傳統向量檢索(僅棋理,無本局回扣) |
| full | 本專案分層記憶(棋理 + 本局回扣 + 一致性) |

比較維度:解說一致性、回扣能力、洞察深度、(可選)人工評分。

---

## 11. 建議專案結構

```
chess-broadcaster/
├── app.py                      # Streamlit entry
├── config.py                   # 門檻、路徑、模型字串、Stockfish binary 路徑
├── requirements.txt
├── graph/
│   ├── state.py                # BroadcastState, FactsPacket, EventInfo, ...
│   ├── build_graph.py          # LangGraph 組裝(第 8 節)
│   └── nodes/
│       ├── perception.py
│       ├── event_detector.py
│       ├── director.py         # 路由 + register(核心)
│       ├── memory.py
│       ├── booth.py            # play_by_play + analyst
│       └── render.py
├── engine/
│   ├── stockfish_client.py     # UCI 封裝、快取
│   └── chess_utils.py          # material、phase、motif 偵測
├── memory/
│   ├── chroma_store.py
│   └── seed_theory.py          # 載入開局/戰術/戰略知識
├── personas/
│   └── personas.py             # 三種 persona prompt 模板
├── eval/
│   └── ablation.py             # 三層消融
└── games/                      # demo 用的預選高張力 PGN
```

---

## 12. 反模式 / Non-goals(務必避免)⚠️

AI coding 工具最容易把這專案做成玩具版。以下行為**明確禁止**:

1. **禁止讓 LLM 評估局面或算棋。** 不要出現「請 LLM 判斷這手好不好」「請 LLM 給最佳走法」。棋理一律來自 Stockfish。LLM 只複述 `FactsPacket`。
2. **禁止把 persona / 語氣當成核心功能或固定下拉選單作為賣點。** 語氣強度由 `register`(事件嚴重度)推導。persona 只是風格皮。
3. **禁止每步都呼叫深模型 / 深搜索。** 例行步必須走輕量路徑。違反就失去動態決策的意義,也會慢到無法即時。
4. **禁止依賴外部分析網站。** 用本地 Stockfish binary。
5. **禁止強迫每步都產生文字。** routine 步可以靜默。
6. **禁止生成前不檢查 `said_so_far`。** 必須避免重複用語與重複觀點。
7. **不要把「記憶檢索」當成「動態決策」。** 動態決策是 Director 的路由分支。

---

## 13. 建議建置順序(milestones)

- **M0 — 感知地基:** perception + Stockfish + event_detector。先把每步的 `EventInfo` 印到 console,人工驗證嚴重度判定合理。
- **M1 — 接地單主播:** 單一 play_by_play agent,只吃 `FactsPacket`,文字輸出到 Streamlit。先確認「接地、不亂講」成立。
- **M2 — Director 路由(核心):** 加入 light / deep 分流 + register。這是 agentic 核心,做出來就有技術深度。
- **M3 — 記憶 + 主播台:** ChromaDB 分層記憶 + analyst agent + 回扣。
- **M4 — 三層消融:** eval harness,產出對照數據。
- **M5(stretch):** TTS 語音、流程可視化、棋手風格檔。

---

## 14. 對外定位(放進 README / 簡報開場)

> 我們做的不是會換語氣的棋評機,而是一個**會判斷何時該激動、記得住整局脈絡、且所有棋理都有引擎背書**的多主播台。技術重心在三處:事件驅動的動態路由、工具接地以根除棋局幻覺、以及分層記憶帶來的跨局回扣——這三點是現有的單一 LLM 棋評實作較少觸及的。
