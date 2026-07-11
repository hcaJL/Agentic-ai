"""互動對弈模式 — 你下棋，主播＋賽評即時解說。

    streamlit run play.py

對手可選 Stockfish（可調深度）或雙人對弈；你的每一步和引擎的回應都走同一條
LangGraph 管線（感知 → 事件 → Director 路由 → 記憶 → 主播台），所以解說有
嚴重度分級、棋理引用與本局回扣，跟重播模式（app.py）完全一致。
"""
import streamlit as st
import chess, chess.svg
import config
import tts
from engine.stockfish_client import StockfishClient
from pipeline import make_stepper

st.set_page_config(page_title="Chess Broadcaster — 對弈", layout="wide")

if "engine" not in st.session_state:
    st.session_state.engine = StockfishClient()
    st.session_state.step, st.session_state.flow = make_stepper(st.session_state.engine)
    st.session_state.state = {"board": chess.Board(), "move_history": [], "said_so_far": []}
    st.session_state.feed = []
    st.session_state.new_audio = None

st.title("♟️ 對弈模式 — 主播台實況解說你的棋")
st.sidebar.caption(f"流程：{st.session_state.flow}")
persona = st.sidebar.radio("主播風格", ["calm", "excited", "literary"], index=0)
tts_on = st.sidebar.toggle("🔊 語音播報", value=False, disabled=not config.USE_LLM,
                           help="需要 OPENAI_API_KEY。語氣跟著 register 強度走。")
vs_engine = st.sidebar.radio("對手", ["Stockfish", "雙人對弈"], index=0) == "Stockfish"
human_is_white = st.sidebar.radio("你執", ["白", "黑"], index=0,
                                  disabled=not vs_engine) == "白"
engine_depth = st.sidebar.slider("引擎棋力（搜索深度）", 2, 16, 8,
                                 disabled=not vs_engine)
if st.session_state.engine.mock:
    st.sidebar.warning("MOCK 引擎模式。設定 STOCKFISH_PATH 取得真實評估與對手。")


def play_one(mv: chess.Move):
    """Run one move (human's or engine's) through the pipeline and into the feed."""
    s = st.session_state.state
    s["last_move"] = mv
    s["persona"] = persona
    s = st.session_state.step(s)
    s["move_history"].append(mv)
    st.session_state.state = s
    ev = s["event"]
    clips = []
    if not s["commentary"]:
        st.session_state.feed.append(
            (ev["severity_label"], s["route"], None, s["analysis"]["played_san"], None))
    for turn in s["commentary"]:
        audio = tts.speak(turn["text"], turn["speaker"],
                          s.get("register_intensity", 0.2)) if tts_on else None
        if audio:
            clips.append(audio)
        st.session_state.feed.append(
            (ev["severity_label"], s["route"], turn["speaker"], turn["text"], audio))
    st.session_state.new_audio = b"".join(clips) or None


board: chess.Board = st.session_state.state["board"]
engine_color = (chess.BLACK if human_is_white else chess.WHITE) if vs_engine else None
engine_to_move = (vs_engine and not board.is_game_over()
                  and board.turn == engine_color)

col1, col2 = st.columns([1, 1])

with col1:
    st.components.v1.html(
        chess.svg.board(board, size=420,
                        orientation=chess.WHITE if (human_is_white or not vs_engine)
                        else chess.BLACK,
                        lastmove=board.peek() if board.move_stack else None),
        height=440)
    a = st.session_state.state.get("analysis")
    if a:
        st.metric("Stockfish 評估", f"{a['score_cp']/100:+.1f}",
                  delta=f"{a['delta_cp']/100:+.1f}")

    if board.is_game_over():
        st.success(f"對局結束：{board.result()}"
                   f"{'（將殺）' if board.is_checkmate() else ''}")
    elif engine_to_move:
        if st.button("🤖 讓引擎走棋", use_container_width=True, type="primary"):
            mv = st.session_state.engine.best_move(board, engine_depth)
            if mv:
                play_one(mv)
            st.rerun()
    else:
        legal = sorted(board.san(m) for m in board.legal_moves)
        turn_label = "白方" if board.turn else "黑方"
        pick = st.selectbox(f"輪到{turn_label}——選擇著法", legal, key=f"pick_{len(board.move_stack)}")
        if st.button("▶ 下這步", use_container_width=True, type="primary"):
            try:
                mv = board.parse_san(pick)
            except ValueError:
                mv = None  # stale selection after a rerun — ignore the click
            if mv:
                play_one(mv)
                # engine replies immediately in the same click, so the human
                # never has to press a second button
                b2 = st.session_state.state["board"]
                if vs_engine and not b2.is_game_over():
                    reply = st.session_state.engine.best_move(b2, engine_depth)
                    if reply:
                        play_one(reply)
            st.rerun()

    if st.button("⟲ 重新開局", use_container_width=True):
        st.session_state.state = {"board": chess.Board(), "move_history": [], "said_so_far": []}
        st.session_state.feed = []
        st.session_state.new_audio = None
        st.rerun()

with col2:
    st.subheader("播報")
    if st.session_state.get("new_audio"):
        st.audio(st.session_state.new_audio, format="audio/mp3", autoplay=True)
        st.session_state.new_audio = None
    for label, route, speaker, text, audio in reversed(st.session_state.feed):
        if speaker is None:
            st.caption(f"⚪ {text} ·（例行步，靜默）")
            continue
        color = {"critical": "🔴", "notable": "🟡", "routine": "⚪"}[label]
        st.markdown(f"{color} **{speaker}** · `{route}`  \n{text}")
        if audio:
            st.audio(audio, format="audio/mp3")
