"""Minimal Streamlit UI for the chess broadcaster.

    pip install streamlit
    streamlit run app.py
"""
import sys
import streamlit as st
import chess, chess.svg
import config
import tts
from engine.stockfish_client import StockfishClient
from pipeline import make_stepper, moves_from_pgn

st.set_page_config(page_title="Chess Broadcaster", layout="wide")

DEMO_SAN = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5",
            "exd5", "Nxd5", "Nxf7", "Kxf7", "Qf3+", "Ke6", "Nc3"]


def san_to_moves(san_list):
    b, moves = chess.Board(), []
    for s in san_list:
        m = b.parse_san(s); moves.append(m); b.push(m)
    return moves


if "engine" not in st.session_state:
    st.session_state.engine = StockfishClient()
    st.session_state.step, st.session_state.flow = make_stepper(st.session_state.engine)
    if len(sys.argv) >= 2:
        st.session_state.moves = moves_from_pgn(sys.argv[1])
    else:
        st.session_state.moves = san_to_moves(DEMO_SAN)
    st.session_state.ply = 0
    st.session_state.state = {"board": chess.Board(), "move_history": [], "said_so_far": []}
    st.session_state.feed = []

st.title("♟️ Agentic 西洋棋主播")
st.sidebar.caption(f"棋局：{sys.argv[1] if len(sys.argv) >= 2 else '內建 demo'} · 流程：{st.session_state.flow}")
persona = st.sidebar.radio("主播風格", ["calm", "excited", "literary"], index=0)
tts_on = st.sidebar.toggle("🔊 語音播報", value=False, disabled=not config.USE_LLM,
                           help="需要 OPENAI_API_KEY。語氣跟著 register 強度走。")
if st.session_state.engine.mock:
    st.sidebar.warning("MOCK 引擎模式。設定 STOCKFISH_PATH 取得真實評估。")

col1, col2 = st.columns([1, 1])

with col1:
    board = st.session_state.state["board"]
    st.components.v1.html(chess.svg.board(board, size=420), height=440)
    a = st.session_state.state.get("analysis")
    if a:
        st.metric("Stockfish 評估", f"{a['score_cp']/100:+.1f}",
                  delta=f"{a['delta_cp']/100:+.1f}")

    c1, c2 = st.columns(2)
    if c1.button("▶ 下一步", use_container_width=True,
                 disabled=st.session_state.ply >= len(st.session_state.moves)):
        # Guard against double-fired clicks: only advance if the board's actual
        # move count still matches `ply` (i.e. this click hasn't already been applied).
        if len(board.move_stack) == st.session_state.ply:
            mv = st.session_state.moves[st.session_state.ply]
            st.session_state.state["last_move"] = mv
            st.session_state.state["persona"] = persona
            st.session_state.state = st.session_state.step(st.session_state.state)
            st.session_state.state["move_history"].append(mv)
            ev = st.session_state.state["event"]
            if not st.session_state.state["commentary"]:
                # routine moves are silent by design — still show a muted line so
                # the feed visibly advances
                st.session_state.feed.append(
                    (ev["severity_label"], st.session_state.state["route"], None,
                     st.session_state.state["analysis"]["played_san"], None))
            clips = []
            for turn in st.session_state.state["commentary"]:
                audio = tts.speak(turn["text"], turn["speaker"],
                                  st.session_state.state.get("register_intensity", 0.2)
                                  ) if tts_on else None
                if audio:
                    clips.append(audio)
                st.session_state.feed.append(
                    (ev["severity_label"], st.session_state.state["route"],
                     turn["speaker"], turn["text"], audio))
            # one combined clip so the two speakers play in sequence, not on top
            # of each other
            st.session_state.new_audio = b"".join(clips) or None
            st.session_state.ply += 1
        st.rerun()
    if c2.button("⟲ 重置", use_container_width=True):
        st.session_state.ply = 0
        st.session_state.state = {"board": chess.Board(), "move_history": [], "said_so_far": []}
        st.session_state.feed = []
        st.session_state.new_audio = None
        st.rerun()

with col2:
    st.subheader("播報")
    if st.session_state.get("new_audio"):
        st.audio(st.session_state.new_audio, format="audio/mp3", autoplay=True)
        st.session_state.new_audio = None  # autoplay once, not on every rerun
    for label, route, speaker, text, audio in reversed(st.session_state.feed):
        if speaker is None:
            st.caption(f"⚪ {text} ·（例行步，靜默）")
            continue
        color = {"critical": "🔴", "notable": "🟡", "routine": "⚪"}[label]
        st.markdown(f"{color} **{speaker}** · `{route}`  \n{text}")
        if audio:
            st.audio(audio, format="audio/mp3")
