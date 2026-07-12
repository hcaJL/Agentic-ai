"""互動對弈模式 — 你下棋，主播＋賽評即時解說。

    streamlit run play.py
"""
import io
import streamlit as st
import chess, chess.svg
import cairosvg
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import config
import tts
from engine.stockfish_client import StockfishClient
from pipeline import make_stepper

st.set_page_config(
    page_title="Chess Broadcaster — 對弈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Board constants ────────────────────────────────────────────────────────────
BOARD_PX = 480
# python-chess uses a fixed 15px margin for coordinate labels (verified from source)
COORD_MARGIN = 15
SQ_PX = (BOARD_PX - 2 * COORD_MARGIN) / 8   # 56.25 px per square

# Lichess green theme
BOARD_COLORS = {
    "square light":          "#eeeed2",
    "square dark":           "#769656",
    "square light lastmove": "#f6f669",
    "square dark lastmove":  "#baca2b",
    "margin":                "#2b2821",
    "coord":                 "#ddb88a",
    "arrow green":           "#15781B80",
    "arrow blue":            "#00338980",
    "arrow red":             "#88202080",
    "arrow yellow":          "#e68f00b0",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten the feed scroll area */
section[data-testid="stVerticalBlock"] { gap: 0 !important; }
/* Remove default top padding from main block */
.main .block-container { padding-top: 1rem; }
/* Thinner hr */
hr { margin: 0.6rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Board helpers ──────────────────────────────────────────────────────────────

def _board_image(board: chess.Board,
                 selected: chess.Square | None,
                 legal_dests: list[chess.Square],
                 orientation: chess.Color,
                 last_move: chess.Move | None) -> Image.Image:
    fill = {}
    if selected is not None:
        fill[selected] = "#f6f669"
        for sq in legal_dests:
            fill[sq] = "#dd5555" if board.piece_at(sq) else "#66aa66"
    svg = chess.svg.board(
        board,
        fill=fill,
        lastmove=last_move,
        orientation=orientation,
        size=BOARD_PX,
        coordinates=True,   # adds margin → pieces never touch the edge
        colors=BOARD_COLORS,
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    return Image.open(io.BytesIO(png))


def _pixel_to_square(x: int, y: int, orientation: chess.Color) -> chess.Square:
    # Clamp to inner board area (exclude coordinate margin)
    x = max(COORD_MARGIN, min(BOARD_PX - COORD_MARGIN - 1, x))
    y = max(COORD_MARGIN, min(BOARD_PX - COORD_MARGIN - 1, y))
    fi = int((x - COORD_MARGIN) / SQ_PX)
    ri = int((y - COORD_MARGIN) / SQ_PX)
    fi = max(0, min(7, fi))
    ri = max(0, min(7, ri))
    if orientation == chess.WHITE:
        return chess.square(fi, 7 - ri)
    return chess.square(7 - fi, ri)


def _can_select(piece: chess.Piece | None, sq: chess.Square,
                board: chess.Board, human_color: str) -> bool:
    if piece is None:
        return False
    if not any(m.from_square == sq for m in board.legal_moves):
        return False
    is_white = piece.color == chess.WHITE
    if human_color == "both":
        return (is_white and board.turn == chess.WHITE) or \
               (not is_white and board.turn == chess.BLACK)
    return (human_color == "white" and is_white and board.turn == chess.WHITE) or \
           (human_color == "black" and not is_white and board.turn == chess.BLACK)


# ── Session init ───────────────────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = StockfishClient()
    st.session_state.step, st.session_state.flow = make_stepper(st.session_state.engine)
    st.session_state.state = {"board": chess.Board(), "move_history": [], "said_so_far": []}
    st.session_state.feed = []
    st.session_state.new_audio  = None
    st.session_state.selected_sq = None
    st.session_state.click_gen   = 0


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ♟️ 對弈設定")
    st.caption(f"流程：{st.session_state.flow}")
    persona = st.radio("主播風格", ["calm", "excited", "literary"], index=0)
    tts_on  = st.toggle("🔊 語音播報", value=False, disabled=not config.USE_LLM,
                        help="需要 OPENAI_API_KEY")
    st.divider()
    vs_engine      = st.radio("對手", ["Stockfish", "雙人對弈"], index=0) == "Stockfish"
    human_is_white = st.radio("你執", ["白", "黑"], index=0,
                              disabled=not vs_engine) == "白"
    engine_depth   = st.slider("引擎深度", 2, 16, 8, disabled=not vs_engine)
    if st.session_state.engine.mock:
        st.warning("MOCK 引擎。設定 STOCKFISH_PATH 取得真實對弈。")

# Cache sidebar values so the fragment can read them on partial reruns
st.session_state["_persona"]        = persona
st.session_state["_vs_engine"]      = vs_engine
st.session_state["_human_is_white"] = human_is_white
st.session_state["_engine_depth"]   = engine_depth
st.session_state["_tts_on"]        = tts_on


# ── Pipeline helper ────────────────────────────────────────────────────────────
def play_one(mv: chess.Move):
    _persona = st.session_state["_persona"]
    _tts_on  = st.session_state["_tts_on"]
    s = st.session_state.state
    s["last_move"] = mv
    s["persona"]   = _persona
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
                          s.get("register_intensity", 0.2)) if _tts_on else None
        if audio:
            clips.append(audio)
        st.session_state.feed.append(
            (ev["severity_label"], s["route"], turn["speaker"], turn["text"], audio))
    st.session_state.new_audio = b"".join(clips) or None


# ── Page header ────────────────────────────────────────────────────────────────
st.title("♟️ 對弈模式 — 主播台即時解說")


@st.fragment
def board_ui():
    """Renders the board. Selection clicks rerun only this fragment;
    completed moves trigger a full-app rerun to refresh the feed."""
    board = st.session_state.state["board"]
    vs_engine_f      = st.session_state.get("_vs_engine", True)
    human_is_white_f = st.session_state.get("_human_is_white", True)
    engine_depth_f   = st.session_state.get("_engine_depth", 8)

    engine_color   = (chess.BLACK if human_is_white_f else chess.WHITE) if vs_engine_f else None
    engine_to_move = vs_engine_f and not board.is_game_over() and board.turn == engine_color
    human_color    = "both" if not vs_engine_f else ("white" if human_is_white_f else "black")
    orientation    = chess.WHITE if (human_is_white_f or not vs_engine_f) else chess.BLACK

    # Turn / game-over indicator
    if board.is_game_over():
        result = board.result()
        st.error(f"對局結束：{result}"
                 f"{'（將殺）' if board.is_checkmate() else ''}")
    elif engine_to_move:
        st.info("🤖 等待引擎走棋…")
    else:
        who = "⬜ 白方" if board.turn == chess.WHITE else "⬛ 黑方"
        chk = " ＊將軍＊" if board.is_check() else ""
        st.info(f"{who} 的回合{chk} — 點選棋子")

    # Board image
    selected_sq = st.session_state.selected_sq
    legal_dests = ([m.to_square for m in board.legal_moves
                    if m.from_square == selected_sq]
                   if selected_sq is not None else [])
    last_mv = board.peek() if board.move_stack else None
    img = _board_image(board, selected_sq, legal_dests, orientation, last_mv)

    if not engine_to_move and not board.is_game_over():
        coords = streamlit_image_coordinates(
            img, width=BOARD_PX,
            key=f"bd_{st.session_state.click_gen}")
        if coords:
            clicked_sq = _pixel_to_square(coords["x"], coords["y"], orientation)
            piece = board.piece_at(clicked_sq)
            st.session_state.click_gen += 1

            if selected_sq is None:
                if _can_select(piece, clicked_sq, board, human_color):
                    st.session_state.selected_sq = clicked_sq
                st.rerun(scope="fragment")   # only board re-renders → no full-page flash

            else:
                promo = None
                src_piece = board.piece_at(selected_sq)
                if (src_piece and src_piece.piece_type == chess.PAWN
                        and chess.square_rank(clicked_sq) in (0, 7)):
                    promo = chess.QUEEN
                mv = chess.Move(selected_sq, clicked_sq, promotion=promo)

                if mv in board.legal_moves:
                    st.session_state.selected_sq = None
                    play_one(mv)
                    b2 = st.session_state.state["board"]
                    if vs_engine_f and not b2.is_game_over():
                        reply = st.session_state.engine.best_move(b2, engine_depth_f)
                        if reply:
                            play_one(reply)
                    st.rerun()               # full rerun → feed + analysis update

                elif (_can_select(piece, clicked_sq, board, human_color)
                      and clicked_sq != selected_sq):
                    st.session_state.selected_sq = clicked_sq
                    st.rerun(scope="fragment")

                else:
                    st.session_state.selected_sq = None
                    st.rerun(scope="fragment")
    else:
        st.image(img, width=BOARD_PX)

    # Engine button
    if engine_to_move and not board.is_game_over():
        if st.button("🤖 讓引擎走棋", use_container_width=True, type="primary"):
            mv = st.session_state.engine.best_move(board, engine_depth_f)
            if mv:
                play_one(mv)
            st.rerun()

    # Evaluation metrics
    a = st.session_state.state.get("analysis")
    if a:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("引擎評估", f"{a['score_cp']/100:+.2f}",
                      delta=f"Δ {a['delta_cp']/100:+.2f}")
        with m2:
            phase_map = {"opening": "開局", "middlegame": "中局", "endgame": "殘局"}
            phase = st.session_state.state.get("event", {}).get("phase", "")
            st.metric("階段", phase_map.get(phase, "—"))

    # Last 4 full moves in compact notation
    mv_hist = st.session_state.state.get("move_history", [])
    if mv_hist:
        tmp = chess.Board()
        sans = []
        for m in mv_hist:
            try:
                sans.append(tmp.san(m)); tmp.push(m)
            except Exception:
                break
        pairs = []
        it = iter(sans)
        for i, w in enumerate(it, 1):
            b_san = next(it, "")
            pairs.append(f"{i}. {w} {b_san}".strip())
        st.caption("  ".join(pairs[-4:]))

    st.divider()
    if st.button("⟲ 重新開局", use_container_width=True):
        st.session_state.state       = {"board": chess.Board(),
                                        "move_history": [], "said_so_far": []}
        st.session_state.feed        = []
        st.session_state.new_audio   = None
        st.session_state.selected_sq = None
        st.session_state.click_gen   = 0
        st.rerun()


# ── Main layout ────────────────────────────────────────────────────────────────
col_board, col_feed = st.columns([10, 9], gap="large")

with col_board:
    board_ui()

# ── Feed column ────────────────────────────────────────────────────────────────
with col_feed:
    st.subheader("📡 即時播報")

    if st.session_state.get("new_audio"):
        st.audio(st.session_state.new_audio, format="audio/mp3", autoplay=True)
        st.session_state.new_audio = None

    feed = st.session_state.feed
    if not feed:
        st.markdown(
            '<p style="color:#888;margin-top:2rem">開始下棋後，播報會出現在這裡。</p>',
            unsafe_allow_html=True)

    SEV_COLOR = {"critical": "#ef4444", "notable": "#f59e0b", "routine": "#6b7280"}
    SEV_EMOJI = {"critical": "🔴",     "notable": "🟡",     "routine": "⚪"}

    for label, route, speaker, text, audio in reversed(feed):
        color = SEV_COLOR.get(label, "#6b7280")
        emoji = SEV_EMOJI.get(label, "⚪")

        if speaker is None:
            st.markdown(
                f'<div style="color:#777;font-size:0.78em;padding:3px 0 3px 4px">'
                f'{emoji} <code>{text}</code>'
                f'<span style="opacity:0.5"> · 例行，靜默</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="border-left:3px solid {color};
            padding:8px 14px;margin:5px 0;
            background:rgba(255,255,255,0.04);
            border-radius:0 6px 6px 0">
  <div style="color:{color};font-size:0.70em;font-weight:700;
              letter-spacing:.05em;margin-bottom:4px">
    {emoji}&nbsp;{speaker.upper()}&nbsp;·&nbsp;{route}
  </div>
  <div style="line-height:1.6;font-size:0.95em">{text}</div>
</div>""", unsafe_allow_html=True)

        if audio:
            st.audio(audio, format="audio/mp3")
