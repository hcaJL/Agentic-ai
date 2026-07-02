"""Pure python-chess helpers — no engine needed. Fully unit-testable."""
import chess

PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def material_balance(board: chess.Board) -> int:
    """Positive = White ahead (in pawns)."""
    bal = 0
    for piece_type, val in PIECE_VALUE.items():
        bal += val * len(board.pieces(piece_type, chess.WHITE))
        bal -= val * len(board.pieces(piece_type, chess.BLACK))
    return bal


def non_pawn_material(board: chess.Board) -> int:
    total = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += PIECE_VALUE[pt] * (
            len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        )
    return total


def game_phase(board: chess.Board) -> str:
    """Heuristic phase. Tune to taste."""
    if board.fullmove_number <= 10:
        return "opening"
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    if non_pawn_material(board) <= 10 or (queens == 0 and non_pawn_material(board) <= 16):
        return "endgame"
    return "middlegame"


def captured_value(board_before: chess.Board, move: chess.Move) -> int:
    """Value of the piece captured by `move` on `board_before`. 0 if not a capture."""
    if board_before.is_en_passant(move):
        return PIECE_VALUE[chess.PAWN]
    victim = board_before.piece_at(move.to_square)
    return PIECE_VALUE[victim.piece_type] if victim else 0


def is_en_prise(board_after: chess.Board, move: chess.Move) -> tuple:
    """Is the piece that just moved hanging? Returns (hanging: bool, piece_value: int).
    One-ply proxy for sacrifice/offer detection."""
    sq = move.to_square
    piece = board_after.piece_at(sq)
    if not piece:
        return (False, 0)
    opp = board_after.turn                 # side to move = mover's opponent
    attackers = board_after.attackers(opp, sq)
    defenders = board_after.attackers(not opp, sq)
    if not attackers:
        return (False, PIECE_VALUE[piece.piece_type])
    min_attacker = min(PIECE_VALUE[board_after.piece_at(s).piece_type] for s in attackers)
    pv = PIECE_VALUE[piece.piece_type]
    hanging = (len(attackers) > len(defenders)) or (min_attacker < pv)
    return (hanging, pv)


def detects_fork(board_after: chess.Board, move: chess.Move) -> bool:
    """Crude fork heuristic: the moved piece now attacks >=2 enemy pieces of value >= knight."""
    attacker_color = not board_after.turn  # side that just moved
    targets = 0
    for sq in board_after.attacks(move.to_square):
        p = board_after.piece_at(sq)
        if p and p.color != attacker_color and PIECE_VALUE[p.piece_type] >= 3:
            targets += 1
    return targets >= 2
