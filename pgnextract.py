import chess.pgn
from stockfish import Stockfish
import numpy as np 
import shutil
import h5py

piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

#missing 50 move rule
def board_to_bitmap(board):
    bitBoards = []
    for color in [chess.WHITE,chess.BLACK]:
        for piece_type in piece_types:
            bitBoards.append(int(board.pieces(piece_type, color)))

    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]

    for right in castling_rights:
        bitBoards.append(0xffffffffffffffff if right else 0x0)

    if board.ep_square is not None:
        en_passant_bitboard = 1 << board.ep_square
    else:
        en_passant_bitboard = 0

    bitBoards.append(en_passant_bitboard)

    turn_board = 0xffffffffffffffff if board.turn == chess.WHITE else 0x0

    bitBoards.append(turn_board)

    return np.array(bitBoards, dtype='uint64')



if __name__ == "__main__":
    board = chess.Board()
    print(board)
    print(board_to_bitmap(board))
    sf = Stockfish(path=shutil.which("stockfish"))
    sf.set_depth(8)

    sf.set_fen_position("4r2k/1pR2N2/2bp4/5p2/1P2rP1K/R6P/8/8 b - - 7 52")
    evaluation = sf.get_evaluation()
    print(evaluation)