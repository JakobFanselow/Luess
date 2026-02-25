import torch
import chess.pgn
from stockfish import Stockfish
import shutil



from LuessModel import LuessModel
from pgnextract import board_to_bitmap

res_blocks = int(input("Enter number res blocks: "))

model = LuessModel(num_res_blocks=res_blocks)

model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

sf = Stockfish(path=shutil.which("stockfish"))
sf.set_depth(15)
model.eval()

def get_position_eval(board):
    sf.set_fen_position(board.fen())
    eval_data = sf.get_evaluation()

    if eval_data["type"] == "cp":
        evaluation = eval_data["value"]
    else:
        evaluation = 10000
    if board.turn == chess.BLACK:
        evaluation = -evaluation
    bitmap = board_to_bitmap(board)
    return bitmap,evaluation


def bitmap_to_tensor(x):
    shifts = torch.arange(63, -1, -1, dtype=torch.int64)
    data_tensor = torch.from_numpy(x.astype('int64'))
    unpacked = (data_tensor.unsqueeze(-1) >> shifts) & 1
    data = unpacked.float()
    data = data.view(18, 8, 8)
    return data



while True:
    command = input("Enter FEN: ")
    if command == "exit":
        break

    board = chess.Board()
    board.set_fen(command)
    _,sf_eval = get_position_eval(board)
    sf_eval /= 100
    bitmap = board_to_bitmap(board)

    #print(board)

    tensor = bitmap_to_tensor(bitmap)

    tensor = tensor.unsqueeze(0)

    model_eval = model.forward(tensor).item()/100
    
    print(f"I think {model_eval} but stockfish thinks {sf_eval}")