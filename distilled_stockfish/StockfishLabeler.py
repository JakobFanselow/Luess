import chess.pgn
import shutil
from stockfish import Stockfish
from pgnextract import board_to_bitmap
from ChessPositionWriter import ChessPositionWriter
import concurrent.futures
import os
from tqdm import tqdm
import random
import itertools


worker_sf = None

def init_worker(depth):
    global worker_sf
    worker_sf = Stockfish(path=shutil.which("stockfish"))
    worker_sf.set_depth(depth)

def get_position_eval(board):
    worker_sf.set_fen_position(board.fen())
    evaluation = worker_sf.get_evaluation()["value"]
    bitmap = board_to_bitmap(board)
    return (bitmap,evaluation)

def get_game_evals(game, first_labeled=0, skip=0, label_percentage = 1.0):
    game_eval = []
    board = game.board()
    skip_delay = 0
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        if i < first_labeled:
            continue
        if random.random() > label_percentage:
            continue
        if skip_delay != 0:
            skip_delay -= 1
            continue
        skip_delay = skip
        bitmap, evaluation = get_position_eval(board)
        game_eval.append((bitmap,evaluation))
    return game_eval
    
def board_loader(pgn_path, limit):
    with open(pgn_path, 'r') as f:
        i = 0
        while True:
            game = chess.pgn.read_game(f)
            if not game: break
            if i == limit: break
            yield i,game
            i += 1

def label_game_positions(pgn_path, count, depth=8,firstlabeled=7, skip=3, label_percentage = 0.2):
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker,
        initargs=(depth,)
        ) as executor:

        positions = 0
        progress_bar = tqdm(total=count, desc="Analyzing games")
        games = board_loader(pgn_path, count)
        writer = ChessPositionWriter("data/stockfishlabel.h5")
        futures = {}
        for _ in range(os.cpu_count() * 2):
            try:
                idx, board = next(games)
                fut = executor.submit(get_game_evals, board, firstlabeled, skip, label_percentage)
                futures[fut] = idx
            except StopIteration:
                break

        while futures:
            done_future = next(concurrent.futures.as_completed(futures))
            result = done_future.result()
            game_evals = done_future.result()
            

            if game_evals:
                positions += len(game_evals)
                for bitmap,eval in game_evals:
                    writer.add_position(bitmap,eval)

            


            try:
                idx, board = next(games)
                fut = executor.submit(get_game_evals, board, firstlabeled, skip, label_percentage)
                futures[fut] = idx
            except StopIteration:
                1
            progress_bar.update(1)
            del futures[done_future]
        
        #result_map = {
        #    "1-0": 1,
        #    "0-1": -1,
        #    "1/2-1/2": 0
        #}
        writer.flush()
        progress_bar.close()
        print(f"Analyzed {positions} positions!")

            
                


if __name__ == "__main__":
    random.seed(0)
    label_game_positions("data/data.pgn", 120000, depth=15)