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
import requests
import zstandard as zstd
import io
import re

GAME_TERMINATOR = re.compile(r" (1-0|0-1|1/2-1/2|\*)\n")

def get_game_evals(game, first_labeled=8, skip=4, label_percentage=0.15):

    game_evals = []
    board = game.board()
    skip_counter = 0
    node = game
    
    

    
    for i , move in enumerate(game.mainline_moves()):
        board.push(move)
        node = node.next()
        
        
        if i < first_labeled:
            continue
            
        
            
        if skip_counter > 0:
            skip_counter -= 1
            continue
        if random.random() > label_percentage:
            continue
        
        eval_obj = node.eval()
        if eval_obj is None:
            continue


        evaluation = eval_obj.pov(chess.WHITE).score(mate_score=10000)
        
        skip_counter = skip
        
        bitmap = board_to_bitmap(board)
        game_evals.append((bitmap, evaluation))
        
    return game_evals


def evaluated_game_loader(url):
    response = requests.get(url, stream=True)
    dctx = zstd.ZstdDecompressor()
    i = 0
    with dctx.stream_reader(response.raw) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        
        game_buffer = []
        for line in text_stream:
            game_buffer.append(line)
            
            if GAME_TERMINATOR.search(line):
                i += 1
                full_game_text = "".join(game_buffer)
                
                if "[%eval" in full_game_text:
                    game = chess.pgn.read_game(io.StringIO(full_game_text))
                    yield i, game
                
                game_buffer = [] 

def label_game_positions(url,firstlabeled=8, skip=6, label_percentage = 0.15):
    position_count = 0
    game_count = 0
    writer = ChessPositionWriter("data/downloaded_stockfishlabel.h5")
    progress_bar = tqdm(desc="Processing", unit=" games")
    game_gen = evaluated_game_loader(url)
    
    for idx, game in game_gen:
            
        game_evals = get_game_evals(
            game, 
            first_labeled=firstlabeled, 
            skip=skip, 
            label_percentage=label_percentage
        )

        if game_evals:
            for bitmap, evaluation in game_evals:
                writer.add_position(bitmap, evaluation)
            position_count += len(game_evals)
        
        progress_bar.n = idx 
        progress_bar.refresh()

                    
    writer.flush()
    progress_bar.close()
    print(f"\nFinished! Saved {position_count} positions from {game_count} games.")


if __name__ == "__main__":
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst"
    label_game_positions(url)
