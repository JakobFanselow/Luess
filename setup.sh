#/usr/bin/env bash
mkdir tmp
mkdir -p data
cd tmp
kaggle datasets files chessmontdb/chessmont-big-dataset
kaggle datasets download chessmontdb/chessmont-big-dataset -f twic.pgn.zst --force 
mv DownloadDataset twic.pgn.zst
kaggle datasets download chessmontdb/chessmont-big-dataset -f pgnmentor.pgn.zst --force 
mv DownloadDataset pgnmentor.pgn.zst
unzstd twic.pgn.zst
unzstd pgnmentor.pgn.zst
rm twic.pgn.zst
rm pgnmentor.pgn.zst
cat twic.pgn pgnmentor.pgn > ../data/data.pgn
rm twic.pgn
rm pgnmentor.pgn
cd ..
rm -d tmp