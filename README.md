## Luess
In this project I try to train a chess position evaluation function using reinforcement learning and compare it to one trained to imitate stockfish evaluations.

Currently the evaluation function trained to imitate stockfish is capable of winning against me.
THere is no evaluation function trained using reinforcement learning yet

# Distilled Stockfish
In Distilled Stockfish the model defined in LuessModel.py is trained using supervised learning to predict the evaluation stockfish would give a position.
This serves two purposes:
1. Validate that the chosen model can capture the nuances of chess
2. Establish a baseline against which the strength of the model trained using reinforcement learning can be measured