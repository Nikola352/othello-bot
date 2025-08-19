from string import ascii_lowercase
import pandas as pd

from environment.constants import BLACK, WHITE
from environment.environment import EnvState
from model.network import state_to_tensor


def label_to_coords(label: str) -> tuple[int, int]:
    """Convert board square label to (row, column) tuple"""
    row = ascii_lowercase.index(label[0])
    col = int(label[1]) - 1
    return row, col


def prepare_dataset(data_path: str) -> tuple[list, list]:
    states = []
    values = []

    df = pd.read_csv(data_path)
    for _, row in df.iterrows():
        game_moves = row['game_moves']
        winner = row['winner']

        move_pairs = [game_moves[i:i+2] for i in range(0, len(game_moves), 2)]
        moves = [label_to_coords(move) for move in move_pairs]

        state = EnvState()
        for move in moves:
            states.append(state_to_tensor(state))

            if winner == 1:
                value = 1 if state.turn == BLACK else -1
            elif winner == -1:
                value = 1 if state.turn == WHITE else -1
            else:
                value = 0
            values.append(value)

            state.act(move)
    
    return states, values
