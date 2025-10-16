from string import ascii_lowercase
import pandas as pd

from environment.constants import BLACK
from model.settings import GAMMA
from model.agent import ActorCriticAgent
from environment.environment import EnvState
from model.network import state_to_tensor


def label_to_coords(label: str) -> tuple[int, int]:
    """Convert board square label to (row, column) tuple"""
    row = ascii_lowercase.index(label[0])
    col = int(label[1]) - 1
    return row, col


def prepare_dataset(data_path: str):
    states = []
    actions = []
    values = []

    df = pd.read_csv(data_path)
    for _, row in df.iterrows():
        winner = row['winner']
        game_moves = row['game_moves']

        move_pairs = [game_moves[i:i+2] for i in range(0, len(game_moves), 2)]
        moves = [label_to_coords(move) for move in move_pairs]
        num_moves = len(moves)

        state = EnvState()
        for i, move in enumerate(moves):
            result_from_current_perspective = (winner if state.turn == BLACK else -winner)

            moves_left = num_moves - i
            value = (GAMMA ** moves_left) * result_from_current_perspective

            states.append(state_to_tensor(state))
            actions.append(move[0] * 8 + move[1])
            values.append(value)

            state.act(move)

    return states, actions, values


def preload_expert_memory(agent: ActorCriticAgent, data_path: str, max_games: int = None):
    """Fill replay memory with expert game trajectories"""
    df = pd.read_csv(data_path)
    if max_games:
        df = df.head(max_games)

    for game_moves in df['game_moves']:
        move_pairs = [game_moves[i:i+2] for i in range(0, len(game_moves), 2)]
        moves = [label_to_coords(move) for move in move_pairs]

        state = EnvState()
        for move in moves:
            prev_state = state.copy()
            reward = state.act(move)
            next_state = state

            agent.add_to_memory(prev_state, move, reward, next_state)

            if next_state.is_final():
                break