import random
import numpy as np

from environment.constants import WHITE
from environment.environment import EnvState


def random_strategy(state: EnvState):
    actions = state.get_available_actions()
    return random.choice(actions) if actions else None


def greedy_strategy(state: EnvState):
    actions = state.get_available_actions()
    best_action = None
    max_pieces = -1
    for action in actions:
        new_state = state.copy()
        new_state.act(action)
        piece_diff = np.sum(new_state.board)
        if state.turn == WHITE:
            piece_diff = -piece_diff
        if piece_diff > max_pieces:
            best_action = action
            max_pieces = piece_diff
    return best_action
