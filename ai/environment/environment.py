from collections import deque

import numpy as np
from model.settings import GUIDED_REWARD_FACTOR
from environment.constants import BLACK, EMPTY, WHITE
from othello_engine import PyGameState as GameState

HISTORY_CAP = 2

class EnvState(object):
    def __init__(self):
        self.game_state = GameState()
        self.prev_states = deque()

    @property
    def turn(self) -> int:
        return self.game_state.get_turn()
    
    @property
    def board(self) -> np.ndarray:
        return self.game_state.get_board()

    def get_available_actions(self) -> list[tuple[int, int]]:
        return self.game_state.get_legal_moves()

    def act(self, action: tuple[int, int]) -> int:
        self.prev_states.append(self.copy())
        if len(self.prev_states) > HISTORY_CAP:
            self.prev_states.popleft()

        self.game_state = self.game_state.play_move(action[0], action[1])
        
        if self.is_final():
            return self.get_result() if -self.turn == BLACK else -self.get_result()
        
        heuristic_reward = GUIDED_REWARD_FACTOR * self.reward_heuristic()
        if -self.turn == WHITE:
            heuristic_reward = -heuristic_reward
        return heuristic_reward

    def skip_move(self):
        self.prev_states.append(self.copy())
        if len(self.prev_states) > HISTORY_CAP:
            self.prev_states.popleft()
        
        self.game_state = self.game_state.skip_turn()

    def is_final(self) -> bool:
        return self.game_state.is_final()

    def get_result(self) -> int:
        score = self.game_state.get_score()
        return 1 if score > 0 else -1 if score < 0 else 0 # reward clipping

    def get_legal_moves_mask(self) -> np.ndarray:
        return self.game_state.get_legal_moves_mask()
    
    def get_stability_mask(self) -> tuple[np.ndarray, np.ndarray]:
        return self.game_state.get_stability_mask()
    
    def reward_heuristic(self) -> float:
        return (
            3 * self.game_state.get_relative_corner_diff() + 
            2 * self.game_state.get_relative_stability_diff() +
            1 * self.game_state.get_relative_piece_diff()
        ) / 6.0
    
    def copy(self) -> "EnvState":
        state = EnvState()
        state.game_state = self.game_state.clone()
        return state