import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment.constants import BLACK, WHITE
from environment.environment import EnvState


class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc4 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=8*8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def state_to_tensor(state: EnvState) -> torch.Tensor:
    board = state.board

    # Current state channels
    black_pieces = (board == BLACK).astype(np.float32)
    white_pieces = (board == WHITE).astype(np.float32)

    legal_moves = state.get_legal_moves_mask()

    player_indicator = np.ones((8,8), dtype=np.float32) if state.turn == BLACK else np.zeros((8,8), dtype=np.float32)

    # Previous states (always include 2 previous states, zero-padded if not available)
    prev_black1 = np.zeros((8, 8), dtype=np.float32)
    prev_white1 = np.zeros((8, 8), dtype=np.float32)
    prev_black2 = np.zeros((8, 8), dtype=np.float32)
    prev_white2 = np.zeros((8, 8), dtype=np.float32)
    
    if len(state.prev_states) >= 1:
        prev_board = np.array(state.prev_states[-1].board)  # most recent previous state
        prev_black1 = (prev_board == BLACK).astype(np.float32)
        prev_white1 = (prev_board == WHITE).astype(np.float32)
    
    if len(state.prev_states) >= 2:
        prev_board = np.array(state.prev_states[-2].board)  # second most recent previous state
        prev_black2 = (prev_board == BLACK).astype(np.float32)
        prev_white2 = (prev_board == WHITE).astype(np.float32)

    # Stack all channels
    stacked = np.stack([
        prev_black2, prev_white2,
        prev_black1, prev_white1,
        black_pieces, white_pieces,
        legal_moves,
        player_indicator
    ])

    return torch.from_numpy(stacked)