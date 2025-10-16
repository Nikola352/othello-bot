import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from environment.constants import BLACK, WHITE
from environment.environment import EnvState


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Policy head - maps spatial features to move probabilities
        self.policy_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        self.policy_fc = nn.Linear(in_features=2*8*8, out_features=64)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Policy head
        x = F.relu(self.policy_conv(x))
        x = x.view(x.size(0), -1)
        logits = self.policy_fc(x)
        
        return logits
    
    def predict_probs(self, x, legal_moves_mask=None):
        logits = self.forward(x)
        
        if legal_moves_mask is not None:
            if isinstance(legal_moves_mask, np.ndarray):
                legal_moves_mask = torch.tensor(legal_moves_mask, dtype=torch.float32, device=logits.device)
            elif legal_moves_mask.device != logits.device:
                legal_moves_mask = legal_moves_mask.to(logits.device)
        
        logits = logits + (legal_moves_mask - 1) * 1e9
        
        probs = F.softmax(logits, dim=1)
        return probs


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        
        # Value head - reduces to single scalar
        self.value_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.value_fc1 = nn.Linear(in_features=8*8, out_features=256)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)
        
    def forward(self, x):
        # Feature extraction with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        # Value head
        x = F.relu(self.value_conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(x))
        
        return value


def state_to_tensor(state: EnvState) -> torch.Tensor:
    """
    Convert game state to neural network input tensor.
    
    Input channels (10 total):
    - 0-1: Previous state t-2 (black, white)
    - 2-3: Previous state t-1 (black, white)
    - 4-5: Current state (black, white)
    - 6: Legal moves mask
    - 7-8: Stable pieces (black, white)
    - 9: Current player indicator
    """
    board = state.board

    # Current state channels
    black_pieces = (board == BLACK).astype(np.float32)
    white_pieces = (board == WHITE).astype(np.float32)

    legal_moves = state.get_legal_moves_mask()

    player_indicator = np.ones((8,8), dtype=np.float32) if state.turn == BLACK else np.zeros((8,8), dtype=np.float32)

    black_stable, white_stable = state.get_stability_mask()

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
        black_stable, white_stable,
        player_indicator
    ])

    return torch.from_numpy(stacked)
