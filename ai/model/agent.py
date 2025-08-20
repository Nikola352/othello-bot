import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from environment.environment import EnvState
from model.network import DeepQNetwork, state_to_tensor
from model.replay_memory import ReplayMemory
from model.settings import *


class DqnAgent(object):
    def __init__(self, device, policy_net=DeepQNetwork()):
        self.device = device

        self.policy_net = policy_net.to(device)

        self.target_net = DeepQNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(REPLAY_CAP)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=RL_LR)

    def select_action(self, state: EnvState, eps: float):
        """Select an action using ε-greedy strategy"""
        legal_moves = state.get_available_actions()
        if not legal_moves:
            return None
        
        if random.random() < eps:
            return random.choice(legal_moves)
        
        state_tensor = state_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(state_tensor).cpu().numpy().flatten()

        # Mask illegal moves
        mask = np.full(8*8, -np.inf, dtype=np.float32)
        for x, y in legal_moves:
            mask[x*8 + y] = q_vals[x*8 + y]

        max_idx = np.argmax(mask)
        return (max_idx // 8, max_idx % 8)
    
    def select_optimal_action(self, state: EnvState) -> tuple[int, int]:
        """Select an action using the optimal strategy"""
        return self.select_action(state, eps=0.0)

    def add_to_memory(self, state, action, reward, next_state):
        self.memory.add((
            state_to_tensor(state),
            action[0]*8 + action[1],
            reward,
            state_to_tensor(next_state),
            next_state.is_final(),
        ))

    def optimize(self):
        if len(self.memory) < EXP_REPLAY_BATCH_SIZE:
            return

        transitions = self.memory.sample(EXP_REPLAY_BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        state_batch = torch.stack(states).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.stack(next_states).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: Select action using policy net and evaluate it using target net
        next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
        next_state_values = self.target_net(next_state_batch).gather(1, next_actions)

        # Adjust sign if it's White's turn in next_state
        next_player_is_white = (next_state_batch[:, 3, 0, 0] == 0).float().unsqueeze(1)
        sign = torch.where(next_player_is_white == 1, -1.0, 1.0)

        next_state_values = next_state_values * sign

        expected_q_values = reward_batch + (1-done_batch) * GAMMA * next_state_values
        
        loss = F.smooth_l1_loss(input=q_values, target=expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str):
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @classmethod
    def load_for_inference(cls, model_path: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent = cls(device)
        agent.load_model(model_path)
        agent.policy_net.eval()
        return agent

    def save_policy_net(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
