import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from environment.constants import BLACK
from model.network import PolicyNetwork, ValueNetwork, state_to_tensor
from model.replay_memory import ReplayMemory
from model.settings import GAMMA, ACTOR_LR, REPLAY_CAP, EXP_REPLAY_BATCH_SIZE


class ActorCriticAgent:
    def __init__(self, device, policy_net=None, value_net=None, actor_frozen=False):
        self.device = device

        self.policy_net = (policy_net or PolicyNetwork()).to(device)
        self.value_net = (value_net or ValueNetwork()).to(device)

        self.memory = ReplayMemory(REPLAY_CAP)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=ACTOR_LR)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=ACTOR_LR)

        self.actor_frozen = actor_frozen

    def select_action(self, state, eps=0.0):
        """Select an action using ε-greedy strategy"""
        legal_moves = state.get_available_actions()
        if not legal_moves:
            return None

        if random.random() < eps:
            return random.choice(legal_moves)

        state_tensor = state_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net.predict_probs(state_tensor, legal_moves_mask=state.get_legal_moves_mask())

        action_index = np.random.choice(np.arange(8 * 8), p=probs)
        return (action_index // 8, action_index % 8)

    def add_to_memory(self, state, action, reward, next_state):
        self.memory.add((
            state_to_tensor(state),
            action[0]*8 + action[1],
            reward,
            state_to_tensor(next_state),
            next_state.is_final(),
            1 if state.turn == BLACK else -1,
        ))

    def optimize(self):
        if len(self.memory) < EXP_REPLAY_BATCH_SIZE:
            return None

        transitions = self.memory.sample(EXP_REPLAY_BATCH_SIZE)
        states, actions, rewards, next_states, dones, turns = zip(*transitions)

        state_batch = torch.stack(states).to(self.device)
        next_state_batch = torch.stack(next_states).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        turn_batch = torch.tensor(turns, dtype=torch.float32, device=self.device).unsqueeze(1)

        # --- Critic ---
        values = self.value_net(state_batch)
        with torch.no_grad():
            next_state_values = self.value_net(next_state_batch)
            target_values = reward_batch + (1-done_batch) * (-turn_batch) * GAMMA * next_state_values
        advantages = target_values - values

        critic_loss = advantages.pow(2).mean()

        # --- Actor ---
        logits = self.policy_net(state_batch)
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, action_batch.unsqueeze(1))

        actor_loss = -(selected_log_probs * advantages.detach()).mean()

        # Backprop
        if not self.actor_frozen:
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()

        return actor_loss, critic_loss

    def save_model(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'value_opt': self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_opt'])
        self.value_optimizer.load_state_dict(checkpoint['value_opt'])

    def load_policy_net(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_policy_net(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
