import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train.process_games import preload_expert_memory
from model.network import DeepQNetwork
from model.agent import DqnAgent
from model.settings import EPS_DECAY, LR, START_EPS, END_EPS, EPISODES, TARGET_LIFESPAN
from environment.environment import EnvState


def train_rl(
        save_path: str,
        start_checkpoint_path: str = None, 
        start_episode = 1, 
        checkpoint_dir: str = None, 
        policy_net: DeepQNetwork = None,
        expert_data_path: str = None,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DqnAgent(device, policy_net=policy_net)
    if start_checkpoint_path and os.path.exists(start_checkpoint_path):
        agent.load_model(start_checkpoint_path)

    if expert_data_path:
        preload_expert_memory(agent, expert_data_path)

    losses = []

    for episode in range(start_episode, EPISODES+1):
        eps = END_EPS + (START_EPS - END_EPS) * np.exp(-1.0 * episode / EPS_DECAY)
        total_reward = 0.0

        state = EnvState()
        while not state.is_final():
            action = agent.select_action(state, eps)
            if action is None:
                state.skip_move()
                continue

            prev_state = state.copy()
            reward = state.act(action)
            total_reward += reward
            
            agent.add_to_memory(prev_state, action, reward, state)
            loss = agent.optimize()

            if loss is not None:
                losses.append(loss.detach().item())
            
        if episode % TARGET_LIFESPAN == 0:
            agent.update_target()

        if episode % 1000 == 0:
            avg_loss = np.mean(losses[-1000:]) if losses else 0
            print(f"EP {episode:5d} | eps={eps:.3f} | avg_loss={avg_loss:.4f} | mem={len(agent.memory)}")

        if episode % 5000 == 0 and checkpoint_dir:
            agent.save_model(os.path.join(checkpoint_dir, f"checkpoint_{episode:05d}.pth"))

    # Save to disk
    agent.save_model(os.path.join(checkpoint_dir, f"checkpoint_final.pth"))
    agent.save_policy_net(save_path)

    # Plot training results
    plt.figure(figsize=(15, 10))

    # Training loss plot
    # plt.subplot(2, 2, 2)
    plt.plot(losses, alpha=0.6)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()
