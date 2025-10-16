import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train.process_games import preload_expert_memory
from model.network import PolicyNetwork
from model.agent import ActorCriticAgent
from model.settings import EPS_DECAY, LR, START_EPS, END_EPS, EPISODES, TARGET_LIFESPAN
from environment.environment import EnvState


def train_rl(
        save_path: str,
        start_checkpoint_path: str = None, 
        start_episode = 1, 
        checkpoint_dir: str = None, 
        policy_net: PolicyNetwork = None,
        expert_data_path: str = None,
        start_from_policy: bool = True,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = ActorCriticAgent(device, policy_net=policy_net, actor_frozen=True)
    if start_checkpoint_path and os.path.exists(start_checkpoint_path):
        if start_from_policy:
            agent.load_policy_net(start_checkpoint_path)
        else:
            agent.load_model(start_checkpoint_path)

    if expert_data_path:
        preload_expert_memory(agent, expert_data_path)

    actor_losses = []
    critic_losses = []

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
            actor_loss, critic_loss = agent.optimize()

            if actor_loss is not None:
                actor_losses.append(actor_loss.detach().item())

            if critic_loss is not None:
                critic_losses.append(critic_loss.detach().item())
            
        if episode % TARGET_LIFESPAN == 0:
            agent.update_target()

        if episode % 1000 == 0:
            avg_loss = np.mean(critic_losses[-1000:]) if critic_losses else 0
            print(f"EP {episode:5d} | eps={eps:.3f} | avg_critic_loss={avg_loss:.4f} | mem={len(agent.memory)}")
            avg_loss = np.mean(actor_losses[-1000:]) if actor_losses else 0
            print(f"EP {episode:5d} | eps={eps:.3f} | avg_actor_loss={avg_loss:.4f} | mem={len(agent.memory)}")

        if episode % 5000 == 0 and checkpoint_dir:
            agent.save_model(os.path.join(checkpoint_dir, f"checkpoint_{episode:05d}.pth"))

    # Save to disk
    agent.save_model(os.path.join(checkpoint_dir, f"checkpoint_final.pth"))
    agent.save_policy_net(save_path)

    # Plot training results
    plt.figure(figsize=(15, 10))

    # Training loss plot
    # plt.subplot(2, 2, 2)
    plt.plot(critic_losses, alpha=0.6)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()
