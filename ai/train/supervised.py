import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from model.network import DeepQNetwork
from model.settings import BATCH_SIZE, EPOCHS, LR


class GameDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = torch.tensor(self.actions[idx], dtype=torch.int64)
        return state, action


def train_supervised(states, actions, save_path: str = None, start_model_path: str = None) -> DeepQNetwork:
    """
    Pretrain agent's policy network using (state -> action) pairs
    """
    dataset = GameDataset(states, actions)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = DeepQNetwork().to(device)
    if start_model_path and os.path.exists(start_model_path):
        state_dict = torch.load(start_model_path, map_location=device, weights_only=True)
        network.load_state_dict(state_dict)
    network.train()

    optimizer = optim.Adam(network.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for state, action in loader:
            state = state.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            q_values = network(state)
            loss = criterion(q_values, action)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss/len(loader)
        losses.append(avg_loss)
        print(f"[Pretrain] Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Save to disk
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict, save_path)

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), losses, 'b-', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return network
