import torch
from model.agent import DqnAgent
from train.process_games import prepare_dataset
from train.supervised import train_supervised
from train.rl import train_rl


def main():
    states, values = prepare_dataset("../data/othello_dataset.csv")
    network = train_supervised(states, values, save_path="../output/supervised/network.pth")
    train_rl("../output/rl/model.pth", policy_net=network, checkpoint_dir="../output/rl/checkpoints", expert_data_path="../data/othello_dataset.csv")


def convert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DqnAgent(device)
    agent.load_model("../output/rl/checkpoints/checkpoint_10000.pth")
    agent.save_policy_net("../output/rl/network_10000.pth")


if __name__ == "__main__":
    main()