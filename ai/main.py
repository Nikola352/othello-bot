from train.process_games import prepare_dataset
from train.supervised import train_supervised
from train.rl import train_rl


def main():
    states, actions = prepare_dataset("../data/othello_dataset.csv")
    network = train_supervised(states, actions, save_path="../output/supervised/network.pth")
    train_rl("../output/rl/model.pth", policy_net=network)


if __name__ == "__main__":
    main()