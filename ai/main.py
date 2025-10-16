from matplotlib import pyplot as plt
from train.value_pretrain import pretrain_value_network
from train.process_games import prepare_dataset
from train.policy_pretrain import pretrain_policy_network
from train.rl import train_rl


def main():
    states, actions, values = prepare_dataset("../data/othello_dataset.csv")

    policy_net = pretrain_policy_network(states, actions, save_path="../output/supervised/policy/policy_net.pth")['network']
    value_net = pretrain_value_network(states, values, save_path="../output/supervised/value/value_net.pth")['network']

    train_rl("../output/rl/model.pth", policy_net=policy_net, value_net=value_net, checkpoint_dir="../output/rl/checkpoints", expert_data_path="../data/othello_dataset.csv")

if __name__ == "__main__":
    main()