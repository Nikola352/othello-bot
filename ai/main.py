from matplotlib import pyplot as plt
import torch
from ai.train.combine_checkpoints import combine_pretrained_checkpoints
from train.value_pretrain import pretrain_value_network
from train.process_games import prepare_dataset
from train.policy_pretrain import pretrain_policy_network
from train.rl import train_rl


def main():
    states, actions, values = prepare_dataset("../data/othello_dataset.csv")

    policy_result = pretrain_policy_network(states, actions, save_path="../output/supervised/policy")
    value_result = pretrain_value_network(states, values, save_path="../output/supervised/value")

    _, checkpoint_path = combine_pretrained_checkpoints(
        policy_checkpoint_path=policy_result['run_dir'] / "checkpoints" / "best_model.pt",
        value_checkpoint_path=value_result['run_dir'] / "checkpoints" / "best_model.pt",
        output_path='../output/pretrained_agent.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        initialize_optimizers=True
    )

    train_rl("../output/rl/model.pth", start_checkpoint_path=checkpoint_path, checkpoint_dir="../output/rl/checkpoints", expert_data_path="../data/othello_dataset.csv")


if __name__ == "__main__":
    main()