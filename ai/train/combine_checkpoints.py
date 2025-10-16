import torch
from pathlib import Path
from typing import Optional, Tuple, Union
from model.network import PolicyNetwork, ValueNetwork
from model.settings import ACTOR_LR, CRITIC_LR


def combine_pretrained_checkpoints(
    policy_checkpoint_path: Union[str, Path],
    value_checkpoint_path: Union[str, Path],
    output_path: Optional[str] = None,
    device: str = 'cpu',
    initialize_optimizers: bool = True
) -> Tuple[dict, str]:
    """
    Combine best model checkpoints from policy and value network pretraining
    into a single checkpoint suitable for ActorCriticAgent initialization.
    
    Args:
        policy_checkpoint_path: Path to best policy network checkpoint
                               (from pretrain_policy supervised.py)
        value_checkpoint_path: Path to best value network checkpoint
                              (from pretrain_value_network supervised.py)
        output_path: Path to save combined checkpoint. If None, saves to
                    './pretrained_checkpoints/combined_pretrained.pt'
        device: Device to load checkpoints on ('cpu', 'cuda', etc.)
        initialize_optimizers: Whether to initialize fresh optimizer states
                              (recommended for continued training)
    
    Returns:
        Tuple of (combined_checkpoint_dict, output_path)
    
    Example:
        >>> checkpoint, path = combine_pretrained_checkpoints(
        ...     'runs/policy_run_20250101_120000/checkpoints/best_model_state_dict.pt',
        ...     'runs/value_run_20250101_121000/checkpoints/best_value_model_state_dict.pt',
        ...     output_path='./models/pretrained_agent.pt'
        ... )
        >>> agent = ActorCriticAgent(device='cuda')
        >>> agent.load_model(path)
    """
    
    # Set default output path
    if output_path is None:
        output_path = './pretrained_checkpoints/combined_pretrained.pt'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Combining Pretrained Policy and Value Network Checkpoints")
    print("=" * 70)
    
    # Load policy checkpoint
    print(f"\nLoading policy network from: {policy_checkpoint_path}")
    policy_checkpoint = torch.load(policy_checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and state_dict-only formats
    if isinstance(policy_checkpoint, dict) and 'model_state_dict' in policy_checkpoint:
        policy_state_dict = policy_checkpoint['model_state_dict']
        policy_opt_state = policy_checkpoint.get('optimizer_state_dict', None)
        print(f"  ✓ Loaded full checkpoint format")
        print(f"    - Best epoch: {policy_checkpoint.get('epoch', 'N/A')}")
        print(f"    - Val accuracy: {policy_checkpoint.get('val_accuracy', 'N/A'):.2%}")
    else:
        policy_state_dict = policy_checkpoint
        policy_opt_state = None
        print(f"  ✓ Loaded state dict only format")
    
    # Load value checkpoint
    print(f"\nLoading value network from: {value_checkpoint_path}")
    value_checkpoint = torch.load(value_checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and state_dict-only formats
    if isinstance(value_checkpoint, dict) and 'model_state_dict' in value_checkpoint:
        value_state_dict = value_checkpoint['model_state_dict']
        value_opt_state = value_checkpoint.get('optimizer_state_dict', None)
        print(f"  ✓ Loaded full checkpoint format")
        print(f"    - Best epoch: {value_checkpoint.get('epoch', 'N/A')}")
        print(f"    - Val R²: {value_checkpoint.get('val_r2', 'N/A'):.4f}")
    else:
        value_state_dict = value_checkpoint
        value_opt_state = None
        print(f"  ✓ Loaded state dict only format")
    
    # Validate state dicts by loading into fresh networks
    print("\nValidating checkpoint compatibility...")
    try:
        policy_net = PolicyNetwork().to(device)
        policy_net.load_state_dict(policy_state_dict)
        print("  ✓ Policy network state dict validated")
    except RuntimeError as e:
        raise RuntimeError(f"Policy checkpoint incompatible: {e}")
    
    try:
        value_net = ValueNetwork().to(device)
        value_net.load_state_dict(value_state_dict)
        print("  ✓ Value network state dict validated")
    except RuntimeError as e:
        raise RuntimeError(f"Value checkpoint incompatible: {e}")
    
    # Create combined checkpoint for ActorCriticAgent
    print("\nCreating combined checkpoint...")
    
    if initialize_optimizers:
        # Initialize fresh optimizer states
        print("  - Initializing fresh optimizer states")
        policy_opt = torch.optim.Adam(policy_net.parameters(), lr=ACTOR_LR)
        value_opt = torch.optim.Adam(value_net.parameters(), lr=CRITIC_LR)
        policy_opt_state_dict = policy_opt.state_dict()
        value_opt_state_dict = value_opt.state_dict()
    else:
        # Use optimizer states from checkpoints if available
        policy_opt_state_dict = policy_opt_state if policy_opt_state else None
        value_opt_state_dict = value_opt_state if value_opt_state else None
        
        if policy_opt_state_dict is None or value_opt_state_dict is None:
            print("  ⚠ Warning: Optimizer states not found in checkpoints, initializing fresh")
            policy_opt = torch.optim.Adam(policy_net.parameters(), lr=ACTOR_LR)
            value_opt = torch.optim.Adam(value_net.parameters(), lr=CRITIC_LR)
            policy_opt_state_dict = policy_opt.state_dict()
            value_opt_state_dict = value_opt.state_dict()
    
    combined_checkpoint = {
        'policy_net': policy_state_dict,
        'value_net': value_state_dict,
        'policy_opt': policy_opt_state_dict,
        'value_opt': value_opt_state_dict,
    }
    
    # Save combined checkpoint
    print(f"\nSaving combined checkpoint to: {output_path}")
    torch.save(combined_checkpoint, output_path)
    print(f"  ✓ Successfully saved combined checkpoint")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Policy Network:  {policy_checkpoint_path}")
    print(f"Value Network:   {value_checkpoint_path}")
    print(f"Output:          {output_path}")
    print(f"Total Size:      {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("\nUsage:")
    print("  agent = ActorCriticAgent(device='cuda')")
    print(f"  agent.load_model('{output_path}')")
    print("=" * 70)
    
    return combined_checkpoint, str(output_path)
