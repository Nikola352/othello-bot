import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from datetime import datetime
from model.network import PolicyNetwork
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


class TrainingMetrics:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float, lr: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch
    
    def save_metrics(self):
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch
        }
        
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    
    def plot_metrics(self):
        """Generate and save all training plots"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch + 1, color='g', linestyle='--', 
                          label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[0, 0].set_title('Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].axvline(x=self.best_epoch + 1, color='g', linestyle='--', 
                          label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[0, 1].set_title('Accuracy Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate schedule
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Overfitting analysis (train vs val gap)
        loss_gap = [t - v for t, v in zip(self.train_losses, self.val_losses)]
        acc_gap = [v - t for t, v in zip(self.train_accuracies, self.val_accuracies)]
        
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(epochs, loss_gap, 'b-', label='Loss Gap (Train - Val)', linewidth=2)
        line2 = ax4_twin.plot(epochs, acc_gap, 'r-', label='Accuracy Gap (Val - Train)', linewidth=2)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Gap', color='b')
        ax4_twin.set_ylabel('Accuracy Gap (%)', color='r')
        ax4.tick_params(axis='y', labelcolor='b')
        ax4_twin.tick_params(axis='y', labelcolor='r')
        ax4.set_title('Overfitting Analysis')
        ax4.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual plots for better visibility
        self._save_individual_plots(epochs)
    
    def _save_individual_plots(self, epochs):
        """Save individual metric plots"""
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.axvline(x=self.best_epoch + 1, color='g', linestyle='--', 
                   label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        plt.title('Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.axvline(x=self.best_epoch + 1, color='g', linestyle='--', 
                   label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        plt.title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / 'accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def evaluate_model(network: nn.Module, loader: DataLoader, 
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    network.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for state, action in loader:
            state = state.to(device)
            action = action.to(device)
            
            q_values = network(state)
            loss = criterion(q_values, action)
            accuracy = compute_accuracy(q_values, action)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def save_checkpoint(network: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int, metrics: TrainingMetrics, 
                   checkpoint_path: Path, is_best: bool = False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': metrics.train_losses[-1],
        'val_loss': metrics.val_losses[-1],
        'train_accuracy': metrics.train_accuracies[-1],
        'val_accuracy': metrics.val_accuracies[-1],
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)
        torch.save(network.state_dict(), checkpoint_path.parent / 'best_model_state_dict.pt')


def pretrain_policy_network(states, actions, 
                    save_path: Optional[str] = None,
                    train_split: float = 0.7,
                    val_split: float = 0.15,
                    test_split: float = 0.15,
                    checkpoint_interval: int = 5,
                    early_stopping_patience: int = 10) -> Dict:
    """
    Pretrain agent's policy network using (state -> action) pairs
    
    Args:
        states: Input states
        actions: Target actions
        save_path: Path to save final model
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        checkpoint_interval: Save checkpoint every N epochs
        early_stopping_patience: Stop if no improvement for N epochs
    
    Returns:
        Dictionary containing trained network and metrics
    """

    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Train, val, and test splits must sum to 1.0"
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_path).parent if save_path else Path("./training_output")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create and split dataset
    full_dataset = GameDataset(states, actions)
    total_size = len(full_dataset)
    
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset splits:")
    print(f"  Training:   {train_size} samples ({train_split*100:.1f}%)")
    print(f"  Validation: {val_size} samples ({val_split*100:.1f}%)")
    print(f"  Test:       {test_size} samples ({test_split*100:.1f}%)")
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize network
    network = PolicyNetwork().to(device)
    start_epoch = 0
    
    network.train()
    
    # Setup training
    optimizer = optim.Adam(network.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    metrics = TrainingMetrics(run_dir)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("=" * 70)
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        # Training phase
        network.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_train_batches = 0
        
        for state, action in train_loader:
            state = state.to(device)
            action = action.to(device)
            
            optimizer.zero_grad()
            q_values = network(state)
            loss = criterion(q_values, action)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_accuracy += compute_accuracy(q_values, action)
            num_train_batches += 1
        
        avg_train_loss = train_loss / num_train_batches
        avg_train_accuracy = train_accuracy / num_train_batches
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(network, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics.update(epoch, avg_train_loss, val_loss, 
                      avg_train_accuracy, val_accuracy, current_lr)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {avg_train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.2e}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  → New best model! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        if (epoch + 1) % checkpoint_interval == 0 or is_best:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(network, optimizer, epoch, metrics, checkpoint_path, is_best)
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {early_stopping_patience} consecutive epochs")
            break
    
    print("=" * 70)
    print("\nTraining completed!")
    
    # Final test evaluation
    best_model_path = checkpoint_dir / 'best_model.pt'
    if best_model_path.exists():
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate_model(network, test_loader, criterion, device)
    
    print(f"\nFinal Results:")
    print(f"  Best Validation Loss:     {metrics.best_val_loss:.4f} (Epoch {metrics.best_epoch + 1})")
    print(f"  Best Validation Accuracy: {metrics.best_val_accuracy:.2f}%")
    print(f"  Test Loss:                {test_loss:.4f}")
    print(f"  Test Accuracy:            {test_accuracy:.2f}%")
    
    # Save final model
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), save_path)
        print(f"\nFinal model saved to: {save_path}")
    
    # Save metrics and generate plots
    print("\nGenerating training visualizations...")
    metrics.save_metrics()
    metrics.plot_metrics()
    
    summary = {
        'total_epochs': epoch + 1,
        'best_epoch': metrics.best_epoch + 1,
        'best_val_loss': metrics.best_val_loss,
        'best_val_accuracy': metrics.best_val_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'dataset_splits': {
            'train': train_size,
            'val': val_size,
            'test': test_size
        },
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'initial_lr': LR,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        }
    }
    
    with open(run_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to: {run_dir}")
    print(f"  - Metrics: {run_dir / 'metrics.json'}")
    print(f"  - Plots: {run_dir / 'training_metrics.png'}")
    print(f"  - Best model: {checkpoint_dir / 'best_model_state_dict.pt'}")
    print(f"  - Summary: {run_dir / 'training_summary.json'}")
    
    return {
        'network': network,
        'metrics': metrics,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'run_dir': run_dir
    }
