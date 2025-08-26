import os
import hrm
from hrm.loss import SudokuConstraintLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hrm import HierarchicalReasoningModel, ModelConfig


class HRMTrainer:
    
    def __init__(self, 
                 model: HierarchicalReasoningModel, 
                 config=None, device=None):
        
        self.model = model
        self.config = config or ModelConfig()
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.model.to(self.device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()#SudokuConstraintLoss(constraint_weight=0.7)
        
        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        self.patience = 15
        
        # Create results directory
        os.makedirs('results', exist_ok=True)

    def _accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        # logits: (B, 81, C) targets: (B,81)
        print(logits.shape, targets.shape)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float().sum().item()
        total = targets.numel()
        return correct / total

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        
        for batch in loader:
            puzzles, solutions = batch["puzzle"], batch["solution"]
            puzzles = puzzles.to(self.device)
            solutions = solutions.to(self.device).long()
            
            if train:
                self.optimizer.zero_grad()

            model_output = self.model(puzzles)
             
            # Calculate loss based on output dimensions
            batch_size, seq_len, num_classes = model_output.shape

            output_flat = model_output.view(-1, num_classes)  # (B*81, C)
            solutions_flat = solutions.view(-1)               # (B*81,)
            
            loss = self.criterion(output_flat, solutions_flat)
            constraint_loss = hrm.constraint_violation_loss(model_output, puzzles.squeeze(-1))
            total_loss = loss + 0.5 * constraint_loss 
            

            preds = output_flat.argmax(dim=-1)  # (B*81,)
            acc = (preds == solutions_flat).float().mean().item()  # Both same shape now

            if train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            epoch_loss += total_loss.item()
            epoch_acc += acc
            total_batches += 1
                
        return epoch_loss / max(1, total_batches), epoch_acc / max(1, total_batches)

    def train(self, train_dataset, val_dataset=None):
        epochs = self.config.max_epochs
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size) if val_dataset is not None else None

        val_frequency = 2

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)

            # warmup learning rate
            lr = self.config.learning_rate * min(1.0, epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            if val_loader is not None and epoch % val_frequency == 0:
                with torch.no_grad():
                    val_loss, val_acc = self._run_epoch(val_loader, train=False)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                self.scheduler.step(val_loss)
            else:
                val_loss, val_acc = train_loss, train_acc  # fallback

            improved = val_loss < self.best_val_loss - 1e-5
            if improved:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} | LR {self.optimizer.param_groups[0]['lr']:.2e}")

            if self.epochs_without_improvement >= self.patience:
                print("Early stopping triggered.")
                break

        # Save best checkpoint
        if self.best_model_state is not None:
            torch.save(self.best_model_state, 'results/best_model.pt')
            print(f"Best model (val_loss={self.best_model_state['val_loss']:.4f}) saved to results/best_model.pt")

    def evaluate(self, dataset):
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        with torch.no_grad():
            loss, acc = self._run_epoch(loader, train=False)
        print(f"Eval Loss {loss:.4f} Acc {acc:.4f}")
        return loss, acc