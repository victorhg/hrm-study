import torch
import torch.nn as nn

class SudokuConstraintLoss(nn.Module):
    def __init__(self, constraint_weight: float = 0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.constraint_weight = constraint_weight

    def forward(self, logits: torch.Tensor, puzzles: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(logits, puzzles)
        constraint_loss = constraint_violation_loss(logits, puzzles.squeeze(-1))
        return ce_loss + self.constraint_weight * constraint_loss

def constraint_violation_loss(logits, puzzles):
    """
    Calculates a loss based on Sudoku rule violations.
    logits: (B, 81, 10)
    puzzles: (B, 81) the original puzzles
    """
    preds_prob = torch.softmax(logits, dim=-1) # (B, 81, 10)
    
    # Create a mask for empty cells
    empty_mask = (puzzles == 0).float().unsqueeze(-1) # (B, 81, 1)
    
    # Only consider probabilities for empty cells
    preds_prob = preds_prob * empty_mask # (B, 81, 10)
    
    # Reshape to grid and exclude the "0" class
    preds_prob_grid = preds_prob[:, :, 1:].view(-1, 9, 9, 9) # (B, 9, 9, 9 digits)
    
    # Calculate sum of probabilities for each digit in each row, col, and box
    row_sum_prob = torch.sum(preds_prob_grid, dim=2) # (B, 9, 9 digits)
    col_sum_prob = torch.sum(preds_prob_grid, dim=1) # (B, 9, 9 digits)
    
    box_sum_prob = torch.nn.functional.avg_pool2d(
        preds_prob_grid.permute(0, 3, 1, 2), kernel_size=3, stride=3
    ).permute(0, 2, 3, 1).reshape(-1, 9, 9) * 9 # (B, 9, 9 digits)

    # The loss is the squared difference from 1 (each digit should appear once)
    row_loss = torch.mean(torch.square(row_sum_prob - 1.0))
    col_loss = torch.mean(torch.square(col_sum_prob - 1.0))
    box_loss = torch.mean(torch.square(box_sum_prob - 1.0))
    
    return row_loss + col_loss + box_loss
