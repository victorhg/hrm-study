import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Tuple,Optional




def build_board():
    """
    Build the Sudoku board from the input puzzle and metadata.
    
    Args:
        input_puzzle: List of lists representing the Sudoku puzzle

    Returns:
        81-positions Array
    """
   
    while True:
        try:
            puzzle  = [[0]*9 for i in range(9)] # start with blank puzzle
            rows    = [set(range(1,10)) for i in range(9)] # set of available
            columns = [set(range(1,10)) for i in range(9)] #   numbers for each
            boxes   = [set(range(1,10)) for i in range(9)] #   row, column and square
            for r in range(9):
                for c in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[r].intersection(columns[c]).intersection(boxes[(r//3)*3 + c//3])
                    choice  = random.choice(list(choices))

                    puzzle[r][c] = choice

                    rows[r].discard(choice)
                    columns[c].discard(choice)
                    boxes[(r//3)*3 + c//3].discard(choice)
                    # success! every cell is filled.
            return puzzle

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass



def generate_puzzle(board, difficulty=0.5):
    copy = board.copy()
    """
    Generate a Sudoku puzzle from the completed board by removing numbers.

    Args:
        board: 2D list representing the completed Sudoku board
        difficulty: Float between 0 and 1 representing the difficulty level

    Returns:
        2D list representing the Sudoku puzzle
    """
    puzzle = [row[:] for row in copy]  # make a copy of the board
    num_remove = int(difficulty * 81)  # number of cells to remove

    for _ in range(num_remove):
        r, c = random.randint(0, 8), random.randint(0, 8)
        while puzzle[r][c] == 0:
            r, c = random.randint(0, 8), random.randint(0, 8)
        puzzle[r][c] = 0

    return puzzle



def display_puzzle_pair(input_puzzle, solution_puzzle):
    """Display input and solution side by side"""
    # Convert to grids
    print("\nINPUT (_ = blank)        SOLUTION")
    print("  0 1 2 3 4 5 6 7 8      0 1 2 3 4 5 6 7 8")
    print("  -----------------      -----------------")

    for i in range(9):
        # Input row
        input_row = f"{i}|"

        for val in input_puzzle[i]:
            if val == 0:
                input_row += " _"
            else:
                input_row += f" {val}"
        
        # Solution row
        solution_row = f"    {i}|"
        for val in solution_puzzle[i]:
            solution_row += f" {val}"
            
        print(input_row + solution_row)
    
    # Count filled vs blank cells
    filled_cells = np.sum(np.array(input_puzzle) != 0)
    blank_cells = 81 - filled_cells
    print(f"\nStatistics: {filled_cells} filled, {blank_cells} blank cells")




class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray):
        self.puzzles = puzzles
        self.solutions = solutions

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return {
            'puzzle': torch.tensor(self.puzzles[idx], dtype=torch.float32),
            'solution': torch.tensor(self.solutions[idx], dtype=torch.long)
        }
    

#  generate embeddings from sudoku game




class SudokuAdapter(nn.Module):
    """""
    Converts sudoku puzzles to embeddings and HRM iputs to valid moves

    Expects 9x9 sudoku grids 
    """
    def __init__(self, 
                 hidden_dim: int = 256,
                 hrm_input_dim: int = 512,
                 hrm_output_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hrm_input_dim = hrm_input_dim
        self.hrm_output_dim = hrm_output_dim

        self.digit_embedding = nn.Embedding(10, self.hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(81, self.hidden_dim))

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hrm_input_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hrm_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 10) 
        )

        self.constraint_attention = nn.MultiheadAttention(
            embed_dim=hrm_input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.encoder, self.decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def encode_puzzle(self, puzzle: torch.Tensor) -> torch.Tensor:
        """
        Encodes a Sudoku puzzle into a latent representation.
        Args:
            puzzle (torch.Tensor): A tensor representing the Sudoku puzzle as (batch_size, 81)
        """
        batch_size = puzzle.size(0)

        # Get digit embeddings: (batch_size, 81, hidden_dim)
        digit_embeds = self.digit_embedding(puzzle)

        # Get position embeddings: (batch_size, 81, hidden_dim)
        position_embeds = self.position_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate along the feature dimension: (batch_size, 81, hidden_dim * 2)
        combined = torch.cat([digit_embeds, position_embeds], dim=-1)

        # Encode: (batch_size, 81, hrm_input_dim)
        encoded = self.encoder(combined)

        return encoded

    def decode_solution(self, 
                        hrm_output: torch.Tensor,
                        original_puzzle: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Returns:
            logits: (B, 81, 10)
            predictions: (B, 81) with original givens preserved
        """
        batch_size = hrm_output.size(0)

        # Expect hrm_output shape (B, 81, hrm_input_dim)
        assert hrm_output.dim() == 3 and hrm_output.size(1) == 81, f"Unexpected hrm_output shape {hrm_output.shape}"

        logits = self.decoder(hrm_output)          # (B, 81, 10)
        assert logits.size(-1) == 10, f"Decoder last dim must be 10, got {logits.shape}"

        pred_digits = torch.argmax(logits, dim=-1) # (B, 81)
        mask = (original_puzzle == 0)              # (B, 81)
        final_predictions = pred_digits * mask + original_puzzle * (~mask)  # preserve givens
        return logits, final_predictions
