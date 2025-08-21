import random
import numpy as np


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
