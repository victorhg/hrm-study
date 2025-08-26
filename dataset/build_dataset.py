
from logging import config
from typing import Optional
import os
import numpy as np
from tqdm import tqdm

import sudoku as sudoku

from argdantic import ArgParser
import pydantic

cli = ArgParser()


class DataProcessConfig(pydantic.BaseModel):
    sample_size: Optional[int] = 10000
    difficulty: Optional[int] = 5
    split: Optional[float] = 0.8
    output_dir: Optional[str] = None

def generate_sudoku_solution_field(sample_size: int, difficulty: float) -> dict:
    """Generate a Sudoku puzzle and its solution."""
    # 2. create sample_size base examples with config.difficulty
  
    dataset = dict()
    for diff in tqdm(range(1, difficulty), desc=f"Generating Sudoku puzzles up to {difficulty}"):  
        for puzzle_ID in tqdm(range(sample_size//difficulty), desc=f"Difficulty {diff}"):
            solution = np.array(sudoku.build_board())
            puzzle = np.array(sudoku.generate_puzzle(solution, difficulty=(diff/10)))
            dataset[puzzle_ID] = {
                "id": puzzle_ID,
            "puzzle": puzzle.flatten(),
            "solution": solution.flatten()
        }
    
    return dataset

def load_sudoku_data(data_path: str, max_samples: int = 1000):
    # Load the dictionary that was saved
    data = np.load(data_path, allow_pickle=True).item()
 
    puzzles = []
    solutions = []

    # Extract puzzles and solutions from the dictionary
    for i in range(min(max_samples, len(data))):
        if i in data:
            puzzles.append(data[i]["puzzle"])
            solutions.append(data[i]["solution"])

    return np.array(puzzles), np.array(solutions)


def create_dataset(config: DataProcessConfig):
    print("Generating Sudoku puzzles...")
    dataset = generate_sudoku_solution_field(config.sample_size, config.difficulty)

    # shuffle dataset
    dataset_shuffled = dataset #{k: dataset[k] for k in np.random.permutation(len(dataset))}
    keys = list(dataset_shuffled.keys())

    # split and save
    split_train = int(len(dataset) * config.split)
    train_set = {k: dataset_shuffled[k] for k in keys[:split_train]}
    test_set = {i: dataset_shuffled[k] for i, k in enumerate(keys[split_train:])}

    np.save(os.path.join(config.output_dir, "sudoku_train.npy"), train_set)
    np.save(os.path.join(config.output_dir, "sudoku_test.npy"), test_set)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    create_dataset(config)


if __name__ == "__main__":
    cli()