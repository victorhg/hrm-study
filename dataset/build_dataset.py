
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
    for puzzle_ID in tqdm(range(sample_size)):
        solution = np.array(sudoku.build_board())
        puzzle = np.array(sudoku.generate_puzzle(solution, difficulty=(difficulty/10)))
        dataset[puzzle_ID] = {
            "id": puzzle_ID,
            "puzzle": puzzle.flatten(),
            "solution": solution.flatten()
        }
    
    return dataset


def create_dataset(config: DataProcessConfig):
    print("Generating Sudoku puzzles...")
    dataset = generate_sudoku_solution_field(config.sample_size, config.difficulty)

    # shuffle dataset
    dataset_shuffled = dataset #{k: dataset[k] for k in np.random.permutation(len(dataset))}

    # split and save
    split_train = int(len(dataset) * config.split)
    train_set = {k: dataset_shuffled[k] for k in list(dataset_shuffled)[:split_train]}
    test_set = {k: dataset_shuffled[k] for k in list(dataset_shuffled)[split_train:]}

    np.save(os.path.join(config.output_dir, "sudoku_train.npy"), train_set)
    np.save(os.path.join(config.output_dir, "sudoku_test.npy"), test_set)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    create_dataset(config)


if __name__ == "__main__":
    cli()