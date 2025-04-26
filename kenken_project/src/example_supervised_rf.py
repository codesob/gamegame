import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generator import generate_kenken
from puzzle import Puzzle
from src.supervised_solver import SupervisedSolver

def main():
    size = 4
    puzzle_data, solution = generate_kenken(size)

    print("Generated puzzle:")
    for cage in puzzle_data.get("cages", []):
        if 'target' in cage:
            cage['value'] = cage.pop('target')
    puzzle = Puzzle.from_dict(puzzle_data)

    for row in puzzle.get_grid_copy():
        print(row)

    solver = SupervisedSolver(puzzle, solution_grid=solution)

    print("Training Random Forest solver...")
    solver.train_random_forest()

    print("Solving puzzle with Random Forest solver...")
    solved_grid = solver.solve()

    print("Solved puzzle:")
    for row in solved_grid:
        print(row)

if __name__ == "__main__":
    main()
