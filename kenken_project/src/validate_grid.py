import json
from src.puzzle import Puzzle
from src.solver import Solver

def main():
    # The given 9x9 grid to validate
    grid = [
        [6, 9, 1, 2, 5, 8, 4, 3, 7],
        [5, 3, 8, 6, 2, 1, 9, 7, 4],
        [9, 6, 2, 1, 4, 3, 7, 8, 5],
        [8, 7, 5, 4, 6, 2, 3, 9, 1],
        [3, 1, 7, 8, 9, 5, 6, 4, 2],
        [4, 2, 9, 5, 3, 7, 1, 6, 8],
        [1, 8, 3, 9, 7, 4, 2, 5, 6],
        [2, 4, 6, 7, 8, 9, 5, 1, 3],
        [7, 5, 4, 3, 1, 6, 8, 2, 9]
    ]

    # Load puzzle from JSON file
    puzzle_file = "puzzles/kenken_9x9_20250426_215301.json"
    with open(puzzle_file, 'r') as f:
        data = json.load(f)
    puzzle_data = data.get("puzzle")
    if not puzzle_data:
        print("Invalid puzzle JSON format: missing 'puzzle' key.")
        return

    puzzle = Puzzle.from_dict(puzzle_data)

    # Create solver instance
    solver = Solver(puzzle)

    # Validate the given grid with detailed feedback
    is_valid, errors = solver.validate_solution_detailed(grid)

    if is_valid:
        print("The given grid is a valid solution for the puzzle.")
    else:
        print("The given grid is NOT a valid solution for the puzzle.")
        print("Validation errors:")
        for error in errors:
            print(f" - {error}")

if __name__ == "__main__":
    main()
