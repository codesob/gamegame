from accuracy_recorder import AccuracyRecorder
from puzzle import Puzzle
from solver import Solver
import json
import os
from pathlib import Path

# Load a puzzle from the puzzles directory
def load_puzzle(size=5):
    puzzles_dir = Path(__file__).parent.parent / "puzzles"
    # Find a puzzle file matching the size
    puzzle_files = [f for f in os.listdir(puzzles_dir) if f.startswith(f"kenken_{size}x{size}")]
    if not puzzle_files:
        raise ValueError(f"No {size}x{size} puzzle found")
    
    # Use the most recent puzzle
    puzzle_file = sorted(puzzle_files)[-1]
    puzzle_path = puzzles_dir / puzzle_file
    
    with open(puzzle_path, 'r') as f:
        data = json.load(f)
        puzzle_data = data.get('puzzle')
        solution = data.get('solution')
    
    puzzle = Puzzle.from_dict(puzzle_data)
    return puzzle, solution, puzzle_file

def test_all_methods():
    # Load a 5x5 puzzle
    puzzle, solution, puzzle_file = load_puzzle(5)
    print(f"Using puzzle: {puzzle_file}")
    
    # Create solver instance
    solver = Solver(puzzle, known_solution=solution)
    
    # List of methods to test
    methods = [
        'backtracking',
        'mrv',
        'lcv',
        'heuristics'
    ]
    
    # Try each method
    for method in methods:
        print(f"\nSolving with {method}...")
        success, metrics = solver.solve(method=method)
        print(f"Success: {success}")
        print(f"Time: {metrics['time_seconds']:.3f} seconds")
        print(f"Nodes visited: {metrics['nodes_visited']}")
        
        # Reset puzzle for next method
        puzzle.reset_grid()

if __name__ == "__main__":
    test_all_methods()
