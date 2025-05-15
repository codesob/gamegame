import sys
import os # Added os for path operations, though Pathlib is also used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.puzzle import Puzzle
from src.solver import Solver
from pathlib import Path
from src.accuracy_recorder import AccuracyRecorder
import json

def load_puzzle_from_file(filepath):
    """Load puzzle and solution from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    puzzle_data = data.get('puzzle')
    solution = data.get('solution')
    puzzle = Puzzle.from_dict(puzzle_data)
    return puzzle, solution

def test_all_methods(filepath):
    puzzle, solution = load_puzzle_from_file(filepath)
    solver = Solver(puzzle)
    recorder = AccuracyRecorder()
    puzzle_size = puzzle.size
    puzzle_file = os.path.basename(filepath)
    
    # All solving methods to test
    methods = [
        'backtracking',
        'mrv',
        'lcv',
        'heuristics', # Changed for consistency
        'supervised_mrv_lcv'
    ]
    
    for method in methods:
        print(f"\nTesting {method} method on puzzle from {puzzle_file}")
        success, metrics = solver.solve(method=method)
        print(f"Success: {success}, Time: {metrics['time_seconds']:.3f}s, Nodes: {metrics['nodes_visited']}")
        
        # Record metrics - ML method metrics go to both files
        if method == 'supervised_mrv_lcv':
            accuracy = None  # Would need to compare with solution to calculate accuracy
            recorder.record(
                method=method,
                puzzle_size=puzzle_size,
                puzzle_file=puzzle_file,
                accuracy=accuracy,
                elapsed_time=metrics['time_seconds'],
                nodes_visited=metrics['nodes_visited']
            )
        else:
            # Non-ML methods only record nodes visited
            recorder.record(
                method=method,
                puzzle_size=puzzle_size,
                puzzle_file=puzzle_file,
                elapsed_time=metrics['time_seconds'],
                nodes_visited=metrics['nodes_visited']
            )

def main():
    puzzles_dir = Path(__file__).parent.parent / "puzzles"
    
    # Find all puzzle files
    puzzle_files = sorted(list(puzzles_dir.glob("*.json")))

    if not puzzle_files:
        print(f"No JSON puzzle files found in {puzzles_dir}. Nothing to test.")
        return

    print(f"Found {len(puzzle_files)} puzzle files in {puzzles_dir}. Starting tests...")

    for puzzle_file_path_obj in puzzle_files:
        puzzle_file_str = str(puzzle_file_path_obj)
        print(f"\n=== Testing puzzle: {puzzle_file_path_obj.name} ===")
        try:
            test_all_methods(puzzle_file_str)
        except Exception as e:
            print(f"  ERROR processing {puzzle_file_path_obj.name}: {e}")
            print(f"  Skipping this puzzle file.")
            continue # Move to the next puzzle file

if __name__ == "__main__":
    main()
