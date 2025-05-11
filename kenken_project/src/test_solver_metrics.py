from puzzle import Puzzle
from solver import Solver
from pathlib import Path
from accuracy_nodes_recorder import AccuracyNodesRecorder
import json
import os

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
    recorder = AccuracyNodesRecorder()
    puzzle_size = puzzle.size
    puzzle_file = os.path.basename(filepath)
    
    # All solving methods to test
    methods = [
        'backtracking',
        'mrv',
        'lcv',
        'heuristic',
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
    # Test on different sized puzzles
    sizes = [3, 5, 7, 9]  # Testing on various sizes
    puzzles_dir = Path(__file__).parent.parent / "puzzles"
    
    for size in sizes:
        # Find a puzzle of this size
        pattern = f"kenken_{size}x{size}_*.json"
        matching_files = list(puzzles_dir.glob(pattern))
        if matching_files:
            # Use the first matching puzzle file
            puzzle_file = str(matching_files[0])
            print(f"\n=== Testing {size}x{size} puzzle ===")
            test_all_methods(puzzle_file)

if __name__ == "__main__":
    main()
