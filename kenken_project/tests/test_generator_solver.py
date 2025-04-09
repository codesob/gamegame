from src.generator import generate_kenken
from src.solver import Solver
from src.visualizer import KenKenVisualizer
from src.puzzle import Puzzle
from src.cage import Cage
import matplotlib.pyplot as plt
import time

def map_operation(op):
    """Map operation to correct format."""
    op_map = {
        '+': '+', '-': '-', '*': '*', '/': '/', '=': '='
    }
    if op in op_map:
        return op_map[op]
    raise ValueError(f"Invalid operation: {op}")

def create_puzzle_from_dict(puzzle_dict):
    """Convert dictionary representation to Puzzle object."""
    size = puzzle_dict['size']
    cages = []
    for cage_data in puzzle_dict['cages']:
        # Convert cells from list format to tuple format
        cells = [tuple(cell) for cell in cage_data['cells']]
        operation = map_operation(cage_data['operation'])
        cage = Cage(operation, cage_data['target'], cells)
        cages.append(cage)
    return Puzzle(size, cages)

def test_generate_and_solve(size=6):
    """Test puzzle generation and solving with all methods."""
    print(f"\nTesting {size}x{size} puzzle...")
    
    # Generate puzzle
    print("Generating puzzle...")
    puzzle_dict, solution = generate_kenken(size)
    print(f"Generated puzzle with {len(puzzle_dict['cages'])} cages")
    
    # Convert dictionary to Puzzle object
    puzzle = create_puzzle_from_dict(puzzle_dict)
    
    # Create solver with known solution
    solver = Solver(puzzle, known_solution=solution)
    
    # Try all solving methods
    methods = ['backtracking', 'mrv', 'lcv', 'heuristics']
    results = {}
    
    for method in methods:
        print(f"\nTrying {method} method...")
        success, metrics = solver.solve(method=method)
        results[method] = {
            'success': success,
            'metrics': metrics
        }
        
        print(f"Success: {success}")
        print(f"Time: {metrics['time_seconds']:.3f} seconds")
        print(f"Nodes visited: {metrics['nodes_visited']}")
        print(f"Matches solution: {metrics['matches_known_solution']}")
        
        # Reset puzzle for next method
        solver.puzzle.reset_grid()
    
    return results

def main():
    """Run tests for different puzzle sizes."""
    # Test small, medium and large puzzles
    sizes = [3, 6, 9]
    all_results = {}
    
    for size in sizes:
        try:
            results = test_generate_and_solve(size)
            all_results[size] = results
        except Exception as e:
            print(f"Error testing size {size}: {str(e)}")
            continue
    
    # Print summary
    print("\n=== Summary ===")
    for size in all_results:
        print(f"\n{size}x{size} Puzzle:")
        for method in all_results[size]:
            metrics = all_results[size][method]['metrics']
            print(f"{method:12} - Time: {metrics['time_seconds']:.3f}s, "
                  f"Nodes: {metrics['nodes_visited']}")

if __name__ == "__main__":
    main()
