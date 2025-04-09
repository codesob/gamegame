import argparse
import json
from pathlib import Path
from src.solver import Solver
from src.puzzle import Puzzle
from src.cage import Cage
from src.pygame_renderer import KenKenRenderer
from src.process_visualizer import ProcessVisualizer

def load_puzzle_from_file(filepath):
    """Load puzzle and solution from JSON file."""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Puzzle file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {filepath}. Error: {str(e)}")
                raise
        
        # Validate puzzle structure
        if not isinstance(data, dict):
            raise ValueError(f"Invalid puzzle file: root must be an object, got {type(data)}")
            
        if 'puzzle' not in data:
            raise ValueError(f"Invalid puzzle file: missing 'puzzle' key in {filepath}")
            
        puzzle_data = data['puzzle']
        if 'size' not in puzzle_data:
            raise ValueError("Invalid puzzle data: missing 'size'")
        if 'cages' not in puzzle_data:
            raise ValueError("Invalid puzzle data: missing 'cages'")
            
        solution = data.get('solution')
        
        # Validate cell coverage
        size = puzzle_data['size']
        covered_cells = set()
        for cage_data in puzzle_data['cages']:
            cells = set(tuple(cell) for cell in cage_data['cells'])
            covered_cells.update(cells)
            
        all_cells = {(r, c) for r in range(size) for c in range(size)}
        missing_cells = all_cells - covered_cells
        
        if missing_cells:
            raise ValueError(f"Invalid puzzle definition: Cells not covered by any cage: {sorted(list(missing_cells))}")
        
        # Convert JSON cage data to Cage objects
        cages = []
        for i, cage_data in enumerate(puzzle_data['cages']):
            try:
                if 'cells' not in cage_data:
                    raise ValueError(f"Cage {i}: missing 'cells'")
                if 'operation' not in cage_data:
                    raise ValueError(f"Cage {i}: missing 'operation'")
                if 'target' not in cage_data:
                    raise ValueError(f"Cage {i}: missing 'target'")
                    
                cells = [tuple(cell) for cell in cage_data['cells']]
                cage = Cage(
                    cage_data['operation'],
                    cage_data['target'],
                    cells
                )
                cages.append(cage)
            except Exception as e:
                raise ValueError(f"Error in cage {i}: {str(e)}")
        
        # Create and validate Puzzle object
        puzzle = Puzzle(puzzle_data['size'], cages)
        
        return puzzle, solution
        
    except FileNotFoundError:
        raise ValueError(f"Puzzle file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading puzzle: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='KenKen Puzzle Solver')
    parser.add_argument('--puzzle', type=str, required=True, help='Path to puzzle JSON file')
    parser.add_argument('--method', choices=['backtracking', 'mrv', 'lcv', 'heuristics'],
                      default='heuristics', help='Solving method')
    parser.add_argument('--visualize', action='store_true', help='Visualize the solution')
    args = parser.parse_args()

    try:
        # Load puzzle
        puzzle, solution = load_puzzle_from_file(args.puzzle)
        
        # Create solver and visualizer
        process_viz = ProcessVisualizer(puzzle) if args.visualize else None
        solver = Solver(puzzle,
                      update_callback=process_viz.update if args.visualize else None,
                      delay_ms=300 if args.visualize else 0)
        
        # Start visualization if requested
        if args.visualize and not process_viz.start():
            print("Visualization cancelled by user")
            return
        
        # Solve puzzle
        success, metrics = solver.solve(method=args.method)
        
        # Show results
        print(f"\nSolved using {args.method}!")
        print(f"Time: {metrics['time_seconds']:.3f} seconds")
        print(f"Nodes visited: {metrics['nodes_visited']}")
        
        # Finish visualization if active
        if args.visualize:
            process_viz.finish(success, args.method)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
