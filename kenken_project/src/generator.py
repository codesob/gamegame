import random
import json
import os
from datetime import datetime

def generate_latin_square(n):
    base = list(range(1, n+1))
    grid = []
    while len(grid) < n:
        row = base[:]
        random.shuffle(row)
        if all(row[i] != grid[j][i] for j in range(len(grid)) for i in range(n)):
            grid.append(row)
    return grid

def get_neighbors(cell, n):
    r, c = cell
    neighbors = []
    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            neighbors.append((nr, nc))
    return neighbors

def partition_into_cages(grid):
    n = len(grid)
    all_cells = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(all_cells)
    visited = set()
    cages = []

    while all_cells:
        start = all_cells.pop()
        if start in visited:
            continue
        cage = [start]
        visited.add(start)

        cage_size = random.choices([1, 2, 3, 4], weights=[40, 35, 20, 5])[0]
        for _ in range(cage_size - 1):
            neighbors = [nbr for cell in cage for nbr in get_neighbors(cell, n) if nbr not in visited]
            if not neighbors:
                break
            next_cell = random.choice(neighbors)
            cage.append(next_cell)
            visited.add(next_cell)
        cages.append(cage)

    return cages

def get_operation_str(op_code):
    """Map operation codes to string representations."""
    op_map = {
        1: '+',
        2: '-',
        3: '*', 
        4: '/',
        5: '='
    }
    return op_map.get(op_code, '+')  # Default to + if unknown

def compute_target_and_operator(cage, grid):
    """Compute target value and operation for a cage."""
    values = [grid[r][c] for r, c in cage]
    if len(cage) == 1:
        return values[0], "="
    
    random.shuffle(values)
    if len(cage) == 2:
        a, b = max(values), min(values)  # Ensure consistent order
        ops = []
        if a + b <= 100: 
            ops.append(('+', a + b))
        if a - b > 0:
            ops.append(('-', a - b))
        if a * b <= 100:
            ops.append(('*', a * b))
        if a % b == 0:
            ops.append(('/', a // b))
        if not ops:
            return a + b, '+'  # Fallback to addition
        op, target = random.choice(ops)
        return target, op
    else:
        # For 3+ cells, use only + or *
        if random.random() < 0.5:
            return sum(values), '+'
        else:
            product = 1
            for v in values:
                product *= v
            return product, '*'

def generate_kenken(n=5):
    """Generate a KenKen puzzle with string operations."""
    grid = generate_latin_square(n)
    cages = partition_into_cages(grid)
    puzzle = {
        "size": n,
        "cages": []
    }
    
    for cage in cages:
        target, operation = compute_target_and_operator(cage, grid)
        # Ensure operation is a valid string
        if not isinstance(operation, str):
            operation = get_operation_str(operation)
            
        puzzle["cages"].append({
            "cells": [[r, c] for r, c in cage],
            "operation": operation,
            "target": target
        })
    
    return puzzle, grid

def save_puzzle(puzzle, solution, base_dir="puzzles"):
    """Save puzzle and solution to JSON file."""
    # Create puzzles directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    size = puzzle['size']
    filename = f"kenken_{size}x{size}_{timestamp}.json"
    filepath = os.path.join(base_dir, filename)
    
    # Prepare data to save
    data = {
        "puzzle": puzzle,
        "solution": solution
    }
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath

def visualize_puzzle(puzzle, solution=None):
    """Visualize the generated puzzle using pygame."""
    from .pygame_renderer import KenKenRenderer
    from .cage import Cage
    
    # Convert dictionary cages to Cage objects
    cage_objects = []
    for cage_dict in puzzle['cages']:
        cage = Cage(
            cage_dict['operation'],
            cage_dict['target'],
            [tuple(cell) for cell in cage_dict['cells']]
        )
        cage_objects.append(cage)
    
    renderer = KenKenRenderer(puzzle['size'])
    running = True
    
    while running:
        renderer.draw_grid(cage_objects, solution)
        renderer.handle_events()

if __name__ == "__main__":
    size = int(input("Enter KenKen puzzle size (3-9): "))
    if size < 3 or size > 9:
        print("Only sizes between 3 and 9 are supported.")
    else:
        puzzle, solution = generate_kenken(size)
        
        # Save the puzzle
        saved_path = save_puzzle(puzzle, solution)
        print(f"\nPuzzle saved to: {saved_path}")
        
        print("\nGenerated KenKen Puzzle:")
        print(json.dumps(puzzle, indent=2))
        print("\nSolution Grid:")
        for row in solution:
            print(row)
        
        # Visualize the puzzle
        visualize_puzzle(puzzle, solution)
