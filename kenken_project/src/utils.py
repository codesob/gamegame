# src/utils.py
import re
from typing import Tuple, List, Optional

# Need to import Cage relatively IF parse_text_puzzle creates Cage objects directly
# If parse_text_puzzle only returns raw data (size, list_of_cage_data), then Cage import isn't needed here.
# Let's assume the version that returns size and list of Cages:
try:
    # This relative import works when the module is part of the 'src' package
    from .cage import Cage
except ImportError:
    # Fallback for potential direct script execution (less ideal)
     print("Warning: Relative import of Cage failed in utils.py. Ensure running as part of package.")
     # Define a dummy Cage if needed for type hinting, or handle differently
     class Cage: pass # Dummy

def cell_notation_to_coords(cell_str: str) -> Optional[Tuple[int, int]]:
    """
    Converts spreadsheet notation (e.g., 'A1', 'b3', 'C10') to (row, col) tuple (0-indexed).
    Returns None if format is invalid. Handles columns A-Z.
    """
    cell_str = cell_str.strip().upper()
    match = re.match(r"([A-Z])(\d+)$", cell_str)

    if not match:
        return None # Invalid format

    col_str, row_str = match.groups()

    col_index = ord(col_str) - ord('A')

    try:
        row_index = int(row_str) - 1
        if row_index < 0:
             return None
    except ValueError:
        return None

    return (row_index, col_index)


def parse_text_puzzle(text_input: str) -> Tuple[int, List[Cage]]:
    """Parse a text representation of a KenKen puzzle."""
    lines = [line.strip() for line in text_input.splitlines() if line.strip()]
    
    # First line must be size
    try:
        size = int(lines[0])
        if size < 3:
            raise ValueError("Puzzle size must be at least 3")
    except ValueError as e:
        raise ValueError(f"Invalid size line: {lines[0]}") from e
    
    cages = []
    for i, line in enumerate(lines[1:], 1):
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line {i}: '{line}'.")
        
        value_op_str = parts[0]
        op_match = re.match(r"(\d+)([+\-*/])", value_op_str)
        if not op_match:
            raise ValueError(f"Invalid target/op line {i}: '{value_op_str}'.")

        value_str, op_symbol = op_match.groups()
        try: 
            value = int(value_str)
        except ValueError: 
            raise ValueError(f"Invalid value line {i}: '{value_str}'.")

        if op_symbol == 'x': 
            op_symbol = '*'
        if op_symbol == '÷': 
            op_symbol = '/'
        if op_symbol not in ('+', '-', '*', '/'):
            raise ValueError(f"Invalid op symbol line {i}: '{op_symbol}'.")

        cell_coords_list = []
        for cell_notation in parts[1:]:
            coords = cell_notation_to_coords(cell_notation)
            if coords is None:
                raise ValueError(f"Invalid cell notation line {i}: '{cell_notation}'.")
            if not (0 <= coords[0] < size and 0 <= coords[1] < size):
                raise ValueError(f"Cell '{cell_notation}' ({coords}) line {i} out of bounds (size {size}).")
            cell_coords_list.append(coords)

        if not cell_coords_list:
            raise ValueError(f"No cells defined line {i}: '{line}'.")

        try:
            cage = Cage(operation=op_symbol, value=value, cells=cell_coords_list)
            cages.append(cage)
        except ValueError as e:
            raise ValueError(f"Error creating cage line {i}: {e}")

    # Validate all cells are covered
    all_grid_cells = set((r, c) for r in range(size) for c in range(size))
    all_cage_cells = set(cell for cage in cages for cell in cage.cells)
    if all_grid_cells != all_cage_cells:
        missing = sorted(list(all_grid_cells - all_cage_cells))
        extra = sorted(list(all_cage_cells - all_grid_cells))
        error_msgs = []
        if missing: 
            error_msgs.append(f"Cells not covered by any cage: {missing}")
        if extra: 
            error_msgs.append(f"Cells outside grid bounds: {sorted(list(extra))}")
        raise ValueError("Invalid puzzle definition: " + "; ".join(error_msgs))

    return size, cages