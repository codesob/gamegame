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
    """
    Parses a multi-line string representation of a KenKen puzzle.
    Returns (size, list_of_cages). Raises ValueError on errors.
    """
    lines = [line.strip() for line in text_input.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Input text is empty.")

    try:
        size = int(lines[0])
        if size <= 0: raise ValueError("Grid size must be positive.")
    except ValueError:
        raise ValueError(f"Invalid grid size: '{lines[0]}'. Expected integer.")

    cages = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) < 3:
             raise ValueError(f"Invalid format line {i}: '{line}'.")

        value_op_str = parts[0]
        op_match = re.match(r"(\d+)([+\-*/x÷=])", value_op_str)
        if not op_match:
             raise ValueError(f"Invalid target/op line {i}: '{value_op_str}'.")

        value_str, op_symbol = op_match.groups()
        try: value = int(value_str)
        except ValueError: raise ValueError(f"Invalid value line {i}: '{value_str}'.")

        if op_symbol == 'x': op_symbol = '*'
        if op_symbol == '÷': op_symbol = '/'
        if op_symbol not in ('+', '-', '*', '/', '='):
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

    # --- Add validation: Check if all cells 1..N*N are covered ---
    all_grid_cells = set((r, c) for r in range(size) for c in range(size))
    all_cage_cells = set(cell for cage in cages for cell in cage.cells)
    if all_grid_cells != all_cage_cells:
         missing = sorted(list(all_grid_cells - all_cage_cells))
         extra = sorted(list(all_cage_cells - all_grid_cells))
         error_msgs = []
         if missing: error_msgs.append(f"Cells not covered by any cage: {missing}")
         if extra: error_msgs.append(f"Cage cells outside grid: {extra}") # Should be caught earlier
         raise ValueError("Invalid text puzzle definition: " + "; ".join(error_msgs))
    # --- End validation ---

    return size, cages