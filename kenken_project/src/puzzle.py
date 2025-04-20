import json
from typing import List, Tuple, Optional, Dict
from .cage import Cage

class Puzzle:
    """Represents the KenKen puzzle grid and its constraints."""

    def __init__(self, size: int, cages: List[Cage]):
        self.size = size
        self.cages = cages
        # Initialize grid with 0 representing empty cells
        self.grid: List[List[int]] = [[0] * size for _ in range(size)]
        # Precompute cell-to-cage mapping for faster lookups
        self._cell_to_cage_map: Dict[Tuple[int, int], Cage] = {}
        for cage in cages:
            for cell in cage.cells:
                r, c = tuple(cell)  # Convert to tuple in case it's a list
                if not (0 <= r < size and 0 <= c < size):
                    raise ValueError(f"Cell {cell} in cage {cage} is outside the grid boundaries (0 to {size-1}).")
                if (r, c) in self._cell_to_cage_map:
                    raise ValueError(f"Cell {cell} belongs to multiple cages (Cage {cage} and Cage {self._cell_to_cage_map[(r, c)]}).")
                self._cell_to_cage_map[(r, c)] = cage

        # Check if all cells are covered by cages - Essential for valid KenKen
        all_grid_cells = set((r, c) for r in range(size) for c in range(size))
        all_cage_cells = set(tuple(cell) for cage in cages for cell in cage.cells)

        if all_grid_cells != all_cage_cells:
            missing = all_grid_cells - all_cage_cells
            extra = all_cage_cells - all_grid_cells
            error_msgs = []
            if missing:
                error_msgs.append(f"Cells not covered by any cage: {sorted(list(missing))}")
            if extra:
                error_msgs.append(f"Cage cells outside grid bounds: {sorted(list(extra))}")
            if error_msgs:
                raise ValueError("Invalid puzzle definition: " + "; ".join(error_msgs))

    def reset_grid(self):
        """Reset the grid to empty state."""
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]

    @classmethod
    def from_json(cls, filepath: str) -> 'Puzzle':
        """Loads a puzzle definition from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        size = data['size']
        cages_data = data['cages']
        cages = []
        for i, c_data in enumerate(cages_data):
            try:
                cells = [tuple(cell) for cell in c_data['cells']]
                cage = Cage(c_data['operation'], c_data['value'], cells)
                cages.append(cage)
            except KeyError as e:
                raise ValueError(f"Missing key {e} in cage definition #{i+1}: {c_data}")
            except ValueError as e:
                 raise ValueError(f"Error in cage definition #{i+1} ({c_data}): {e}")

        return cls(size, cages)

    @classmethod
    def from_dict(cls, data):
        """Create a Puzzle object from a dictionary."""
        size = data.get("size")
        cages = [Cage(cage['operation'], cage['value'], [tuple(cell) for cell in cage['cells']]) for cage in data.get("cages", [])]
        return cls(size=size, cages=cages)

    def get_cage(self, row: int, col: int) -> Optional[Cage]:
        """Returns the cage that contains the given cell."""
        return self._cell_to_cage_map.get((row, col))

    def get_cell_value(self, row: int, col: int) -> int:
        """Gets the value of a cell (0 if empty)."""
        return self.grid[row][col]

    def set_cell_value(self, row: int, col: int, value: int):
        """Sets the value of a cell."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError("Cell coordinates out of bounds.")
        if not (0 <= value <= self.size):
             raise ValueError(f"Invalid value '{value}' for grid size {self.size}. Must be 0-{self.size}.")
        self.grid[row][col] = value

    def get_cage_values(self, cage: Cage) -> List[int]:
        """Gets the current values of cells within a specific cage."""
        return [self.grid[r][c] for r, c in cage.cells]

    def is_cell_empty(self, row: int, col: int) -> bool:
        """Checks if a cell is empty (contains 0)."""
        return self.grid[row][col] == 0

    def find_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Finds the next empty cell (top-left bias: row by row, col by col)."""
        for r in range(self.size):
            for c in range(self.size):
                if self.is_cell_empty(r, c):
                    return (r, c)
        return None # No empty cells left

    def get_grid_copy(self):
        """Returns a deep copy of the current grid state."""
        return [row.copy() for row in self.grid]

    def display(self):
        """Prints the current grid state to the console."""
        if not self.grid:
            print("Grid is empty.")
            return
        # Adjust spacing based on max number size (e.g., 2 digits for size 10+)
        width = len(str(self.size)) + 1
        header_line = "+".join(["-" * width] * self.size)
        print(header_line)
        for r in range(self.size):
            row_str = "|".join(f"{val:^{width}}" if val != 0 else f"{'.':^{width}}" for val in self.grid[r])
            print(f"|{row_str}|")
            if r < self.size - 1:
                 print(header_line) # Print separator between rows
        print(header_line) # Footer line

    def display_with_cages(self):
        """Prints the grid along with cage definitions to the console."""
        print(f"--- Puzzle {self.size}x{self.size} ---")
        self.display()
        print("\nCages:")
        for i, cage in enumerate(self.cages):
            print(f"  {i+1}: {cage}")
        print("--------------------")