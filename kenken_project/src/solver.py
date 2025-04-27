import time
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Set, Dict
from .puzzle import Puzzle
from .cage import Cage
from .supervised_solver import SupervisedSolver

@dataclass
class MoveData:
    """Stores data about each move made during solving."""
    puzzle_id: str  # Identifier for the puzzle being solved
    grid_state: List[List[int]]  # Current puzzle state
    chosen_cell: Tuple[int, int]  # (row, col) of selected cell
    value_placed: int  # Value attempted
    is_valid: bool  # Whether move was valid
    move_number: int  # Sequential move counter
    domain_size: int  # From get_current_domain()
    degree: int  # From get_degree()
    inverse_domain_size: float  # 1/(domain_size + epsilon)

class Solver:
    """Implements solving algorithms for KenKen puzzles."""

    def __init__(self, puzzle: Puzzle,
                 update_callback: Optional[Callable[[int, int, int], None]] = None,
                 delay_ms: int = 0,
                 collect_move_data: bool = False):
        self.puzzle = puzzle
        self.size = puzzle.size
        self.update_callback = update_callback
        self.delay_ms = delay_ms
        self.nodes_visited = 0
        self.solution_count = 0
        self.collect_move_data = collect_move_data
        self.current_move_number = 0
        self.move_data = [] if collect_move_data else None
        self._domain_cache = {}  # Cache for domain values
        self._constraint_cache = {}  # Cache for constraint checking results

    def _clear_domain_cache(self, row: int, col: int):
        """Clear domain cache for affected cells when a value changes"""
        # Clear cache for the changed cell
        self._domain_cache.pop((row, col), None)
        
        # Clear cache for cells in same row/column
        for i in range(self.size):
            self._domain_cache.pop((row, i), None)
            self._domain_cache.pop((i, col), None)
            
        # Clear cache for cells in same cage
        cage = self.puzzle.get_cage(row, col)
        if cage:
            for r, c in cage.cells:
                self._domain_cache.pop((r, c), None)

    def get_current_domain(self, row: int, col: int) -> List[int]:
        """Get possible values for a cell based on all constraints."""
        # Check cache first
        cache_key = (row, col)
        if cache_key in self._domain_cache:
            return self._domain_cache[cache_key]

        if self.puzzle.get_cell_value(row, col) != 0:
            return [self.puzzle.get_cell_value(row, col)]
        
        # Get basic domain from row/column constraints
        possible = set(range(1, self.size + 1))
        row_values = {self.puzzle.get_cell_value(row, i) for i in range(self.size) if i != col and self.puzzle.get_cell_value(row, i) != 0}
        col_values = {self.puzzle.get_cell_value(i, col) for i in range(self.size) if i != row and self.puzzle.get_cell_value(i, col) != 0}
        
        # Remove constrained values efficiently
        possible -= row_values | col_values
        
        # Further filter based on cage constraints
        cage = self.puzzle.get_cage(row, col)
        if cage:
            valid_values = set()
            for val in possible:
                # Try this value
                self.puzzle.set_cell_value(row, col, val)
                if self._check_cage_constraint(cage):
                    valid_values.add(val)
                self.puzzle.set_cell_value(row, col, 0)
            possible = valid_values
        
        result = sorted(list(possible))
        self._domain_cache[cache_key] = result
        return result

    def is_safe(self, row: int, col: int, num: int) -> bool:
        """Check if number can be placed in cell without violating row/col constraints."""
        for c in range(self.size):
            if self.puzzle.get_cell_value(row, c) == num: return False
        for r in range(self.size):
            if self.puzzle.get_cell_value(r, col) == num: return False
        return True

    def _check_cage_constraint(self, cage: Cage) -> bool:
        """Check if cage satisfies its constraint."""
        if not cage:
            return True

        cell_values = []
        for r, c in cage.cells:
            val = self.puzzle.get_cell_value(r, c)
            if val == 0:  # Empty cell
                return True  # Don't check incomplete cages
            cell_values.append(val)

        # Always evaluate cage as its operation (no special case for single cells)
        return cage.check(cell_values)

    def choose_next_variable_standard(self) -> Optional[Tuple[int, int]]:
        """Standard variable selection (top-left to bottom-right)."""
        return self.puzzle.find_empty_cell()

    def get_degree(self, row: int, col: int) -> int:
        """Count unassigned neighbors for degree heuristic."""
        degree = 0
        # Count unassigned in row
        for c in range(self.size):
            if c != col and self.puzzle.is_cell_empty(row, c): degree += 1
        # Count unassigned in column
        for r in range(self.size):
            if r != row and self.puzzle.is_cell_empty(r, col): degree += 1
        return degree

    def choose_next_variable_mrv(self) -> Optional[Tuple[int, int]]:
        """Minimum Remaining Values heuristic with degree tie-breaker."""
        empty_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.puzzle.is_cell_empty(r, c):
                    empty_cells.append((r, c))

        if not empty_cells:
            return None  # Grid is full

        best_cell = None
        min_domain_size = float('inf')
        max_degree = -1

        for r, c in empty_cells:
            domain_size = len(self.get_current_domain(r, c))
            degree = self.get_degree(r, c)

            if domain_size < min_domain_size:
                min_domain_size = domain_size
                max_degree = degree
                best_cell = (r, c)
            elif domain_size == min_domain_size and degree > max_degree:
                max_degree = degree
                best_cell = (r, c)

        return best_cell or empty_cells[0]  # Fallback

    def _get_unassigned_neighbors(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get unassigned neighbors in row, column and cage."""
        neighbors = set()
        # Row neighbors
        for c in range(self.size):
            if c != col and self.puzzle.is_cell_empty(row, c):
                neighbors.add((row, c))
        # Column neighbors
        for r in range(self.size):
            if r != row and self.puzzle.is_cell_empty(r, col):
                neighbors.add((r, col))
        # Cage neighbors
        cage = self.puzzle.get_cage(row, col)
        if cage:
            for r_c, c_c in cage.cells:
                if (r_c, c_c) != (row, col) and self.puzzle.is_cell_empty(r_c, c_c):
                    neighbors.add((r_c, c_c))
        return neighbors

    def get_lcv_ordered_values(self, row: int, col: int) -> List[int]:
        """Least Constraining Value ordering for domain values."""
        domain = self.get_current_domain(row, col)
        if not domain:
            return []

        value_constraints = []
        neighbors = self._get_unassigned_neighbors(row, col)

        for value in domain:
            constraint_count = 0
            # Count how many neighbors would lose this value from their domain
            for nr, nc in neighbors:
                neighbor_domain = self.get_current_domain(nr, nc)
                if value in neighbor_domain:
                    constraint_count += 1
            value_constraints.append((value, constraint_count))

        # Sort by least constraining first
        value_constraints.sort(key=lambda item: item[1])
        return [value for value, count in value_constraints]


    def solve(self, method: str = 'heuristics', find_all: bool = False) -> Tuple[bool, Dict[str, float]]:
        """Main solving interface. Returns (success, metrics)."""
        print(f"Solving with method: {method}")
        print("Initial puzzle state:")
        for row in self.puzzle.get_grid_copy():
            print(row)

        start_time = time.time()
        self.nodes_visited = 0
        self.solution_count = 0
        
        # Reset puzzle to initial state
        for r in range(self.size):
            for c in range(self.size):
                self.puzzle.set_cell_value(r, c, 0)
        
        # Perform solving
        if method == 'backtracking':
            success = self.solve_backtracking(find_all)
        elif method == 'mrv':
            success = self.solve_with_mrv(find_all)
        elif method == 'lcv':
            success = self.solve_with_lcv(find_all)
        elif method == 'supervised_mrv_lcv':
            success = self.solve_with_supervised_mrv_lcv(find_all)
        else:  # heuristics
            success = self.solve_with_heuristics(find_all)
            
        end_time = time.time()
        
        metrics = {
            'time_seconds': end_time - start_time,
            'nodes_visited': self.nodes_visited,
            'method': method
        }

        print("Final puzzle state:")
        for row in self.puzzle.get_grid_copy():
            print(row)
        print(f"Solved: {success}, Metrics: {metrics}")
            
        return success, metrics

    def _propagate_constraints(self, row: int, col: int, value: int) -> bool:
        """Propagate constraints to neighbors and check if assignment is still feasible."""
        # Check row/column neighbors
        for i in range(self.size):
            if i != col and self.puzzle.is_cell_empty(row, i):
                domain = self.get_current_domain(row, i)
                if value in domain and len(domain) == 1:
                    return False
            if i != row and self.puzzle.is_cell_empty(i, col):
                domain = self.get_current_domain(i, col)
                if value in domain and len(domain) == 1:
                    return False
        
        # Check cage neighbors
        cage = self.puzzle.get_cage(row, col)
        if cage:
            cage_values = self.puzzle.get_cage_values(cage)
            empty_cells = [(r, c) for r, c in cage.cells if self.puzzle.is_cell_empty(r, c)]
            if len(empty_cells) == 1:  # Last cell in cage
                remaining_val = cage.value
                for val in cage_values:
                    if val != 0:
                        if cage.operation_str == '+':
                            remaining_val -= val
                        elif cage.operation_str == '*':
                            remaining_val //= val
                if remaining_val < 1 or remaining_val > self.size:
                    return False
        return True

    def _forward_check(self, row: int, col: int, value: int) -> bool:
        """Check if assigning value causes any domain to become empty."""
        # Temporarily assign value
        self.puzzle.set_cell_value(row, col, value)
        
        # Check all affected neighbors
        result = self._propagate_constraints(row, col, value)
        
        # Restore original state
        self.puzzle.set_cell_value(row, col, 0)
        return result

    def solve_backtracking(self, find_all=False, depth=0, max_depth=1000, max_nodes=100000) -> bool:
        """Standard backtracking solver with forward checking and limits."""
        if depth > max_depth:
            print(f"Backtracking: Max depth {max_depth} exceeded at depth {depth}. Backtracking.")
            return False
        if self.nodes_visited > max_nodes:
            print(f"Backtracking: Max nodes {max_nodes} exceeded. Stopping search.")
            return False

        next_cell = self.choose_next_variable_standard()
        if not next_cell:
            self.solution_count += 1
            print(f"Solution found! Total solutions: {self.solution_count}")
            return True

        row, col = next_cell
        current_cage = self.puzzle.get_cage(row, col)
        domain = self.get_current_domain(row, col)
        print(f"Backtracking: Trying cell ({row}, {col}) with domain {domain}")
        
        for num in domain:
            self.nodes_visited += 1
            print(f"Backtracking: Trying value {num} at ({row}, {col}), nodes visited: {self.nodes_visited}")
            if self.is_safe(row, col, num) and self._forward_check(row, col, num):
                self.puzzle.set_cell_value(row, col, num)
                self._clear_domain_cache(row, col)
                
                if self.update_callback:
                    try: self.update_callback(row, col, num)
                    except InterruptedError: raise

                if self._check_cage_constraint(current_cage if current_cage else None):
                    if self.solve_backtracking(find_all, depth + 1, max_depth, max_nodes):
                        if not find_all:
                            return True

                print(f"Backtracking: Backtracking from value {num} at ({row}, {col})")
                self.puzzle.set_cell_value(row, col, 0)
                self._clear_domain_cache(row, col)
                if self.update_callback:
                    try: self.update_callback(row, col, 0)
                    except InterruptedError: raise

        return self.solution_count > 0 if find_all else False

    def solve_with_mrv(self, find_all=False, depth=0, max_depth=1000, max_nodes=100000) -> bool:
        """Solver using MRV heuristic with forward checking and limits."""
        if depth > max_depth:
            print(f"MRV: Max depth {max_depth} exceeded at depth {depth}. Backtracking.")
            return False
        if self.nodes_visited > max_nodes:
            print(f"MRV: Max nodes {max_nodes} exceeded. Stopping search.")
            return False

        next_cell = self.choose_next_variable_mrv()
        if not next_cell:
            self.solution_count += 1
            print(f"MRV: Solution found! Total solutions: {self.solution_count}")
            return True

        row, col = next_cell
        if self.collect_move_data:
            self.current_move_number += 1

        domain = self.get_current_domain(row, col)
        current_cage = self.puzzle.get_cage(row, col)
        print(f"MRV: Trying cell ({row}, {col}) with domain {domain}")
        
        for num in domain:
            self.nodes_visited += 1
            print(f"MRV: Trying value {num} at ({row}, {col}), nodes visited: {self.nodes_visited}")
            if self.is_safe(row, col, num) and self._forward_check(row, col, num):
                self.puzzle.set_cell_value(row, col, num)
                self._clear_domain_cache(row, col)
                
                if self.update_callback:
                    try: self.update_callback(row, col, num)
                    except InterruptedError: raise

                if self._check_cage_constraint(current_cage if current_cage else None):
                    if self.solve_with_mrv(find_all, depth + 1, max_depth, max_nodes):
                        if not find_all:
                            return True

                print(f"MRV: Backtracking from value {num} at ({row}, {col})")
                self.puzzle.set_cell_value(row, col, 0)
                self._clear_domain_cache(row, col)
                if self.update_callback:
                    try: self.update_callback(row, col, 0)
                    except InterruptedError: raise

        return self.solution_count > 0 if find_all else False

    def solve_with_lcv(self, find_all=False, depth=0, max_depth=1000, max_nodes=100000) -> bool:
        """Solver using only LCV heuristic for value ordering with standard variable selection and limits."""
        if depth > max_depth:
            print(f"LCV: Max depth {max_depth} exceeded at depth {depth}. Backtracking.")
            return False
        if self.nodes_visited > max_nodes:
            print(f"LCV: Max nodes {max_nodes} exceeded. Stopping search.")
            return False

        next_cell = self.choose_next_variable_standard()  # Standard selection
        if not next_cell:
            # Verify all cage constraints are satisfied
            for cage in self.puzzle.cages:
                if not self._check_cage_constraint(cage):
                    print("LCV: Cage constraint failed at solution check.")
                    return False
            self.solution_count += 1
            print(f"LCV: Solution found! Total solutions: {self.solution_count}")
            return True

        row, col = next_cell
        if self.collect_move_data:
            self.current_move_number += 1

        ordered_values = self.get_lcv_ordered_values(row, col)  # LCV ordering
        print(f"LCV: Trying cell ({row}, {col}) with ordered values {ordered_values}")
        for num in ordered_values:
            self.nodes_visited += 1
            print(f"LCV: Trying value {num} at ({row}, {col}), nodes visited: {self.nodes_visited}")
            self.puzzle.set_cell_value(row, col, num)
            self._clear_domain_cache(row, col)

            if self.collect_move_data:
                domain_size = len(self.get_current_domain(row, col))
                degree = self.get_degree(row, col)
                move = MoveData(
                    puzzle_id=self.puzzle.name if hasattr(self.puzzle, 'name') else 'unknown',
                    grid_state=self.puzzle.get_grid_copy(),
                    chosen_cell=(row, col),
                    value_placed=num,
                    is_valid=True,
                    move_number=self.current_move_number,
                    domain_size=domain_size,
                    degree=degree,
                    inverse_domain_size=1.0/(domain_size + 1e-6)
                )
                self.move_data.append(move)

            if self.solve_with_lcv(find_all, depth + 1, max_depth, max_nodes):
                if not find_all:
                    return True

            print(f"LCV: Backtracking from value {num} at ({row}, {col})")
            self.puzzle.set_cell_value(row, col, 0)
            self._clear_domain_cache(row, col)

        return self.solution_count > 0 if find_all else False

    def solve_with_heuristics(self, find_all=False, depth=0, max_depth=1000) -> bool:
        """Solver using both MRV for variable selection and LCV for value ordering."""
        if depth > max_depth:
            print(f"Heuristics: Max depth {max_depth} exceeded at depth {depth}. Backtracking.")
            return False

        next_cell = self.choose_next_variable_mrv()
        if not next_cell:
            # Verify all cage constraints are satisfied
            for cage in self.puzzle.cages:
                if not self._check_cage_constraint(cage):
                    print("Heuristics: Cage constraint failed at solution check.")
                    return False
            self.solution_count += 1
            print(f"Heuristics: Solution found! Total solutions: {self.solution_count}")
            return True

        row, col = next_cell
        ordered_values = self.get_lcv_ordered_values(row, col)
        print(f"Heuristics: Trying cell ({row}, {col}) with ordered values {ordered_values}")
        
        for num in ordered_values:
            self.nodes_visited += 1
            print(f"Heuristics: Trying value {num} at ({row}, {col}), nodes visited: {self.nodes_visited}")
            if self.is_safe(row, col, num):
                self.puzzle.set_cell_value(row, col, num)
                self._clear_domain_cache(row, col)
                
                # Check cage constraint
                current_cage = self.puzzle.get_cage(row, col)
                cage_constraint_ok = True
                if current_cage:
                    cage_values = self.puzzle.get_cage_values(current_cage)
                    if 0 not in cage_values and not self._check_cage_constraint(current_cage):
                        cage_constraint_ok = False
                
                if cage_constraint_ok and self.solve_with_heuristics(find_all, depth + 1, max_depth):
                    return True if not find_all else self.solution_count > 0
                
                print(f"Heuristics: Backtracking from value {num} at ({row}, {col})")
                self.puzzle.set_cell_value(row, col, 0)
                self._clear_domain_cache(row, col)

        return False

    def get_solution(self) -> List[List[int]]:
        """Retrieve the current state of the puzzle grid as the solution."""
        return self.puzzle.get_grid_copy()

    def solve_with_supervised_mrv_lcv(self, find_all=False, depth=0, max_depth=None, max_nodes=None, supervised_solver=None) -> bool:
        """
        Solver combining supervised predictions with MRV for variable selection and LCV for value ordering.
        """
        if max_depth is not None and depth > max_depth:
            return False
        if max_nodes is not None and self.nodes_visited > max_nodes:
            return False

        # Train supervised solver once before recursion
        if supervised_solver is None:
            supervised_solver = SupervisedSolver(self.puzzle)
            supervised_solver.train_decision_tree()

        next_cell = self.choose_next_variable_mrv()
        if not next_cell:
            # Verify all cage constraints are satisfied
            for cage in self.puzzle.cages:
                if not self._check_cage_constraint(cage):
                    return False
            self.solution_count += 1
            return True

        row, col = next_cell

        # Get domain values ordered by supervised prediction confidence or value
        domain = self.get_current_domain(row, col)
        if not domain:
            return False

        # Predict values for the cell using supervised solver
        try:
            predicted_value = supervised_solver.predict(row, col)
        except Exception:
            predicted_value = None

        # Order domain values: put predicted value first if valid
        if predicted_value in domain:
            ordered_values = [predicted_value] + [v for v in domain if v != predicted_value]
        else:
            ordered_values = domain

        # Further order by LCV heuristic
        lcv_ordered = self.get_lcv_ordered_values(row, col)
        # Merge orders: keep predicted value first, then LCV order for rest
        ordered_values = [v for v in ordered_values if v == predicted_value] + [v for v in lcv_ordered if v != predicted_value]

        for num in ordered_values:
            self.nodes_visited += 1
            if self.is_safe(row, col, num):
                self.puzzle.set_cell_value(row, col, num)
                self._clear_domain_cache(row, col)

                current_cage = self.puzzle.get_cage(row, col)
                cage_constraint_ok = True
                if current_cage:
                    cage_values = self.puzzle.get_cage_values(current_cage)
                    if 0 not in cage_values and not self._check_cage_constraint(current_cage):
                        cage_constraint_ok = False

                if cage_constraint_ok and self.solve_with_supervised_mrv_lcv(find_all, depth + 1, max_depth, max_nodes, supervised_solver):
                    return True if not find_all else self.solution_count > 0

                self.puzzle.set_cell_value(row, col, 0)
                self._clear_domain_cache(row, col)

        return False

    def validate_solution(self, solution_grid: List[List[int]]) -> bool:
        """
        Validate a given solution grid against the puzzle constraints:
        - Each row contains unique numbers 1..size
        - Each column contains unique numbers 1..size
        - Each cage satisfies its operation and target
        """
        size = self.size

        # Check rows for uniqueness and valid values
        for r in range(size):
            row_values = solution_grid[r]
            if sorted(row_values) != list(range(1, size + 1)):
                print(f"Validation failed: Row {r} has invalid or duplicate values: {row_values}")
                return False

        # Check columns for uniqueness and valid values
        for c in range(size):
            col_values = [solution_grid[r][c] for r in range(size)]
            if sorted(col_values) != list(range(1, size + 1)):
                print(f"Validation failed: Column {c} has invalid or duplicate values: {col_values}")
                return False

        # Check cages constraints
        for cage in self.puzzle.cages:
            cage_values = [solution_grid[r][c] for r, c in cage.cells]
            if not cage.check(cage_values):
                print(f"Validation failed: Cage {cage} constraint not satisfied with values {cage_values}")
                return False

        # All checks passed
        print("Validation succeeded: Solution satisfies all constraints.")
        return True

    def validate_solution_detailed(self, solution_grid: List[List[int]]) -> (bool, list):
        """
        Validate a given solution grid against the puzzle constraints.
        Returns a tuple (is_valid, error_messages).
        """
        size = self.size
        errors = []

        # Check rows for uniqueness and valid values
        for r in range(size):
            row_values = solution_grid[r]
            if sorted(row_values) != list(range(1, size + 1)):
                errors.append(f"Row {r} has invalid or duplicate values: {row_values}")

        # Check columns for uniqueness and valid values
        for c in range(size):
            col_values = [solution_grid[r][c] for r in range(size)]
            if sorted(col_values) != list(range(1, size + 1)):
                errors.append(f"Column {c} has invalid or duplicate values: {col_values}")

        # Check cages constraints
        for cage in self.puzzle.cages:
            cage_values = [solution_grid[r][c] for r, c in cage.cells]
            if not cage.check(cage_values):
                errors.append(f"Cage {cage} constraint not satisfied with values {cage_values}")

        is_valid = len(errors) == 0
        return is_valid, errors
