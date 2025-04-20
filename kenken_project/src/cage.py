import operator
from typing import List, Tuple, Optional

class Cage:
    """Represents a single cage in a KenKen puzzle."""

    def __init__(self, operation, value, cells):
        self.operation_str = operation
        self.value = value
        # Convert cells to a list if it's not already
        self.cells = list(cells) if not isinstance(cells, list) else cells

        valid_operations = {'+', '-', '*', '/'}
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}")

        if operation in ('-', '/') and len(cells) != 2:
            raise ValueError(f"Operation '{operation}' requires exactly 2 cells.")

    def check(self, values):
        if 0 in values:
            return True  # Don't check incomplete cages

        if self.operation_str == '+':
            return sum(values) == self.value
        elif self.operation_str == '*':
            result = 1
            for v in values:
                result *= v
            return result == self.value
        elif self.operation_str == '-':
            if len(values) != 2:
                return False
            return abs(values[0] - values[1]) == self.value
        elif self.operation_str == '/':
            if len(values) != 2 or 0 in values:
                return False
            return max(values) / min(values) == self.value
        return False

    def __repr__(self) -> str:
        return f"Cage({self.value}{self.operation_str}, cells={self.cells})"