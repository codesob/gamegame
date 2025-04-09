import operator
from typing import List, Tuple, Optional

class Cage:
    """Represents a single cage in a KenKen puzzle."""

    def __init__(self, operation, value, cells):
        self.operation_str = operation
        self.value = value
        self.cells = set(cells)

        valid_operations = {'+', '-', '*', '/', '='}
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}")

        if operation == '=' and len(cells) != 1:
            raise ValueError("Operation '=' requires exactly 1 cell.")
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
            return abs(values[0] - values[1]) == self.value
        elif self.operation_str == '/':
            if values[1] == 0:
                return False
            return values[0] / values[1] == self.value or values[1] / values[0] == self.value
        elif self.operation_str == '=':
            return values[0] == self.value
        return False

    def __repr__(self) -> str:
        return f"Cage({self.value}{self.operation_str}, cells={self.cells})"