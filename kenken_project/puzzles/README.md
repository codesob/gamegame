## Puzzle File Format (JSON)

Each puzzle is defined in a JSON file with the following structure:

{
  "size": N, // The grid size (N x N)
  "cages": [
    {
      "operation": "+", // or "-", "*", "/", "=" (for single-cell cages)
      "value": TARGET_VALUE,
      "cells": [ [row1, col1], [row2, col2], ... ] // 0-indexed cell coordinates
    },
    // ... more cages
  ]
}

### Operations Notes:

*   `+`: Addition
*   `*`: Multiplication
*   `-`: Subtraction (between two cells, order doesn't matter, result is positive)
*   `/`: Division (between two cells, order doesn't matter, result is integer quotient)
*   `=`: Assignment (for single-cell cages, value is the number in the cell)