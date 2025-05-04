def calculate_accuracy(true_grid, predicted_grid):
    """
    Calculate accuracy as the proportion of matching cells between true and predicted grids.
    Both grids are expected to be 2D lists of the same size.
    """
    if not true_grid or not predicted_grid:
        return 0.0
    total_cells = 0
    correct_cells = 0
    for r in range(len(true_grid)):
        for c in range(len(true_grid[r])):
            total_cells += 1
            if true_grid[r][c] == predicted_grid[r][c]:
                correct_cells += 1
    return correct_cells / total_cells if total_cells > 0 else 0.0
