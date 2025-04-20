from vision_solver import VisionSolver
from generator import generate_kenken
import matplotlib.pyplot as plt

# Generate a new puzzle
size = 6
puzzle, solution = generate_kenken(size)

# Create a vision solver and use the generated puzzle
solver = VisionSolver(size)
solver.use_generated_puzzle(puzzle)

# Solve the puzzle
puzzle_json = solver.capture_puzzle()
success, solved_puzzle = solver.solve_puzzle(puzzle_json)

if success:
    print("Puzzle solved successfully!")
    print("\nOriginal solution:")
    for row in solution:
        print(row)
    plt.show(block=True)
else:
    print("Failed to solve the puzzle")

solver.release()
