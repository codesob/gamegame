import matplotlib.pyplot as plt
from generator import generate_kenken
from rl_solver import RLSolver
from puzzle import Puzzle

def main():
    size = 4
    puzzle_data, solution = generate_kenken(size)

    print("Generated puzzle:")
    # Convert puzzle_data dict to Puzzle object
    # Adjust keys if needed (target -> value)
    for cage in puzzle_data.get("cages", []):
        if 'target' in cage:
            cage['value'] = cage.pop('target')
    puzzle = Puzzle.from_dict(puzzle_data)

    for row in puzzle.get_grid_copy():
        print(row)

    rl_solver = RLSolver(puzzle)

    print("Training RL solver...")
    rl_solver.train(episodes=100)  # Adjust episodes as needed

    print("Solving puzzle with trained RL solver...")
    solved_grid = rl_solver.solve()

    print("Solved puzzle:")
    for row in solved_grid:
        print(row)

    # Optionally visualize solution here if visualizer is available

if __name__ == "__main__":
    main()
