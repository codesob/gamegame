import argparse
import json
from pathlib import Path
from src.solver import Solver
from src.puzzle import Puzzle
from src.cage import Cage
from src.visualizer import KenKenRenderer, ProcessVisualizer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from src.generator import generate_kenken, save_puzzle

def load_puzzle_from_file(filepath):
    """Load puzzle and solution from JSON file."""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Puzzle file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {filepath}. Error: {str(e)}")
                raise
        
        # Validate puzzle structure
        if not isinstance(data, dict):
            raise ValueError(f"Invalid puzzle file: root must be an object, got {type(data)}")
            
        if 'puzzle' not in data:
            raise ValueError(f"Invalid puzzle file: missing 'puzzle' key in {filepath}")
            
        puzzle_data = data['puzzle']
        if 'size' not in puzzle_data:
            raise ValueError("Invalid puzzle data: missing 'size'")
        if 'cages' not in puzzle_data:
            raise ValueError("Invalid puzzle data: missing 'cages'")
            
        solution = data.get('solution')
        
        # Validate cell coverage
        size = puzzle_data['size']
        covered_cells = set()
        for cage_data in puzzle_data['cages']:
            cells = set(tuple(cell) for cell in cage_data['cells'])
            covered_cells.update(cells)
            
        all_cells = {(r, c) for r in range(size) for c in range(size)}
        missing_cells = all_cells - covered_cells
        
        if missing_cells:
            raise ValueError(f"Invalid puzzle definition: Cells not covered by any cage: {sorted(list(missing_cells))}")
        
        # Convert JSON cage data to Cage objects
        cages = []
        for i, cage_data in enumerate(puzzle_data['cages']):
            try:
                if 'cells' not in cage_data:
                    raise ValueError(f"Cage {i}: missing 'cells'")
                if 'operation' not in cage_data:
                    raise ValueError(f"Cage {i}: missing 'operation'")
                if 'target' not in cage_data:
                    raise ValueError(f"Cage {i}: missing 'target'")
                    
                cells = [tuple(cell) for cell in cage_data['cells']]
                cage = Cage(
                    cage_data['operation'],
                    cage_data['target'],
                    cells
                )
                cages.append(cage)
            except Exception as e:
                raise ValueError(f"Error in cage {i}: {str(e)}")
        
        # Create and validate Puzzle object
        puzzle = Puzzle(puzzle_data['size'], cages)
        
        return puzzle, solution
        
    except FileNotFoundError:
        raise ValueError(f"Puzzle file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading puzzle: {str(e)}")

def center_window(window, width, height):
    """Center a window on the screen."""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

def main_window():
    def open_solve_window():
        solve_window = tk.Toplevel(root)
        solve_window.title("Solve KenKen Puzzle")
        center_window(solve_window, 500, 350)
        solve_window.configure(bg="lightyellow")

        tk.Label(solve_window, text="Select Puzzle Size", font=("Arial", 20), bg="lightyellow").pack(pady=20)

        # Add a scrollable frame for the buttons
        canvas = tk.Canvas(solve_window, bg="lightyellow")
        scrollbar = tk.Scrollbar(solve_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="lightyellow")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for size in range(3, 10):
            tk.Button(scrollable_frame, text=f"{size}x{size}", font=("Arial", 16), command=lambda s=size: open_size_window(s), bg="lightblue", width=10, height=2).pack(pady=5)

    def open_size_window(size):
        """Open a new window for the selected puzzle size."""
        size_window = tk.Toplevel(root)
        size_window.title(f"{size}x{size} Puzzle")
        center_window(size_window, 600, 400)
        size_window.configure(bg="lightgray")

        tk.Label(size_window, text=f"{size}x{size} Puzzle", font=("Arial", 20, "bold"), bg="lightgray").pack(pady=20)

        # Dropdown to select a puzzle from the puzzles folder
        puzzle_files = [f for f in os.listdir("puzzles") if f.startswith(f"kenken_{size}x{size}") and f.endswith(".json")]
        selected_puzzle = tk.StringVar(size_window)
        selected_puzzle.set("Select a puzzle")

        tk.OptionMenu(size_window, selected_puzzle, *puzzle_files).pack(pady=10)

        # Dropdown to select solving algorithm
        algorithms = ["backtracking", "mrv", "lcv", "heuristics"]
        selected_algorithm = tk.StringVar(size_window)
        selected_algorithm.set("Select an algorithm")

        tk.OptionMenu(size_window, selected_algorithm, *algorithms).pack(pady=10)

        def solve_puzzle():
            puzzle_file = selected_puzzle.get()
            algorithm = selected_algorithm.get()

            if puzzle_file == "Select a puzzle" or algorithm == "Select an algorithm":
                messagebox.showerror("Error", "Please select both a puzzle and an algorithm.")
                return

            puzzle_path = os.path.join("puzzles", puzzle_file)
            try:
                puzzle, _ = load_puzzle_from_file(puzzle_path)
                process_viz = ProcessVisualizer(puzzle)  # Ensure process_viz is initialized here
                solver = Solver(puzzle, update_callback=process_viz.update)
                success, metrics = process_viz.start(solver, algorithm)
                messagebox.showinfo("Result", f"Solved: {success}\nMetrics: {metrics}\nNodes Visited: {metrics['nodes_visited']}\nTime Taken: {metrics['time_seconds']:.2f} seconds")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(size_window, text="Solve", font=("Arial", 16), command=solve_puzzle, bg="green", fg="white").pack(pady=20)

    def open_generate_window():
        generate_window = tk.Toplevel(root)
        generate_window.title("Generate KenKen Puzzle")
        center_window(generate_window, 500, 300)
        generate_window.configure(bg="lightgreen")

        tk.Label(generate_window, text="Enter size (3 to 9):", font=("Arial", 18), bg="lightgreen").pack(pady=20)
        size_entry = tk.Entry(generate_window, font=("Arial", 16))
        size_entry.pack(pady=10)

        def generate_puzzle():
            try:
                size = int(size_entry.get())
                if size < 3 or size > 9:
                    raise ValueError
                puzzle, solution = generate_kenken(size)
                save_puzzle(puzzle, solution)
                messagebox.showinfo("Success", f"Generated {size}x{size} puzzle and saved to puzzles folder.")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid size between 3 and 9.")

        tk.Button(generate_window, text="Generate", font=("Arial", 16), command=generate_puzzle, bg="orange", fg="white").pack(pady=20)

    def open_play_window():
        """Open a new window for manual play."""
        play_window = tk.Toplevel(root)
        play_window.title("Play KenKen Puzzle")
        center_window(play_window, 600, 600)
        play_window.configure(bg="lightblue")

        tk.Label(play_window, text="Play KenKen Puzzle", font=("Arial", 20, "bold"), bg="lightblue").pack(pady=20)

        # Dropdown to select a puzzle from the puzzles folder
        puzzle_files = [f for f in os.listdir("puzzles") if f.endswith(".json")]
        selected_puzzle = tk.StringVar(play_window)
        selected_puzzle.set("Select a puzzle")

        tk.OptionMenu(play_window, selected_puzzle, *puzzle_files).pack(pady=10)

        def load_and_play():
            puzzle_file = selected_puzzle.get()
            if puzzle_file == "Select a puzzle":
                messagebox.showerror("Error", "Please select a puzzle.")
                return

            puzzle_path = os.path.join("puzzles", puzzle_file)
            try:
                puzzle, solution = load_puzzle_from_file(puzzle_path)
                size = puzzle.size

                # Create a grid for manual input and display predefined values with operations
                entries = []
                grid_frame = tk.Frame(play_window, bg="lightblue")
                grid_frame.pack(pady=20)

                for r in range(size):
                    row_entries = []
                    for c in range(size):
                        value = puzzle.grid[r][c] if hasattr(puzzle, 'grid') else 0
                        cage_info = next((cage for cage in puzzle.cages if (r, c) in cage.cells), None)
                        if value != 0:  # If the cell has a predefined value
                            label = tk.Label(grid_frame, text=str(value), font=("Arial", 20), width=4, height=2, bg="lightgray", relief="solid")
                            label.grid(row=r, column=c, padx=5, pady=5)
                            row_entries.append(None)  # No entry for predefined cells
                        else:
                            entry = tk.Entry(grid_frame, width=4, font=("Arial", 20), justify="center")
                            entry.grid(row=r, column=c, padx=5, pady=5)
                            row_entries.append(entry)

                            # Restrict input to numeric values within the puzzle size
                            def validate_input(P):
                                if P == "":
                                    return True
                                if P.isdigit():
                                    num = int(P)
                                    return 1 <= num <= size
                                return False

                            reg = play_window.register(validate_input)
                            entry.config(validate="key", validatecommand=(reg, "%P"))

                        # Display cage operation and value in the top-left corner of the first cell of the cage
                        if cage_info and list(cage_info.cells)[0] == (r, c):
                            operation_text = f"{cage_info.operation_str}{cage_info.value}"
                            op_label = tk.Label(grid_frame, text=operation_text, font=("Arial", 12), bg="lightblue")
                            op_label.grid(row=r, column=c, sticky="nw", padx=2, pady=2)

                    entries.append(row_entries)

                def validate_solution():
                    user_solution = []
                    for r in range(size):
                        row = []
                        for c in range(size):
                            value = entries[r][c].get() if entries[r][c] else puzzle.grid[r][c]
                            if not value.isdigit():
                                messagebox.showerror("Error", f"Invalid input at cell ({r+1}, {c+1}). Please enter numbers only.")
                                return
                            row.append(int(value))
                        user_solution.append(row)

                    if user_solution == solution:
                        messagebox.showinfo("Success", "Congratulations! Your solution is correct.")
                    else:
                        messagebox.showerror("Error", "Incorrect solution. Please try again.")

                tk.Button(play_window, text="Validate Solution", font=("Arial", 16), command=validate_solution, bg="green", fg="white").pack(pady=20)

            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(play_window, text="Load Puzzle", font=("Arial", 16), command=load_and_play, bg="orange", fg="white").pack(pady=20)

    root = tk.Tk()
    root.title("KenKen Puzzle")
    root.geometry("600x400")
    center_window(root, 600, 400)
    root.configure(bg="lightblue")

    tk.Label(root, text="KenKen Puzzle", font=("Arial", 28, "bold"), bg="lightblue").pack(pady=30)

    tk.Button(root, text="Solve", font=("Arial", 18, "bold"), command=open_solve_window, width=20, bg="darkgreen", fg="white", relief="raised", bd=3).pack(pady=20)
    tk.Button(root, text="Generate", font=("Arial", 18, "bold"), command=open_generate_window, width=20, bg="darkblue", fg="white", relief="raised", bd=3).pack(pady=20)
    tk.Button(root, text="Play", font=("Arial", 18, "bold"), command=open_play_window, width=20, bg="purple", fg="white", relief="raised", bd=3).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main_window()
