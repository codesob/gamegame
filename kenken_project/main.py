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

PUZZLES_DIR = Path(__file__).parent / "puzzles"

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
        center_window(solve_window, 500, 400)
        solve_window.configure(bg="lightyellow")

        tk.Label(solve_window, text="Select Puzzle Size", font=("Arial", 20), bg="lightyellow").pack(pady=10)

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

        # Add button for supervised solvers
        tk.Button(solve_window, text="Supervised Solvers", font=("Arial", 16, "bold"), bg="orange", fg="black", command=open_supervised_solver_window).pack(pady=10)

    def open_supervised_solver_window():
        sup_window = tk.Toplevel(root)
        sup_window.title("Supervised Solvers")
        center_window(sup_window, 600, 500)
        sup_window.configure(bg="lightcyan")

        tk.Label(sup_window, text="Supervised Solvers", font=("Arial", 24, "bold"), bg="lightcyan").pack(pady=20)

        # Dropdown to select puzzle size
        tk.Label(sup_window, text="Select Puzzle Size:", font=("Arial", 16), bg="lightcyan").pack(pady=5)
        puzzle_sizes = list(range(3, 10))
        selected_size = tk.IntVar(sup_window)
        selected_size.set(puzzle_sizes[0])
        size_menu = tk.OptionMenu(sup_window, selected_size, *puzzle_sizes)
        size_menu.pack(pady=5)

        # Dropdown to select solver type
        tk.Label(sup_window, text="Select Solver Type:", font=("Arial", 16), bg="lightcyan").pack(pady=5)
        solver_types = ["Decision Tree", "Random Forest", "KMeans"]
        selected_solver = tk.StringVar(sup_window)
        selected_solver.set(solver_types[0])
        solver_menu = tk.OptionMenu(sup_window, selected_solver, *solver_types)
        solver_menu.pack(pady=5)

        # Dropdown to select puzzle file
        tk.Label(sup_window, text="Select Puzzle File:", font=("Arial", 16), bg="lightcyan").pack(pady=5)
        puzzle_files = [f for f in os.listdir(str(PUZZLES_DIR)) if f.endswith(".json")]
        selected_puzzle_file = tk.StringVar(sup_window)
        if puzzle_files:
            selected_puzzle_file.set(puzzle_files[0])
        else:
            selected_puzzle_file.set("No puzzles found")
        puzzle_file_menu = tk.OptionMenu(sup_window, selected_puzzle_file, *puzzle_files)
        puzzle_file_menu.pack(pady=5)

        result_text = tk.Text(sup_window, height=10, width=60)
        result_text.pack(pady=10)

        def run_supervised_solver():
            import time
            puzzle_file = selected_puzzle_file.get()
            solver_type = selected_solver.get()
            size = selected_size.get()

            if puzzle_file == "No puzzles found":
                messagebox.showerror("Error", "No puzzle files found in puzzles directory.")
                return

            puzzle_path = os.path.join(str(PUZZLES_DIR), puzzle_file)
            try:
                puzzle, solution = load_puzzle_from_file(puzzle_path)
                from src.supervised_solver import SupervisedSolver
                from src.accuracy_recorder import AccuracyRecorder

                accuracy_recorder = AccuracyRecorder()

                solver = SupervisedSolver(puzzle, solution_grid=solution)

                start_time = time.time()
                if solver_type == "Decision Tree":
                    solver.train_decision_tree()
                elif solver_type == "Random Forest":
                    solver.train_random_forest()
                elif solver_type == "KMeans":
                    solver.train_kmeans(n_clusters=puzzle.size)
                else:
                    messagebox.showerror("Error", f"Unknown solver type: {solver_type}")
                    return
                solved = solver.solve()
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Calculate accuracy
                accuracy = None
                if solution:
                    from src.utils import calculate_accuracy
                    accuracy = calculate_accuracy(solution, solved)
                # Record accuracy
                print(f"Recording supervised solver accuracy: solver_type={solver_type}, size={size}, accuracy={accuracy}, elapsed_time={elapsed_time}")
                accuracy_recorder.record(solver_type, size, puzzle_file=puzzle_file, accuracy=accuracy, elapsed_time=elapsed_time)


                # Display results
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Solver: {solver_type}\n")
                result_text.insert(tk.END, f"Puzzle Size: {size}x{size}\n")
                result_text.insert(tk.END, f"Time Taken: {elapsed_time:.4f} seconds\n")
                if accuracy is not None:
                    result_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n")
                result_text.insert(tk.END, "Solved Grid:\n")
                for row in solved:
                    result_text.insert(tk.END, " ".join(str(val) for val in row) + "\n")

            except Exception as e:
                messagebox.showerror("Error", str(e))

        tk.Button(sup_window, text="Run Solver", font=("Arial", 16), bg="green", fg="white", command=run_supervised_solver).pack(pady=10)

    def open_size_window(size):
        """Open a new window for the selected puzzle size."""
        size_window = tk.Toplevel(root)
        size_window.title(f"{size}x{size} Puzzle")
        center_window(size_window, 600, 400)
        size_window.configure(bg="lightgray")

        tk.Label(size_window, text=f"{size}x{size} Puzzle", font=("Arial", 20, "bold"), bg="lightgray").pack(pady=20)

        # Dropdown to select a puzzle from the puzzles folder
        puzzle_files = [f for f in os.listdir(str(PUZZLES_DIR)) if f.startswith(f"kenken_{size}x{size}") and f.endswith(".json")]
        selected_puzzle = tk.StringVar(size_window)
        selected_puzzle.set("Select a puzzle")

        tk.OptionMenu(size_window, selected_puzzle, *puzzle_files).pack(pady=10)

        # Dropdown to select solving algorithm
        algorithms = ["backtracking", "mrv", "lcv", "heuristics", "supervised_mrv_lcv"]
        selected_algorithm = tk.StringVar(size_window)
        selected_algorithm.set("Select an algorithm")

        tk.OptionMenu(size_window, selected_algorithm, *algorithms).pack(pady=10)

        def solve_puzzle():
            import time
            puzzle_file = selected_puzzle.get()
            algorithm = selected_algorithm.get()

            if puzzle_file == "Select a puzzle" or algorithm == "Select an algorithm":
                messagebox.showerror("Error", "Please select both a puzzle and an algorithm.")
                return

            puzzle_path = os.path.join(str(PUZZLES_DIR), puzzle_file)
            try:
                puzzle, solution = load_puzzle_from_file(puzzle_path)
                from src.accuracy_recorder import AccuracyRecorder

                accuracy_recorder = AccuracyRecorder()

                process_viz = ProcessVisualizer(puzzle)  # Ensure process_viz is initialized here
                solver = Solver(puzzle, update_callback=process_viz.update)

                start_time = time.time()
                success, metrics = process_viz.start(solver, algorithm)
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Add elapsed time to metrics for display
                metrics['elapsed_time'] = elapsed_time

                # Calculate accuracy if solution is available
                accuracy = None
                if solution and success:
                    solved_grid = solver.get_solution()
                    from src.utils import calculate_accuracy
                    accuracy = calculate_accuracy(solution, solved_grid)
                # Record accuracy
                nodes_visited = metrics.get('nodes_visited') if metrics else None
                elapsed_time_corrected = metrics.get('time_seconds', elapsed_time)
                print(f"Recording CSP accuracy: algorithm={algorithm}, size={puzzle.size}, accuracy={accuracy}, elapsed_time={elapsed_time_corrected}, nodes_visited={nodes_visited}")
                accuracy_recorder.record(algorithm, puzzle.size, puzzle_file=os.path.basename(puzzle_path), accuracy=accuracy, elapsed_time=elapsed_time_corrected, nodes_visited=nodes_visited)


                info_message = f"Solved: {success}\nMetrics: {metrics}\nNodes Visited: {metrics.get('nodes_visited', 'N/A')}\nTime Taken: {elapsed_time:.2f} seconds"
                if accuracy is not None:
                    info_message += f"\nAccuracy: {accuracy:.4f}"

                messagebox.showinfo("Result", info_message)
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
                save_puzzle(puzzle, solution, base_dir=str(PUZZLES_DIR))
                messagebox.showinfo("Success", f"Generated {size}x{size} puzzle and saved to puzzles folder.")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid size between 3 and 9.")

        tk.Button(generate_window, text="Generate", font=("Arial", 16), command=generate_puzzle, bg="orange", fg="white").pack(pady=20)

    def open_play_window():
        """Open a new window for manual play."""
        play_window = tk.Toplevel(root)
        play_window.title("Play KenKen Puzzle")
        
        # Get screen dimensions
        screen_width = play_window.winfo_screenwidth()
        screen_height = play_window.winfo_screenheight()
        
        # Set initial window size to 90% of screen size
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Create main frame with scrollbars
        main_frame = tk.Frame(play_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbars
        h_scrollbar = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        v_scrollbar = tk.Scrollbar(main_frame)
        canvas_container = tk.Canvas(main_frame, 
                                   xscrollcommand=h_scrollbar.set,
                                   yscrollcommand=v_scrollbar.set)

        # Configure scrollbars
        h_scrollbar.config(command=canvas_container.xview)
        v_scrollbar.config(command=canvas_container.yview)
        
        # Pack scrollbars and canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create frame inside canvas for content
        content_frame = tk.Frame(canvas_container)
        canvas_container.create_window((0, 0), window=content_frame, anchor="nw")

        # Add title and controls to content frame
        tk.Label(content_frame, text="Play KenKen Puzzle", 
                font=("Arial", 20, "bold"), bg="lightblue").pack(pady=20)

        # Dropdown to select a puzzle
        puzzle_files = [f for f in os.listdir(str(PUZZLES_DIR)) if f.endswith(".json")]
        selected_puzzle = tk.StringVar(play_window)
        selected_puzzle.set("Select a puzzle")
        tk.OptionMenu(content_frame, selected_puzzle, *puzzle_files).pack(pady=10)

        def load_and_play():
            puzzle_file = selected_puzzle.get()
            if puzzle_file == "Select a puzzle":
                messagebox.showerror("Error", "Please select a puzzle.")
                return

            puzzle_path = os.path.join(str(PUZZLES_DIR), puzzle_file)
            try:
                puzzle, solution = load_puzzle_from_file(puzzle_path)
                size = puzzle.size

                # Clear previous puzzle if exists
                for widget in content_frame.winfo_children():
                    if isinstance(widget, tk.Canvas):
                        widget.destroy()

                # Calculate cell size based on window size
                max_puzzle_size = min(window_width - 100, window_height - 200)  # Leave space for controls
                cell_size = min(60, max_puzzle_size // size)  # Adjust cell size based on puzzle size
                padding = 20
                total_size = (cell_size * size) + (2 * padding)

                # Create puzzle canvas
                canvas = tk.Canvas(content_frame, width=total_size, height=total_size, 
                                 bg="white")
                canvas.pack(pady=20)

                entries = []
                entry_widgets = {}

                def create_validator(row, col):
                    def validate(P):
                        if P == "":
                            entry_widgets[(row, col)].config(bg="white")
                            return True
                        if P.isdigit():
                            num = int(P)
                            valid = 1 <= num <= size
                            entry_widgets[(row, col)].config(bg="white" if valid else "pink")
                            return valid
                        return False
                    return validate

                # Draw grid and create entries
                for r in range(size):
                    row_entries = []
                    for c in range(size):
                        x1 = c * cell_size + padding
                        y1 = r * cell_size + padding
                        x2 = x1 + cell_size
                        y2 = y1 + cell_size

                        canvas.create_rectangle(x1, y1, x2, y2, outline="gray")

                        entry = tk.Entry(canvas, width=1, font=("Arial", int(cell_size/2)), 
                                       justify="center")
                        entry_window = canvas.create_window(x1 + cell_size/2, 
                                                          y1 + cell_size/2, 
                                                          window=entry)
                        row_entries.append(entry)
                        entry_widgets[(r, c)] = entry
                        
                        validator = create_validator(r, c)
                        entry.config(validate="key", 
                                   validatecommand=(play_window.register(validator), '%P'))

                    entries.append(row_entries)

                # Draw cage boundaries and operations
                for cage in puzzle.cages:
                    cells = cage.cells
                    operation_text = f"{cage.operation_str}{cage.value}"

                    for i, (r, c) in enumerate(cells):
                        x1 = c * cell_size + padding
                        y1 = r * cell_size + padding
                        x2 = x1 + cell_size
                        y2 = y1 + cell_size

                        for adj_r, adj_c in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                            if (adj_r, adj_c) not in cells:
                                if adj_r == r - 1:
                                    canvas.create_line(x1, y1, x2, y1, width=2, fill="black")
                                elif adj_r == r + 1:
                                    canvas.create_line(x1, y2, x2, y2, width=2, fill="black")
                                elif adj_c == c - 1:
                                    canvas.create_line(x1, y1, x1, y2, width=2, fill="black")
                                elif adj_c == c + 1:
                                    canvas.create_line(x2, y1, x2, y2, width=2, fill="black")

                    # Add operation text
                    first_cell = cells[0]
                    x = first_cell[1] * cell_size + padding + 2
                    y = first_cell[0] * cell_size + padding + 2
                    font_size = max(8, int(cell_size/5))  # Adjust font size based on cell size
                    canvas.create_text(x, y, text=operation_text, 
                                     anchor="nw", font=("Arial", font_size, "bold"))

                # Add a label to show validation messages inline
                validation_message_label = tk.Label(content_frame, text="", font=("Arial", 14), bg="lightgray", fg="red")
                validation_message_label.pack(pady=5)

                def validate_solution():
                    user_solution = []
                    for r in range(size):
                        row = []
                        for c in range(size):
                            value = entries[r][c].get()
                            if not value:
                                validation_message_label.config(text="Please fill in all cells.", fg="red")
                                return
                            if not value.isdigit():
                                validation_message_label.config(text=f"Invalid input at cell ({r+1}, {c+1})", fg="red")
                                return
                            row.append(int(value))
                        user_solution.append(row)

                    if user_solution == solution:
                        validation_message_label.config(text="Congratulations! Your solution is correct!", fg="green")
                        for r in range(size):
                            for c in range(size):
                                entries[r][c].config(bg="lightgreen")
                    else:
                        validation_message_label.config(text="Incorrect solution. Please try again.", fg="red")

                validate_button = tk.Button(content_frame, text="Validate Solution", 
                                          font=("Arial", 16), command=validate_solution,
                                          bg="green", fg="white")
                validate_button.pack(pady=10, side=tk.LEFT, padx=10)

                def clear_entries():
                    for r in range(size):
                        for c in range(size):
                            entries[r][c].delete(0, tk.END)
                            entries[r][c].config(bg="white")

                clear_button = tk.Button(content_frame, text="Clear", font=("Arial", 16),
                                         command=clear_entries, bg="red", fg="white")
                clear_button.pack(pady=10, side=tk.LEFT, padx=10)

                # Update scroll region after adding all widgets
                content_frame.update_idletasks()
                canvas_container.config(scrollregion=canvas_container.bbox("all"))

            except Exception as e:
                messagebox.showerror("Error", str(e))

        load_button = tk.Button(content_frame, text="Load Puzzle", font=("Arial", 16),
                               command=load_and_play, bg="orange", fg="white")
        load_button.pack(pady=20)

        # Set window size and position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        play_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Update scroll region initially
        content_frame.update_idletasks()
        canvas_container.config(scrollregion=canvas_container.bbox("all"))

        # Make window resizable
        play_window.resizable(True, True)

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
