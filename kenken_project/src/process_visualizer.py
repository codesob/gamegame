from .visualizer import KenKenVisualizer
from .puzzle import Puzzle
import time
import pygame

class ProcessVisualizer:
    def __init__(self, puzzle: Puzzle, cell_size: int = 60):
        self.visualizer = KenKenVisualizer(puzzle, cell_size)
        self.delay_ms = 300  # Default delay between steps
        self.start_time = None
        self.nodes_visited = 0
        self.solving_method = ""
        self._update_counter = 0
        self._batch_size = 50  # Update visualization every N steps
    
    def start(self):
        """Initialize visualization"""
        self.start_time = time.time()
        self.visualizer.draw_all("Starting solver...")
        return self.visualizer.wait_for_keypress_or_close("Press any key to start solving...")
    
    def update(self, row: int, col: int, value: int, method: str = "", domain_size: int = 0, degree: int = 0):
        """Update visualization with heuristic information"""
        self.nodes_visited += 1
        
        if method == "mrv":
            message = f"MRV: Choosing cell ({row+1},{col+1}) with smallest domain ({domain_size} options)"
        elif method == "lcv":
            message = f"LCV: Trying value {value} (constrains {degree} neighbors least)"
        elif method == "heuristics":
            message = f"MRV+LCV: Cell ({row+1},{col+1}) with domain {domain_size}, trying {value} (constrains {degree})"
        else:  # backtracking
            message = f"Trying {value} at ({row+1},{col+1})" if value != 0 else f"Backtracking at ({row+1},{col+1})"
        
        self.visualizer.update_cell_display(row, col, value, self.delay_ms)
        self.visualizer.show_message(message)
        pygame.display.flip()
        
        # Reset counter after update
        self._update_counter = 0
            
        # Print progress periodically
        if self.nodes_visited % 1000 == 0:
            print(f"\nProgress: {self.nodes_visited} nodes visited")
    
    def print_solution_grid(self):
        """Print the current grid state in terminal"""
        puzzle = self.visualizer.puzzle
        size = puzzle.size
        print("\nCurrent Solution Grid:")
        print("-" * (size * 4 + 1))
        for i in range(size):
            print("|", end="")
            for j in range(size):
                value = puzzle.grid[i][j] or " "
                print(f" {value} |", end="")
            print("\n" + "-" * (size * 4 + 1))
    
    def finish(self, success: bool, solving_method: str = ""):
        """Show final state and print solution with statistics"""
        end_time = time.time()
        solving_time = end_time - self.start_time
        self.solving_method = solving_method
        
        message = "Solved successfully! Close window to exit." if success else "No solution found. Close window to exit."
        self.visualizer.draw_all(message)
        
        if success:
            print("\n=== KenKen Solution ===")
            print(f"Solved using {self.solving_method} algorithm")
            print(f"Time: {solving_time:.3f} seconds")
            print(f"Nodes visited: {self.nodes_visited}")
            self.print_solution_grid()
            print("\nPress close button on the window to exit.")
        
        self.visualizer.wait_for_close()
    
    def set_delay(self, delay_ms: int):
        """Set delay between steps"""
        self.delay_ms = delay_ms
