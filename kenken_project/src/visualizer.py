import pygame
import sys
import time
from typing import Optional, Callable, Dict, Set, Tuple as TypingTuple # Avoid collision with pygame.Tuple
from .puzzle import Puzzle
from .cage import Cage

# --- Constants ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
RED = (255, 50, 50)
BLUE = (50, 50, 255)
GREEN = (50, 200, 50)

LINE_COLOR = GRAY
THICK_LINE_COLOR = BLACK
NUMBER_COLOR = BLACK
CAGE_INFO_COLOR = BLUE
BACKGROUND_COLOR = WHITE
HIGHLIGHT_COLOR = (255, 255, 150) # Yellowish highlight for current cell


class KenKenVisualizer:
    """Handles Pygame visualization of the KenKen puzzle and solving process."""

    def __init__(self, puzzle: Puzzle, cell_size: int = 60, margin: int = 20):
        if cell_size < 20 or margin < 5:
            print("Warning: Cell size or margin might be too small for clear visualization.")
        self.puzzle = puzzle
        self.size = puzzle.size
        self.cell_size = cell_size
        self.margin = margin
        self.width = self.size * self.cell_size + 2 * self.margin
        self.height = self.size * self.cell_size + 2 * self.margin
        self.bottom_bar_height = 50 # Extra space for messages

        pygame.init()
        pygame.font.init() # Explicitly initialize font module
        self.screen = pygame.display.set_mode((self.width, self.height + self.bottom_bar_height))
        pygame.display.set_caption(f"KenKen Solver ({self.size}x{self.size})")

        # Attempt to load fonts, fall back to default if specific ones fail
        try:
            self.number_font = pygame.font.SysFont('Arial', int(cell_size * 0.6))
        except:
            print("Arial font not found, using default pygame font for numbers.")
            self.number_font = pygame.font.Font(None, int(cell_size * 0.6))
        try:
             self.cage_font = pygame.font.SysFont('Arial', int(cell_size * 0.25))
        except:
            print("Arial font not found, using default pygame font for cages.")
            self.cage_font = pygame.font.Font(None, int(cell_size * 0.25))
        try:
            self.message_font = pygame.font.SysFont('Arial', 30)
        except:
            print("Arial font not found, using default pygame font for messages.")
            self.message_font = pygame.font.Font(None, 30)

        self.clock = pygame.time.Clock()

        # Pre-calculate cage boundaries and info positions for efficiency
        self._cage_boundaries = self._calculate_cage_boundaries()
        self._cage_info_pos = self._calculate_cage_info_positions()

    def _get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get the pygame.Rect for a given cell's drawing area."""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _calculate_cage_boundaries(self) -> Set[TypingTuple[TypingTuple[int, int], TypingTuple[int, int]]]:
        """Determines which internal grid lines should be thick (cage boundaries)."""
        boundaries = set()
        for r in range(self.size):
            for c in range(self.size):
                cell_cage = self.puzzle.get_cage(r, c)
                if not cell_cage: continue # Should not happen with validation in Puzzle

                # Check right boundary
                if c + 1 < self.size:
                    right_neighbor_cage = self.puzzle.get_cage(r, c + 1)
                    # Draw thick line if neighbor is in a different cage
                    if cell_cage != right_neighbor_cage:
                        p1 = (self.margin + (c + 1) * self.cell_size, self.margin + r * self.cell_size)
                        p2 = (self.margin + (c + 1) * self.cell_size, self.margin + (r + 1) * self.cell_size)
                        # Store points in consistent order (e.g., top-to-bottom for vertical)
                        boundaries.add((min(p1,p2), max(p1,p2)))

                # Check bottom boundary
                if r + 1 < self.size:
                    bottom_neighbor_cage = self.puzzle.get_cage(r + 1, c)
                     # Draw thick line if neighbor is in a different cage
                    if cell_cage != bottom_neighbor_cage:
                        p1 = (self.margin + c * self.cell_size, self.margin + (r + 1) * self.cell_size)
                        p2 = (self.margin + (c + 1) * self.cell_size, self.margin + (r + 1) * self.cell_size)
                        # Store points in consistent order (e.g., left-to-right for horizontal)
                        boundaries.add((min(p1,p2), max(p1,p2)))
        return boundaries

    def _calculate_cage_info_positions(self) -> Dict[Cage, TypingTuple[int, int]]:
        """Finds the top-leftmost cell for each cage to display info."""
        positions = {}
        for cage in self.puzzle.cages:
            # Find cell with minimum row, then minimum column within that row
            top_left_cell = min(cage.cells, key=lambda x: (x[0], x[1]))
            rect = self._get_cell_rect(top_left_cell[0], top_left_cell[1])
            # Position slightly offset inside the top-left corner
            positions[cage] = (rect.x + 3, rect.y + 1)
        return positions

    def draw_grid(self):
        """Draws the grid lines (thin and thick cage boundaries)."""
        # Draw thin lines for the entire grid first
        for i in range(self.size + 1):
            # Vertical lines
            start_pos_v = (self.margin + i * self.cell_size, self.margin)
            end_pos_v = (self.margin + i * self.cell_size, self.height - self.margin)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_v, end_pos_v, 1)
            # Horizontal lines
            start_pos_h = (self.margin, self.margin + i * self.cell_size)
            end_pos_h = (self.width - self.margin, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_h, end_pos_h, 1)

        # Draw thick lines for cage boundaries over the thin lines
        for p1, p2 in self._cage_boundaries:
            pygame.draw.line(self.screen, THICK_LINE_COLOR, p1, p2, 3)

        # Draw outer border last to ensure it's on top
        pygame.draw.rect(self.screen, THICK_LINE_COLOR,
                         (self.margin, self.margin, self.size * self.cell_size, self.size * self.cell_size), 3)


    def draw_cages_info(self):
         """Draws the operation and value for each cage."""
         for cage, pos in self._cage_info_pos.items():
             # Construct the text string (e.g., "12+", "3-", "4")
             text = f"{cage.value}"
             if cage.operation_str != "=": # Don't show '=' for single cells
                 text += cage.operation_str

             surface = self.cage_font.render(text, True, CAGE_INFO_COLOR)
             self.screen.blit(surface, pos)


    def draw_numbers(self, highlight_cell: Optional[TypingTuple[int, int]] = None, clear_only=False):
        """Draws the numbers currently in the puzzle grid."""
        for r in range(self.size):
            for c in range(self.size):
                rect = self._get_cell_rect(r, c)
                value = self.puzzle.get_cell_value(r, c)

                # Highlight the cell currently being processed by the solver
                if highlight_cell and (r, c) == highlight_cell:
                    pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect)
                # Fill background for empty cells or when clearing before redraw
                elif value == 0 or clear_only:
                     pygame.draw.rect(self.screen, BACKGROUND_COLOR, rect)

                # Draw the number if the cell is not empty
                if value != 0 and not clear_only:
                    try:
                        num_surface = self.number_font.render(str(value), True, NUMBER_COLOR)
                        num_rect = num_surface.get_rect(center=rect.center)
                        self.screen.blit(num_surface, num_rect)
                    except pygame.error as e:
                         print(f"Error rendering number {value}: {e}")
                         # Optionally draw a placeholder if rendering fails
                         pygame.draw.rect(self.screen, RED, rect, 2) # Draw red box as error indicator


    def draw_all(self, message: str = "", highlight_cell: Optional[TypingTuple[int, int]] = None):
        """Draws the entire puzzle state from scratch."""
        # 1. Fill background
        self.screen.fill(BACKGROUND_COLOR)

        # 2. Draw numbers (clearing background first)
        self.draw_numbers(highlight_cell, clear_only=True) # Clear first
        self.draw_numbers(highlight_cell, clear_only=False) # Then draw numbers/highlight

        # 3. Draw grid lines (thin and thick) on top of numbers/background
        self.draw_grid()

        # 4. Draw cage info on top of grid lines
        self.draw_cages_info()

        # 5. Draw status message at the bottom
        self.show_message(message)

        # 6. Update the display
        pygame.display.flip()


    def show_message(self, message: str, color: tuple = BLACK):
        """Displays a message in the bottom bar."""
        # Clear previous message area
        clear_rect = pygame.Rect(0, self.height, self.width, self.bottom_bar_height)
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, clear_rect)

        if message:
            try:
                text_surface = self.message_font.render(message, True, color)
                text_rect = text_surface.get_rect(center=(self.width / 2, self.height + self.bottom_bar_height / 2))
                self.screen.blit(text_surface, text_rect)
            except pygame.error as e:
                 print(f"Error rendering message '{message}': {e}")
        # Note: flip/update needs to be called *after* this by the main draw call (like draw_all)


    def update_cell_display(self, row: int, col: int, num: int, delay_ms: int = 0):
        """
        Callback function for the solver to update the visualizer efficiently.
        Redraws only the affected cell and its immediate surroundings.
        """
        # 1. Handle Pygame events (essential to keep window responsive and allow quitting)
        if not self.handle_events():
             print("Visualization quit by user.")
             # Raise a specific exception or use a flag to signal the solver to stop
             raise InterruptedError("Visualization closed by user.")


        # 2. Define the rectangle for the cell being updated
        rect = self._get_cell_rect(row, col)

        # 3. Draw the cell background (either normal or highlighted)
        # This covers the previous number or highlight
        if num != 0: # If putting a number, highlight it temporarily
             pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect)
        else: # If clearing (num=0), just draw normal background
             pygame.draw.rect(self.screen, BACKGROUND_COLOR, rect)


        # 4. Draw the number (if not 0)
        if num != 0:
            try:
                num_surface = self.number_font.render(str(num), True, NUMBER_COLOR)
                num_rect = num_surface.get_rect(center=rect.center)
                self.screen.blit(num_surface, num_rect)
            except pygame.error as e:
                 print(f"Error rendering number {num} in update_cell_display: {e}")


        # 5. Redraw grid lines around this cell (they might have been painted over)
        # Thin lines:
        pygame.draw.line(self.screen, LINE_COLOR, rect.topleft, rect.topright, 1)
        pygame.draw.line(self.screen, LINE_COLOR, rect.topleft, rect.bottomleft, 1)
        pygame.draw.line(self.screen, LINE_COLOR, rect.bottomleft, rect.bottomright, 1)
        pygame.draw.line(self.screen, LINE_COLOR, rect.topright, rect.bottomright, 1)
        # Thick cage lines + outer border:
        self._redraw_thick_lines_around_cell(row, col)


        # 6. Redraw cage info if it's the top-left cell for its cage (might be overdrawn)
        self._redraw_cage_info_if_needed(row, col)

        # 7. Update only the changed rectangle on the screen
        pygame.display.update(rect)

        # 8. Optional delay to visualize steps
        if delay_ms > 0:
             # Use pygame.time.wait for simplicity, ensures delay happens
             pygame.time.wait(delay_ms)


    def _redraw_thick_lines_around_cell(self, r, c):
        """Helper to redraw thick cage lines and outer border around a specific cell."""
        cell_cage = self.puzzle.get_cage(r, c)
        if not cell_cage: return

        # Check right boundary
        if c + 1 < self.size:
            right_neighbor_cage = self.puzzle.get_cage(r, c + 1)
            if cell_cage != right_neighbor_cage:
                p1 = (self.margin + (c + 1) * self.cell_size, self.margin + r * self.cell_size)
                p2 = (self.margin + (c + 1) * self.cell_size, self.margin + (r + 1) * self.cell_size)
                pygame.draw.line(self.screen, THICK_LINE_COLOR, p1, p2, 3)

        # Check bottom boundary
        if r + 1 < self.size:
            bottom_neighbor_cage = self.puzzle.get_cage(r + 1, c)
            if cell_cage != bottom_neighbor_cage:
                p1 = (self.margin + c * self.cell_size, self.margin + (r + 1) * self.cell_size)
                p2 = (self.margin + (c + 1) * self.cell_size, self.margin + (r + 1) * self.cell_size)
                pygame.draw.line(self.screen, THICK_LINE_COLOR, p1, p2, 3)

        # Check left boundary (redraw line to the left of *this* cell)
        if c > 0:
             left_neighbor_cage = self.puzzle.get_cage(r, c-1)
             if cell_cage != left_neighbor_cage:
                 p1 = (self.margin + c * self.cell_size, self.margin + r * self.cell_size)
                 p2 = (self.margin + c * self.cell_size, self.margin + (r + 1) * self.cell_size)
                 pygame.draw.line(self.screen, THICK_LINE_COLOR, p1, p2, 3)

        # Check top boundary (redraw line above *this* cell)
        if r > 0:
            top_neighbor_cage = self.puzzle.get_cage(r - 1, c)
            if cell_cage != top_neighbor_cage:
                p1 = (self.margin + c * self.cell_size, self.margin + r * self.cell_size)
                p2 = (self.margin + (c + 1) * self.cell_size, self.margin + r * self.cell_size)
                pygame.draw.line(self.screen, THICK_LINE_COLOR, p1, p2, 3)

            # Redraw outer border segments if the cell is on the edge
        # --- ADD THIS LINE ---
        rect = self._get_cell_rect(r, c) # Define rect for use in border drawing
        # --- END ADDED LINE ---
        if r == 0: # Top border segment
             pygame.draw.line(self.screen, THICK_LINE_COLOR, (rect.left, rect.top), (rect.right, rect.top), 3)
        if r == self.size - 1: # Bottom border segment
             pygame.draw.line(self.screen, THICK_LINE_COLOR, (rect.left, rect.bottom), (rect.right, rect.bottom), 3)
        if c == 0: # Left border segment
             pygame.draw.line(self.screen, THICK_LINE_COLOR, (rect.left, rect.top), (rect.left, rect.bottom), 3)
        if c == self.size - 1: # Right border segment
             pygame.draw.line(self.screen, THICK_LINE_COLOR, (rect.right, rect.top), (rect.right, rect.bottom), 3)
    def _redraw_cage_info_if_needed(self, row, col):
        """Redraws cage info if it was potentially overwritten in the updated cell."""
        cage = self.puzzle.get_cage(row, col)
        # Check if this cell is the one designated for displaying this cage's info
        if cage and self._cage_info_pos.get(cage) == (self._get_cell_rect(row, col).x + 3, self._get_cell_rect(row, col).y + 1):
             # Simplest is to redraw that specific cage's info
             text = f"{cage.value}"
             if cage.operation_str != "=": text += cage.operation_str
             try:
                 surface = self.cage_font.render(text, True, CAGE_INFO_COLOR)
                 self.screen.blit(surface, self._cage_info_pos[cage])
             except pygame.error as e:
                 print(f"Error rendering cage info {text} in _redraw_cage_info: {e}")


    def handle_events(self) -> bool:
        """Process Pygame events. Returns False if QUIT event occurs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # Allow closing with Esc
                    return False
        return True

    def wait_for_keypress_or_close(self, message="Press any key or close window to continue..."):
        """ Waits for user input before proceeding. Returns False if window closed. """
        self.show_message(message)
        pygame.display.flip() # Ensure message is shown

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False # User closed window
                if event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_ESCAPE: # Esc also closes
                         return False
                     return True # Any other key proceeds
                if event.type == pygame.MOUSEBUTTONDOWN: # Clicking also proceeds
                    return True
            self.clock.tick(30) # Prevent busy-waiting


    def wait_for_close(self, message="Solver finished. Close the window to exit."):
        """Keeps the window open after solving until user closes it."""
        self.show_message(message)
        pygame.display.flip() # Ensure message is shown

        while self.handle_events():
            self.clock.tick(30) # Keep CPU usage reasonable while waiting
        self.quit()

    def quit(self):
        """Cleans up Pygame."""
        print("Quitting Pygame...")
        pygame.quit()