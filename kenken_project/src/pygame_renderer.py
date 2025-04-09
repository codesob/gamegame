import pygame
import sys
from typing import List, Tuple, Optional
from .cage import Cage

class KenKenRenderer:
    def __init__(self, size: int, width: int = 600, height: int = 600):
        pygame.init()
        self.size = size
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('KenKen Puzzle')
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.BLUE = (0, 0, 255)
        
        # Calculate cell size
        self.cell_size = min(width, height) // size
        self.font = pygame.font.Font(None, self.cell_size // 2)
        self.small_font = pygame.font.Font(None, self.cell_size // 3)

    def draw_grid(self, cages: List[Cage], values: Optional[List[List[int]]] = None):
        """Draw the complete KenKen grid with cages and values."""
        self.screen.fill(self.WHITE)
        
        # Draw cages
        for cage in cages:
            self._draw_cage(cage)
        
        # Draw grid lines
        for i in range(self.size + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.BLACK,
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.size * self.cell_size),
                           2 if i % self.size == 0 else 1)
            # Horizontal lines
            pygame.draw.line(self.screen, self.BLACK,
                           (0, i * self.cell_size),
                           (self.size * self.cell_size, i * self.cell_size),
                           2 if i % self.size == 0 else 1)
        
        # Draw values if provided
        if values:
            self._draw_values(values)
        
        pygame.display.flip()

    def _draw_cage(self, cage: Cage):
        """Draw a single cage with its operation and target."""
        cells = list(cage.cells)  # Convert set to list
        operation = cage.operation_str
        target = cage.value
        
        # Draw thick borders around cage
        for cell in cells:
            row, col = cell
            x = col * self.cell_size
            y = row * self.cell_size
            
            # Check if neighboring cells are in the same cage
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            for nr, nc in neighbors:
                if (nr, nc) not in cells:  # Draw thick line if neighbor not in cage
                    if nr == row - 1:  # Top
                        pygame.draw.line(self.screen, self.BLACK,
                                       (x, y), (x + self.cell_size, y), 3)
                    elif nr == row + 1:  # Bottom
                        pygame.draw.line(self.screen, self.BLACK,
                                       (x, y + self.cell_size),
                                       (x + self.cell_size, y + self.cell_size), 3)
                    elif nc == col - 1:  # Left
                        pygame.draw.line(self.screen, self.BLACK,
                                       (x, y), (x, y + self.cell_size), 3)
                    elif nc == col + 1:  # Right
                        pygame.draw.line(self.screen, self.BLACK,
                                       (x + self.cell_size, y),
                                       (x + self.cell_size, y + self.cell_size), 3)
        
        # Draw operation and target in top-left cell
        top_left = min(cells)
        row, col = top_left
        x = col * self.cell_size
        y = row * self.cell_size
        
        text = f"{target}{operation}"
        text_surface = self.small_font.render(text, True, self.BLUE)
        self.screen.blit(text_surface, (x + 5, y + 5))

    def _draw_values(self, values):
        """Draw the current values in the grid."""
        for row in range(self.size):
            for col in range(self.size):
                value = values[row][col]
                if value != 0:
                    x = col * self.cell_size + self.cell_size // 2
                    y = row * self.cell_size + self.cell_size // 2
                    text = str(value)
                    text_surface = self.font.render(text, True, self.BLACK)
                    text_rect = text_surface.get_rect(center=(x, y))
                    self.screen.blit(text_surface, text_rect)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
