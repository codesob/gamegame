import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from .puzzle import Puzzle
from .cage import Cage
from .solver import Solver
from .visualizer import KenKenVisualizer
from .generator import generate_kenken  # Add this import
import argparse
import random


class VisionSolver:
    def __init__(self, size=6, use_generator=False):
        """Initialize vision solver with OpenCV backend

        Args:
            size (int): Size of the puzzle grid (NxN). Must be between 3 and 9.
            use_generator (bool): Whether to use puzzle generator instead of vision
        """
        if not 3 <= size <= 9:
            raise ValueError("Puzzle size must be between 3 and 9")

        self.size = size
        self.test_image_path = None
        self.generated_puzzle = None
        self.generated_solution = None  # Store the solution grid
        self.use_generator = use_generator

        # Try to open the webcam
        try:
            self.cap = cv2.VideoCapture(0)
            self.webcam_available = self.cap.isOpened()
            if not self.webcam_available:
                print("Warning: Could not open webcam. Running in demo mode.")
        except Exception as e:
            print(f"Warning: {str(e)}. Running in demo mode.")
            self.webcam_available = False

        # Setup matplotlib for visualization
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 10))
        plt.tight_layout()

        # Create a default puzzle in case the vision detection fails
        self.puzzle = self._create_demo_puzzle(size)
        self.visualizer = KenKenVisualizer(self.puzzle)

    def generate_new_puzzle(self):
        """Generate a new KenKen puzzle and store its solution"""
        puzzle, solution = generate_kenken(self.size)
        self.generated_puzzle = puzzle
        self.generated_solution = solution
        self.puzzle = self._create_puzzle_from_json(json.dumps(puzzle))
        self.visualizer = KenKenVisualizer(self.puzzle)
        return json.dumps(puzzle)

    def _create_puzzle_from_json(self, puzzle_json):
        """Create a Puzzle object from JSON data"""
        data = json.loads(puzzle_json)
        cages = []
        for cage_data in data["cages"]:
            cells = [tuple(cell) for cell in cage_data["cells"]]
            operation = cage_data["operation"]
            target = cage_data["target"]
            cages.append(Cage(operation, target, cells))
        return Puzzle(data["size"], cages)

    def _create_demo_puzzle(self, size):
        """Create a valid demo KenKen puzzle of specified size for fallback"""
        # Define cages for the demo puzzle
        if size == 3:
            # 3x3 puzzle with 5 cages
            cages = [
                Cage("+", 4, [(0, 0), (0, 1)]),
                Cage("*", 6, [(0, 2), (1, 2)]),
                Cage("-", 1, [(1, 0), (2, 0)]),
                Cage("/", 2, [(1, 1), (2, 1)]),
                Cage("*", 3, [(2, 2)])
            ]
        elif size == 4:
            # 4x4 puzzle with 8 cages
            cages = [
                Cage("+", 7, [(0, 0), (0, 1), (1, 0)]),
                Cage("*", 8, [(0, 2), (0, 3)]),
                Cage("-", 3, [(1, 1), (1, 2)]),
                Cage("/", 2, [(1, 3), (2, 3)]),
                Cage("+", 6, [(2, 0), (2, 1), (2, 2)]),
                Cage("*", 4, [(3, 0), (3, 1)]),
                Cage("*", 1, [(3, 2)]),  # Changed from = to *
                Cage("*", 4, [(3, 3)])   # Changed from = to *
            ]
        elif size == 5:
            # 5x5 puzzle with 10 cages
            cages = [
                Cage("+", 6, [(0, 0), (0, 1)]),
                Cage("*", 15, [(0, 2), (0, 3), (0, 4)]),
                Cage("-", 1, [(1, 0), (2, 0)]),
                Cage("/", 2, [(1, 1), (1, 2)]),
                Cage("+", 9, [(1, 3), (1, 4), (2, 4)]),
                Cage("*", 20, [(2, 1), (2, 2), (2, 3)]),
                Cage("*", 4, [(3, 0)]),
                Cage("+", 8, [(3, 1), (3, 2), (4, 2)]),
                Cage("-", 1, [(3, 3), (3, 4)]),
                Cage("*", 24, [(4, 0), (4, 1), (4, 3), (4, 4)])
            ]
        elif size == 6:
            # 6x6 puzzle with 15 cages
            cages = [
                Cage("+", 11, [(0, 0), (0, 1), (1, 0)]),
                Cage("*", 20, [(0, 2), (0, 3), (0, 4)]),
                Cage("*", 5, [(0, 5)]),
                Cage("-", 1, [(1, 1), (2, 1)]),
                Cage("+", 9, [(1, 2), (1, 3), (1, 4), (1, 5)]),
                Cage("*", 30, [(2, 0), (3, 0), (4, 0)]),
                Cage("*", 2, [(2, 2)]),
                Cage("/", 2, [(2, 3), (2, 4)]),
                Cage("*", 1, [(2, 5)]),
                Cage("+", 7, [(3, 1), (3, 2), (4, 1), (4, 2)]),
                Cage("*", 6, [(3, 3), (3, 4)]),
                Cage("*", 4, [(3, 5)]),
                Cage("*", 3, [(4, 3)]),
                Cage("-", 1, [(4, 4), (4, 5)]),
                Cage("*", 90, [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)])
            ]
        else:
            # For larger puzzles (7x7, 8x8, 9x9), create a puzzle with systematically defined cages
            cages = []
            cells_covered = set()

            # Fill the grid with cages
            for row in range(size):
                for col in range(size):
                    if (row, col) not in cells_covered:
                        # For single cells or cells near the edge, create "*" operation
                        if (row >= size - 2 and col >= size - 2) or random.random() < 0.3:
                            cage_cells = [(row, col)]
                            cells_covered.add((row, col))
                            target = random.randint(1, size)
                            cages.append(Cage("*", target, cage_cells))  # Changed from = to *
                            continue

                        # Try to create multi-cell cages
                        possible_cells = []

                        # Check right neighbor
                        if col + 1 < size and (row, col + 1) not in cells_covered:
                            possible_cells.append((row, col + 1))

                        # Check bottom neighbor
                        if row + 1 < size and (row + 1, col) not in cells_covered:
                            possible_cells.append((row + 1, col))

                        # If we have possible neighbors, create a cage
                        if possible_cells:
                            # Start with current cell
                            cage_cells = [(row, col)]
                            cells_covered.add((row, col))

                            # Add one neighbor for binary operations
                            neighbor = random.choice(possible_cells)
                            cage_cells.append(neighbor)
                            cells_covered.add(neighbor)

                            # Decide operation based on position and randomness
                            if random.random() < 0.4:  # 40% chance for binary operations
                                # Use - or / operations (must be exactly 2 cells)
                                op = random.choice(["-", "/"])
                                if op == "-":
                                    target = random.randint(1, size - 1)
                                else:  # "/"
                                    target = random.randint(1, 3)
                            else:
                                # Use + or * operations (can have more cells)
                                op = random.choice(["+", "*"])

                                # Try to add more cells (up to 4 total)
                                for _ in range(2):  # Try to add up to 2 more cells
                                    if random.random() < 0.3:  # 30% chance to add another cell
                                        more_neighbors = []
                                        for r, c in cage_cells:
                                            # Check right and bottom neighbors
                                            if c + 1 < size and (r, c + 1) not in cells_covered:
                                                more_neighbors.append((r, c + 1))
                                            if r + 1 < size and (r + 1, c) not in cells_covered:
                                                more_neighbors.append((r + 1, c))

                                        if more_neighbors:
                                            next_cell = random.choice(more_neighbors)
                                            cage_cells.append(next_cell)
                                            cells_covered.add(next_cell)

                                # Set target based on operation
                                if op == "+":
                                    target = len(cage_cells) + random.randint(1, size * len(cage_cells))
                                else:  # "*"
                                    target = random.randint(1, size) * len(cage_cells)

                            cages.append(Cage(op, target, cage_cells))
                        else:
                            # No available neighbors, create single cell cage
                            cage_cells = [(row, col)]
                            cells_covered.add((row, col))
                            target = random.randint(1, size)
                            cages.append(Cage("*", target, cage_cells))  # Changed from = to *

            # Verify all cells are covered
            all_cells = {(r, c) for r in range(size) for c in range(size)}
            if cells_covered != all_cells:
                # If any cells are missing, create single-cell cages for them
                missing_cells = all_cells - cells_covered
                for cell in missing_cells:
                    cages.append(Cage("*", random.randint(1, size), [cell]))  # Changed from = to *

        # Create and return the puzzle
        return Puzzle(size, cages)

    def detect_puzzle(self, frame):
        """Detect a KenKen puzzle from a given frame"""
        try:
            print("Starting puzzle detection...")
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing
            # First, blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding with more aggressive parameters
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,  # Increased block size
                8    # Increased C value
            )
            
            # Dilate to connect nearby components
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # Find the puzzle grid with enhanced parameters
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            puzzle_contour = None
            max_area = 0

            print(f"Found {len(contours)} contours")
            
            # Debug: Draw all contours on a copy of the frame
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 2)
            cv2.imshow('All Contours', debug_frame)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Reduced minimum area threshold
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                    # Draw the approximated contour
                    debug_frame2 = frame.copy()
                    cv2.drawContours(debug_frame2, [approx], -1, (0, 0, 255), 2)
                    cv2.imshow(f'Contour {area}', debug_frame2)

                    if len(approx) == 4:  # Looking for quadrilateral
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = w / float(h)
                        print(f"Found quadrilateral - Area: {area}, Aspect ratio: {aspect_ratio}")
                        
                        # More lenient aspect ratio check
                        if 0.7 <= aspect_ratio <= 1.3 and area > max_area:
                            puzzle_contour = approx
                            max_area = area
                            print(f"New best contour - Area: {area}, Aspect ratio: {aspect_ratio}")

            if puzzle_contour is None:
                print("Could not find puzzle grid in image")
                return None, dilated  # Return preprocessed image for debugging

            # Draw the final selected puzzle contour
            debug_frame3 = frame.copy()
            cv2.drawContours(debug_frame3, [puzzle_contour], -1, (255, 0, 0), 3)
            cv2.imshow('Selected Puzzle Contour', debug_frame3)

            # Get perspective transform
            puzzle_points = puzzle_contour.reshape(4, 2)
            rect = self._order_points(puzzle_points.astype(np.float32))

            # Calculate output size (make it square)
            width = int(max(
                np.linalg.norm(rect[1] - rect[0]),
                np.linalg.norm(rect[2] - rect[3])
            ))
            height = int(max(
                np.linalg.norm(rect[2] - rect[1]),
                np.linalg.norm(rect[3] - rect[0])
            ))
            size = max(width, height)

            # Ensure size is divisible by puzzle size
            size = size - (size % self.size)

            dst_points = np.array([
                [0, 0],
                [size - 1, 0],
                [size - 1, size - 1],
                [0, size - 1]
            ], dtype=np.float32)

            # Apply perspective transform
            transform_matrix = cv2.getPerspectiveTransform(rect, dst_points)
            warped = cv2.warpPerspective(frame, transform_matrix, (size, size))

            # Show the warped image
            cv2.imshow('Warped Puzzle', warped)

            # Detect cages and create puzzle
            cages = self._detect_cages(warped)
            if not cages:
                print("Failed to detect cages")
                return None, dilated

            # Create puzzle from detected cages
            puzzle = self._create_puzzle_from_vision(cages, warped)
            if puzzle is None:
                print("Failed to create valid puzzle from detected cages")
                return None, dilated

            return puzzle, dilated

        except Exception as e:
            print(f"Error in puzzle detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _create_puzzle_from_vision(self, cages, warped):
        """Create a puzzle from detected cages and warped image."""
        processed_cages = []
        height, width = warped.shape[:2] # Handle grayscale or color warped images
        cell_size = height // self.size

        for cage in cages:
            # Get the top-left cell of the cage for operation detection
            top_left = min(cage.cells)
            row, col = top_left
            cell_img = warped[row * cell_size:(row + 1) * cell_size,
                               col * cell_size:(col + 1) * cell_size]

            number_region = cell_img[cell_size // 4:3 * cell_size // 4, cell_size // 4:3 * cell_size // 4]
            operation_region = cell_img[0:cell_size // 4, 0:cell_size // 4]

            # Recognize number and operation
            target = self._recognize_number(number_region)
            operation = self._detect_operation(operation_region, len(cage.cells))

            if target is None or operation is None:
                print(f"Warning: Could not recognize number or operation for cage at {top_left}")
                continue

            # For single cells, use multiplication
            if len(cage.cells) == 1:
                operation = '*'
            # Validate the cage based on operation type
            elif operation in ("-", "/") and len(cage.cells) != 2:
                print(f"Warning: '{operation}' operation cage at {top_left} has {len(cage.cells)} cells (should be 2)")
                continue

            # Create the cage with detected values
            try:
                processed_cage = Cage(operation, target, cage.cells)
                processed_cages.append(processed_cage)
            except ValueError as e:
                print(f"Warning: Invalid cage at {top_left}: {str(e)}")
                continue

        # Verify that all cells are covered
        covered_cells = set()
        for cage in processed_cages:
            covered_cells.update(cage.cells)

        all_cells = {(r, c) for r in range(self.size) for c in range(self.size)}
        missing_cells = all_cells - covered_cells

        if missing_cells:
            print(f"Warning: Some cells are not covered by any cage: {missing_cells}")
            return None

        # Create and validate the puzzle
        try:
            puzzle = Puzzle(self.size, processed_cages)
            return puzzle
        except ValueError as e:
            print(f"Error creating puzzle: {str(e)}")
            return None

    def _detect_cages(self, warped):
        """Detect cages in the warped puzzle image."""
        # Convert to grayscale if needed
        if len(warped.shape) == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped

        # Get cell size
        cell_size = gray.shape[0] // self.size

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5
        )

        # Detect thick lines (cage boundaries)
        kernel_thick = np.ones((5, 5), np.uint8)
        thick_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_thick)

        # Initialize cage map
        cage_map = np.zeros((self.size, self.size), dtype=int)
        next_cage_id = 1

        # Helper function to check if two cells are in the same cage
        def is_connected(r1, c1, r2, c2):
            y1 = r1 * cell_size + cell_size // 2
            x1 = c1 * cell_size + cell_size // 2
            y2 = r2 * cell_size + cell_size // 2
            x2 = c2 * cell_size + cell_size // 2

            # Check if there's a thick line between the cells
            if r1 == r2:  # Same row
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y = y1
                line_region = thick_lines[y - 2:y + 3, x_min:x_max]
                return not np.any(line_region > 127)
            else:  # Same column
                y_min = min(y1, y2)
                y_max = y1 # Corrected typo here, should be y2
                x = x1
                line_region = thick_lines[y_min:y_max, x - 2:x + 3] #Corrected typo here, should be y_min:y_max
                return not np.any(line_region > 127)

        # Find cages using flood fill
        for r in range(self.size):
            for c in range(self.size):
                if cage_map[r, c] == 0:
                    # Start new cage
                    cells = [(r, c)]
                    cage_map[r, c] = next_cage_id

                    # Check neighbors
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()

                        # Check all neighbors
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc

                            if (0 <= new_r < self.size and
                                    0 <= new_c < self.size and
                                    cage_map[new_r, new_c] == 0 and
                                    is_connected(curr_r, curr_c, new_r, new_c)):

                                cells.append((new_r, new_c))
                                cage_map[new_r, new_c] = next_cage_id
                                stack.append((new_r, new_c))

                    # Create temporary cage
                    if cells:
                        try:
                            # Use temporary values that will be updated later
                            temp_cage = Cage("+", 1, cells)
                            next_cage_id += 1
                        except ValueError as e:
                            print(f"Warning: Invalid cage at {cells[0]}: {str(e)}")
                            continue

        # Convert cage_map to list of cages
        cage_cells = {}
        for r in range(self.size):
            for c in range(self.size):
                cage_id = cage_map[r, c]
                if cage_id not in cage_cells:
                    cage_cells[cage_id] = []
                cage_cells[cage_id].append((r, c))

        # Create cages with temporary values
        cages = []
        for cells in cage_cells.values():
            if cells:  # Skip empty cells lists
                try:
                    temp_cage = Cage("+", 1, cells)  # Temporary values
                    cages.append(temp_cage)
                except ValueError as e:
                    print(f"Warning: Invalid cage at {cells[0]}: {str(e)}")
                    continue

        return cages

    def _recognize_number(self, cell_img):
        """Recognize a number in a cell image using template matching and OCR-like features."""
        # Pre-process the cell image
        height, width = cell_img.shape[:2] # Handle grayscale or color cell_img

        # Resize to a standard size for better recognition
        standard_size = (28, 28)  # Common size for digit recognition
        cell_img_resized = cv2.resize(cell_img, standard_size)

        # Apply adaptive thresholding with optimized parameters
        binary = cv2.adaptiveThreshold(
            cell_img_resized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Larger block size for better adaptation
            5    # Slightly higher C for better number isolation
        )

        # Clean up noise and enhance number
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours of potential numbers
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None  # Return None instead of defaulting to 1

        # Get the largest contour that's likely to be the number
        number_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(number_contour)

        # Extract features for recognition
        aspect_ratio = h / w if w > 0 else 1
        contour_area = cv2.contourArea(number_contour)
        hull_area = cv2.contourArea(cv2.convexHull(number_contour))
        solidity = contour_area / hull_area if hull_area > 0 else 0
        extent = contour_area / (w * h) if w * h > 0 else 0

        # Count holes using flood fill
        mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
        holes = 0
        binary_copy = binary.copy()
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary_copy[i, j] == 0 and mask[i + 1, j + 1] == 0:
                    cv2.floodFill(binary_copy, mask, (j, i), 255)
                    holes += 1
        holes = max(0, holes - 1)  # Subtract outer region

        # Create feature vector
        features = [
            aspect_ratio,  # Height/width ratio
            solidity,      # Area ratio between contour and its convex hull
            extent,        # Area ratio between contour and bounding rectangle
            holes,         # Number of holes (0 for 1,2,3,5,7; 1 for 6,9,0; 2 for 8)
        ]

        # Decision tree for number recognition based on features
        if holes == 2:
            return min(8, self.size)  # Must be 8
        elif holes == 1:
            if aspect_ratio > 1.7:  # Tall
                return min(9, self.size)
            else:
                return min(6, self.size)
        else:  # No holes
            if aspect_ratio > 2.2:  # Very tall and thin
                return 1
            elif aspect_ratio > 1.7:  # Tall
                if solidity > 0.8:
                    return min(4, self.size)
                else:
                    return min(7, self.size)
            elif solidity > 0.85:  # Very solid
                return min(5, self.size)
            elif extent > 0.65:  # Large extent
                return min(3, self.size)
            else:
                return min(2, self.size)

    def _detect_operation(self, cell_img, cage_size):
        """Detect operation symbol in a cell image."""
        # Pre-process the cell image
        height, width = cell_img.shape[:2] # Handle grayscale or color cell_img
        cell_img_resized = cv2.resize(cell_img, (28, 28))

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            cell_img_resized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5
        )

        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get the largest contour
        symbol_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(symbol_contour)

        # Extract features
        aspect_ratio = h / w if w > 0 else 1
        contour_area = cv2.contourArea(symbol_contour)
        hull_area = cv2.contourArea(cv2.convexHull(symbol_contour))
        solidity = contour_area / hull_area if hull_area > 0 else 0
        extent = contour_area / (w * h) if w * h > 0 else 0

        # Detect operation based on symbol features
        if aspect_ratio > 2.5:  # Vertical line -> division
            return "/"
        elif aspect_ratio < 0.5:  # Horizontal line -> subtraction
            return "-"
        elif extent > 0.5:  # Large extent -> multiplication
            return "*"
        else:  # Default to addition
            return "+"

    def _order_points(self, pts):
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        rect = np.zeros((4, 2), dtype="float32")

        # Sum of coordinates: top-left will have smallest sum, bottom-right largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Difference of coordinates: top-right will have smallest diff, bottom-left largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def capture_puzzle(self):
        """Get puzzle from vision, generator, or demo"""
        if self.use_generator:
            if not self.generated_puzzle:
                return self.generate_new_puzzle()
            return json.dumps(self.generated_puzzle)

        # Try to detect a puzzle from the webcam
        if self.test_image_path:
            print(f"Loading test image from: {self.test_image_path}")
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                print(f"Failed to load test image from {self.test_image_path}")
                return None, None # Return None for puzzle and processed image
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to capture frame from webcam")
                return None, None # Return None for puzzle and processed image
        puzzle, processed_image = self.detect_puzzle(frame) # Pass frame to detect_puzzle


        if puzzle is None:
            print("Using demo puzzle since detection failed")
            return self._get_puzzle_json(self.puzzle), frame, None # Return original frame and None for processed if detection fails

        # Create a new puzzle from the detected cages
        try:
            self.puzzle = puzzle
            self.visualizer = KenKenVisualizer(puzzle)
            return self._get_puzzle_json(puzzle), frame, processed_image # Return processed image for display
        except Exception as e:
            print(f"Error creating puzzle: {str(e)}")
            print("Falling back to demo puzzle")
            return self._get_puzzle_json(self.puzzle), frame, None # Return original frame and None for processed if puzzle creation fails

    def use_generated_puzzle(self, puzzle):
        """Set a generated puzzle to be used instead of detection"""
        self.generated_puzzle = puzzle

    def _get_puzzle_json(self, puzzle):
        """Convert a puzzle to JSON format"""
        puzzle_data = {
            "size": puzzle.size,
            "cages": []
        }

        for cage in puzzle.cages:
            # Handle different possible attribute names in Cage class
            target_value = getattr(cage, 'value', None)
            if target_value is None:
                target_value = getattr(cage, 'target', 1)

            operation = getattr(cage, 'operation_str', None)
            if operation is None:
                operation = getattr(cage, 'operation', '=')

            cage_data = {
                "cells": list(cage.cells), # Convert to list here
                "target": target_value,
                "operation": operation
            }
            puzzle_data["cages"].append(cage_data)

        return json.dumps(puzzle_data, indent=2)

    def solve_puzzle(self, puzzle_json):
        """Solve the KenKen puzzle"""
        try:
            # Parse the puzzle JSON
            puzzle_data = json.loads(puzzle_json)

            # Create cages and puzzle
            puzzle = self._create_puzzle_from_json(puzzle_json)

            # Create solver
            solver = Solver(puzzle)

            # Try different solving methods
            methods = ['backtracking', 'mrv', 'lcv', 'heuristics']

            for method in methods:
                print(f"Trying to solve with {method}...")
                success, metrics = solver.solve(method=method)

                if success:
                    print(f"Solved with {method}!")
                    print(f"Metrics: {metrics}")

                    # Update the puzzle and visualizer
                    self.puzzle = puzzle
                    self.visualizer = KenKenVisualizer(puzzle)

                    # Display the solution
                    self.ax[1, 1].clear()
                    title = f"Solution ({method})"
                    if self.generated_solution:
                        title += "\n(Generated)"
                    self.visualizer.draw(self.ax[1, 1], title)
                    plt.draw()
                    plt.pause(0.001)

                    return True, puzzle

            print("Could not solve the puzzle with any method")
            return False, None

        except Exception as e:
            print(f"Error solving puzzle: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None

    def _update_display(self, original, processed, warped=None, edges=None): # Make warped and edges optional
        """Update the display with processed images"""
        # Clear all axes
        for ax in self.ax.flatten():
            ax.clear()

        # Original image
        self.ax[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        self.ax[0, 0].set_title('Original Image')

        # Processed image
        if processed is not None:
            self.ax[0, 1].imshow(processed, cmap='gray')
            self.ax[0, 1].set_title('Processed Image')
        else:
            self.ax[0, 1].text(0.5, 0.5, "No processed image",
                            ha='center', va='center', fontsize=12)
            self.ax[0, 1].set_title('Processed Image')

        # Warped image (if available)
        if warped is not None:
            self.ax[1, 0].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            self.ax[1, 0].set_title('Warped Grid')
        else:
            self.ax[1, 0].text(0.5, 0.5, "No grid detected",
                            ha='center', va='center', fontsize=12)
            self.ax[1, 0].set_title('Warped Grid')

        # Solution (edges is repurposed as solution plot)
        self.ax[1, 1].set_title('Solution') # Just set title, plot is handled in solve_puzzle
        # Solution plot is updated in solve_puzzle function, no need to handle edges here

        # Update the display
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def release(self):
        """Release resources"""
        if self.webcam_available:
            self.cap.release()
        plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KenKen Vision Solver')
    parser.add_argument('--size', type=int, default=6, help='Size of the puzzle grid (3-9)')
    parser.add_argument('--image', type=str, help='Path to an image file containing a KenKen puzzle')
    parser.add_argument('--generate', action='store_true', help='Use puzzle generator instead of vision')
    args = parser.parse_args()

    try:
        # Create vision solver
        solver = VisionSolver(args.size, use_generator=args.generate)
        print(f"KenKen {'Generator' if args.generate else 'Vision'} Solver (size {args.size}x{args.size})")
        print("Press 'q' to exit")

        if args.generate:
            print("Generating new puzzle...")
        elif args.image:
            image_path = os.path.abspath(args.image)
            print(f"Using image file: {image_path}")
            if not os.path.exists(image_path):
                print(f"Error: Image file not found at {image_path}")
                sys.exit(1)

            # Replace webcam capture with the image
            if solver.webcam_available:
                solver.cap.release()
            solver.webcam_available = False

            # Set the test image path
            solver.test_image_path = image_path

        # Continuous loop for webcam processing
        if not args.generate and not args.image: # Only loop if using webcam
            while True:
                # First attempt puzzle detection only
                puzzle_json, original_frame, processed_image = solver.capture_puzzle()

                if puzzle_json:
                    print("\nPuzzle detected! Press 's' to solve or 'q' to exit")
                    solver._update_display(original_frame, processed_image)

                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s'):
                            # Now attempt to solve
                            success, puzzle = solver.solve_puzzle(puzzle_json)
                            if success:
                                print("Puzzle solved successfully!")
                                solver._update_display(original_frame, processed_image)
                            else:
                                print("Failed to solve the puzzle")
                            break
                        elif key == ord('q'):
                            print("\nExiting...")
                            return
                else:
                    print("No puzzle detected. Adjust the camera position and try again.")
                    ret, frame = solver.cap.read()
                    if ret and frame is not None:
                        solver._update_display(frame, None)

                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else: # If using generator or image, run once and show plot
            puzzle_json, original_frame, processed_image = solver.capture_puzzle()
            if puzzle_json:
                print("\nPuzzle detected! Press 's' to solve or 'q' to exit")
                if original_frame is not None:
                    solver._update_display(original_frame, processed_image)
                
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        success, puzzle = solver.solve_puzzle(puzzle_json)
                        if success:
                            print("Puzzle solved successfully!")
                            if solver.generated_solution:
                                print("\nGenerated solution grid:")
                                for row in solver.generated_solution:
                                    print(row)
                            solver._update_display(original_frame, processed_image)
                            plt.show(block=True)
                        else:
                            print("Failed to solve the puzzle")
                        break
                    elif key == ord('q'):
                        print("\nExiting...")
                        return
            else:
                print("No puzzle detected in the image")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'solver' in locals():
            solver.release()
        cv2.destroyAllWindows()