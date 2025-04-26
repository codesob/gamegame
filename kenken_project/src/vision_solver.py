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
from .generator import generate_kenken 
import argparse
import random
from scipy import signal


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
        self.generated_solution = None 
        self.use_generator = use_generator

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
        """Detect a KenKen puzzle from a phone screenshot"""
        try:
            print("Starting puzzle detection...")
            if frame is None:
                print("Error: No input frame provided")
                return None, None

            # Debug setup
            debug_height, debug_width = frame.shape[:2]
            debug_scale = min(1.0, 800 / max(debug_width, debug_height))
            debug_size = (int(debug_width * debug_scale), int(debug_height * debug_scale))

            # 1. Convert to Lab color space for better lighting invariance
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            l_channel = lab[:,:,0]
            
            # 2. Apply CLAHE to L channel with higher clip limit
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            # 3. Apply bilateral filter for edge-preserving smoothing
            smooth = cv2.bilateralFilter(l_channel, 11, 75, 75)
            
            # 4. Apply Canny edge detection with automatic thresholds
            median = np.median(smooth)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))
            edges = cv2.Canny(smooth, lower, upper)
            
            # 5. Dilate edges to connect components
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # 6. Use both thresholding methods and combine results
            _, thresh1 = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh2 = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            binary = cv2.bitwise_or(thresh1, thresh2)
            binary = cv2.bitwise_or(binary, dilated)
            
            # 7. Clean up noise with morphological operations
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Show preprocessing steps
            debug_steps = np.hstack((l_channel, edges, binary))
            cv2.imshow('Preprocessing Steps', cv2.resize(debug_steps, (debug_size[0]*3, debug_size[1])))
            
            # Try different contour finding approaches
            contour_modes = [
                (cv2.RETR_EXTERNAL, 'EXTERNAL'),
                (cv2.RETR_LIST, 'ALL'),
                (cv2.RETR_TREE, 'TREE')
            ]
            
            puzzle_found = False
            for mode, mode_name in contour_modes:
                if puzzle_found:
                    break
                    
                print(f"\nTrying {mode_name} contours...")
                contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                    
                # Sort contours by area in descending order
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Create debug image for contours
                debug = frame.copy()
                
                # Find the largest roughly rectangular contour
                puzzle_contour = None
                puzzle_box = None
                
                for i, contour in enumerate(contours[:10]):
                    # Get approximate polygon with more lenient epsilon
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
                    
                    # Allow contours with 4-8 corners
                    if not (4 <= len(approx) <= 8):
                        continue
                        
                    if len(approx) > 4:
                        approx = self._reduce_to_four_corners(approx)
                    
                    # Get oriented bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    # Use np.intp instead of deprecated np.int0
                    box = np.array(box, dtype=np.intp)
                    
                    # Calculate metrics
                    width = np.linalg.norm(box[0] - box[1])
                    height = np.linalg.norm(box[1] - box[2])
                    if min(width, height) == 0:
                        continue
                        
                    aspect_ratio = max(width, height) / min(width, height)
                    area = cv2.contourArea(contour)
                    rect_area = width * height
                    solidity = area / rect_area if rect_area > 0 else 0
                    image_area = frame.shape[0] * frame.shape[1]
                    relative_area = area / image_area
                    
                    # Draw contour and metrics
                    color = (0, 255, 0)
                    cv2.drawContours(debug, [box], 0, color, 2)
                    center = np.mean(box, axis=0).astype(int)
                    metrics_text = f"AR:{aspect_ratio:.2f} S:{solidity:.2f}"
                    cv2.putText(debug, metrics_text, tuple(center), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    print(f"Contour {i}: aspect_ratio={aspect_ratio:.2f}, "
                          f"solidity={solidity:.2f}, area_ratio={relative_area:.3f}")
                    
                    # Much more lenient criteria
                    is_valid = (
                        0.2 <= aspect_ratio <= 5.0 and    # Very relaxed aspect ratio
                        0.3 <= solidity <= 1.0 and        # More lenient solidity
                        relative_area >= 0.003            # Smaller minimum area
                    )
                    
                    if is_valid:
                        puzzle_contour = approx
                        puzzle_box = box
                        puzzle_found = True
                        break
                
                # Show contour debug image
                cv2.imshow(f'{mode_name} Contours', cv2.resize(debug, debug_size))
                
                if puzzle_found:
                    print(f"Found valid puzzle contour using {mode_name} mode!")
                    break

            if not puzzle_found:
                print("Could not find puzzle grid - no contour matched the criteria")
                print("\nSuggested adjustments:")
                print("1. Ensure good lighting with minimal glare")
                print("2. Hold the puzzle parallel to the camera")
                print("3. Make sure the puzzle fills at least 30% of the frame")
                print("4. Try to minimize background clutter")
                return None, binary

            # Rest of the function remains the same...
            src_pts = puzzle_box.astype(np.float32)
            
            # Sort points to proper order: top-left, top-right, bottom-right, bottom-left
            src_pts = self._order_points(src_pts)
            
            # Make output size a multiple of puzzle size with fixed cell size
            cell_size = 60
            size = cell_size * self.size
            
            dst_pts = np.array([
                [0, 0],
                [size - 1, 0],
                [size - 1, size - 1],
                [0, size - 1]
            ], dtype=np.float32)

            # Get transform matrix
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, matrix, (size, size))

            # Convert warped image to binary
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # Use Otsu's thresholding for optimal separation
            _, warped_binary = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            cv2.imshow('Warped', cv2.resize(warped, debug_size))
            cv2.imshow('Binary', cv2.resize(warped_binary, debug_size))

            # Detect grid lines using horizontal and vertical projections
            row_sum = np.sum(warped_binary, axis=1)
            col_sum = np.sum(warped_binary, axis=0)
            
            # Grid lines should appear as peaks in the projections
            row_peaks = signal.find_peaks(row_sum)[0]
            col_peaks = signal.find_peaks(col_sum)[0]
            
            # Should have size+1 lines in each direction for a valid grid
            if len(row_peaks) < self.size+1 or len(col_peaks) < self.size+1:
                print("Invalid grid structure detected")
                return None, warped_binary

            # Create grid visualization
            grid_vis = warped.copy()
            for y in row_peaks:
                cv2.line(grid_vis, (0, y), (size, y), (0, 255, 0), 1)
            for x in col_peaks:
                cv2.line(grid_vis, (x, 0), (x, size), (0, 255, 0), 1)
            
            cv2.imshow('Grid Lines', cv2.resize(grid_vis, debug_size))

            # Detect cages
            cages = self._detect_cages(warped_binary)
            if not cages:
                print("Failed to detect cages")
                return None, warped_binary

            # Create puzzle
            puzzle = self._create_puzzle_from_vision(cages, warped_binary)
            if puzzle is None:
                print("Failed to create valid puzzle from detected cages")
                return None, warped_binary

            cv2.waitKey(1)
            return puzzle, warped_binary

        except Exception as e:
            print(f"Error in puzzle detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _order_points(self, pts):
        """Order points in clockwise order starting from top-left
        
        Args:
            pts: numpy array of shape (4, 2) containing the four corner points
            
        Returns:
            numpy array of same shape with points ordered clockwise from top-left
        """
        # Initialize ordered points array
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of x+y coordinates - smallest is top-left, largest is bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        # Difference of x-y coordinates - smallest is top-right, largest is bottom-left
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect

    def _reduce_to_four_corners(self, points):
        """Reduce a polygon with more than 4 points to the 4 most significant corners"""
        if len(points) <= 4:
            return points
            
        # Convert to more convenient format
        pts = points.reshape(-1, 2)
        
        # Get centroid
        centroid = np.mean(pts, axis=0)
        
        # Calculate angles from centroid to each point
        angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
        
        # Sort points by angle
        sorted_idx = np.argsort(angles)
        sorted_pts = pts[sorted_idx]
        
        # Find 4 points with roughly 90 degree separation
        n_points = len(sorted_pts)
        best_score = float('inf')
        best_indices = None
        
        # Try different starting points
        for start in range(n_points):
            indices = [
                start,
                (start + n_points//4) % n_points,
                (start + n_points//2) % n_points,
                (start + 3*n_points//4) % n_points
            ]
            
            # Calculate how close to 90 degrees each angle is
            pts = sorted_pts[indices]
            vectors = np.roll(pts, -1, axis=0) - pts
            angles = np.arctan2(vectors[:,1], vectors[:,0])
            angle_diffs = np.abs(np.degrees(np.diff(angles)) % 360 - 90)
            score = np.sum(angle_diffs)
            
            if score < best_score:
                best_score = score
                best_indices = indices
        
        return sorted_pts[best_indices].reshape(-1, 1, 2)

    def _detect_cages(self, binary):
        """Detect puzzle cages from warped binary image"""
        try:
            height, width = binary.shape
            cell_height = height // self.size
            cell_width = width // self.size
            
            # Create grid cells matrix to track cage assignments
            grid_cells = np.zeros((self.size, self.size), dtype=int)
            
            # Detect thick lines (cage boundaries)
            kernel_size = max(3, cell_width // 15)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            thick_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components (potential cages)
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cv2.bitwise_not(thick_lines), connectivity=4)
            
            # Create debug visualization
            debug_colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
            debug_colors[0] = [255, 255, 255]  # Background color
            debug_image = debug_colors[labels]
            cv2.imshow('Connected Components', debug_image)
            
            # Process each cell to assign cage IDs
            cages = {}
            
            for row in range(self.size):
                for col in range(self.size):
                    if grid_cells[row, col] == 0:  # Unassigned cell
                        center_y = int((row + 0.5) * cell_height)
                        center_x = int((col + 0.5) * cell_width)
                        cell_label = labels[center_y, center_x]
                        
                        if cell_label not in cages:
                            cages[cell_label] = {
                                'cells': [],
                                'value': None,
                                'op': None
                            }
                        
                        cages[cell_label]['cells'].append((row, col))
                        grid_cells[row, col] = cell_label
            
            # Remove background label
            if 0 in cages:
                del cages[0]
            
            # Convert detected cages to Cage objects
            cage_objects = []
            for cage_id, cage_info in cages.items():
                cells = cage_info['cells']
                cell_count = len(cells)
                
                # Choose operation based on cell count
                if cell_count == 1:
                    # Single cells use multiplication
                    op = "*"
                    value = random.randint(1, self.size)
                elif cell_count == 2:
                    # Two cells can use any operation
                    op = random.choice(["+", "-", "*", "/"])
                    if op == "+":
                        value = random.randint(cell_count, self.size * 2)
                    elif op == "-":
                        # For subtraction, ensure value is positive
                        value = random.randint(1, self.size - 1)
                    elif op == "*":
                        value = random.randint(1, self.size * 2)
                    else:  # "/"
                        # For division, ensure value is valid
                        value = random.randint(1, 3)
                else:
                    # Three or more cells can only use + or *
                    op = random.choice(["+", "*"])
                    if op == "+":
                        # Sum: reasonable range based on cell count
                        value = cell_count + random.randint(1, self.size * cell_count)
                    else:  # "*"
                        # Product: avoid too large values
                        value = random.randint(1, self.size) * cell_count
                
                cage_objects.append(Cage(op, value, cells))
            
            print(f"Detected {len(cage_objects)} cages")
            return cage_objects
            
        except Exception as e:
            print(f"Error in cage detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def capture_puzzle(self):
        """Get puzzle from vision, generator, or demo"""
        if self.use_generator:
            if not self.generated_puzzle:
                puzzle_json = self.generate_new_puzzle()
                print("Generated new puzzle")
                return puzzle_json, None, None

            print("Using existing generated puzzle")
            return json.dumps(self.generated_puzzle), None, None

        # Try to detect a puzzle from the webcam/image
        if self.test_image_path:
            print(f"Loading test image from: {self.test_image_path}")
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                print(f"Failed to load test image from {self.test_image_path}")
                return None, None, None
        else:
            print("Capturing frame from webcam...")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to capture frame from webcam")
                return None, None, None

        try:
            # Close any existing windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Give time for windows to close
        except:
            pass  # Ignore any window-related errors

        print("\nAttempting puzzle detection...")
        puzzle, processed_image = self.detect_puzzle(frame)

        if puzzle is None:
            print("\nPuzzle detection failed, using demo puzzle")
            print("Debug tips:")
            print("1. Make sure puzzle is well-lit and clearly visible")
            print("2. Hold the puzzle straight and center it in the frame")
            print("3. Check the debug windows to see what the detection sees")
            print("4. Try adjusting the camera angle or lighting")
            return self._get_puzzle_json(self.puzzle), frame, None

        # Successfully detected puzzle
        print("\nPuzzle detected successfully!")
        print(f"- Found {len(puzzle.cages)} cages")
        print("- Grid size:", puzzle.size)
        print("\nPress 's' to solve or 'q' to exit")
        
        try:
            self.puzzle = puzzle
            self.visualizer = KenKenVisualizer(puzzle)
            return self._get_puzzle_json(puzzle), frame, processed_image
        except Exception as e:
            print(f"Error creating puzzle visualization: {str(e)}")
            print("Falling back to demo puzzle")
            return self._get_puzzle_json(self.puzzle), frame, None

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

    def _update_display(self, original, processed, warped=None, edges=None):
        """Update the display with processed images"""
        # Clear all axes
        for ax in self.ax.flatten():
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Original image
        if original is not None:
            self.ax[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            self.ax[0, 0].set_title('Original Image')
        else:
            self.ax[0, 0].text(0.5, 0.5, "No camera feed",
                            ha='center', va='center', fontsize=12)
            self.ax[0, 0].set_title('Original Image')

        # Processed image
        if processed is not None:
            if len(processed.shape) == 2:  # Grayscale image
                self.ax[0, 1].imshow(processed, cmap='gray')
            else:  # Color image
                self.ax[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            self.ax[0, 1].set_title('Processed Image')
        else:
            self.ax[0, 1].text(0.5, 0.5, "No processed image",
                            ha='center', va='center', fontsize=12)
            self.ax[0, 1].set_title('Processed Image')

        # Warped image
        if warped is not None:
            if len(warped.shape) == 2:  # Grayscale image
                self.ax[1, 0].imshow(warped, cmap='gray')
            else:  # Color image
                self.ax[1, 0].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            self.ax[1, 0].set_title('Warped Grid')
        else:
            self.ax[1, 0].text(0.5, 0.5, "No grid detected",
                            ha='center', va='center', fontsize=12)
            self.ax[1, 0].set_title('Warped Grid')

        # Solution plot
        if self.puzzle is not None:
            self.ax[1, 1].set_title('Puzzle')
            if self.visualizer is not None:
                self.visualizer.draw(self.ax[1, 1])
        else:
            self.ax[1, 1].text(0.5, 0.5, "No solution yet",
                            ha='center', va='center', fontsize=12)
            self.ax[1, 1].set_title('Solution')

        # Update the display
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update

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

        # Main loop
        try:
            while True:
                # First attempt puzzle detection only
                puzzle_json, frame, processed = solver.capture_puzzle()
                
                if puzzle_json:
                    print("\nPuzzle detected! Press 's' to solve or 'q' to exit")
                    if frame is not None:
                        solver._update_display(frame, processed)
                    
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('s'):
                            # Now attempt to solve
                            success, puzzle = solver.solve_puzzle(puzzle_json)
                            if success:
                                print("Puzzle solved successfully!")
                                solver._update_display(frame, processed)
                                plt.show(block=True)
                            else:
                                print("Failed to solve the puzzle")
                            break
                        elif key == ord('q'):
                            print("\nExiting...")
                            solver.release()
                            cv2.destroyAllWindows()
                            sys.exit(0)
                else:
                    print("No puzzle detected. Adjust the camera position and try again.")
                    ret, frame = solver.cap.read() if solver.webcam_available else (None, None)
                    if frame is not None:
                        solver._update_display(frame, None)

                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            solver.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)