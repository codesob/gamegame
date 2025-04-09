# KenKen Puzzle Solver

This project implements solvers for KenKen puzzles with Pygame visualization.

## Features

*   Loads puzzles from JSON files (see `puzzles/README.md` for format).
*   Basic backtracking solver implemented.
*   Optional Pygame visualization of the solving process.
*   Displays puzzle grid and cages.
*   Computer vision capabilities for puzzle recognition (requires Tesseract OCR).

## Project Structure

*   `puzzles/`: Contains example puzzle files in JSON format.
*   `src/`: Core source code for the puzzle representation and solvers.
    *   `cage.py`: Represents a single cage constraint.
    *   `puzzle.py`: Represents the KenKen grid and puzzle state.
    *   `solver.py`: Contains the solving algorithms (currently backtracking).
    *   `visualizer.py`: Handles the Pygame visualization.
    *   `vision_solver.py`: Handles computer vision and OCR for puzzle recognition.
*   `tests/`: Unit tests for the components (basic structure provided).
*   `main.py`: Command-line interface to run the solver.
*   `requirements.txt`: Lists Python package dependencies.

## Installation

1.  Make sure you have Python 3.6+ installed.
2.  Install Tesseract OCR:
    *   **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
    *   **macOS**: `brew install tesseract`
    *   **Linux**: `sudo apt-get install tesseract-ocr`
3.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **With Visualization:**
    ```bash
    python main.py puzzles/4x4_easy.json --visualize
    ```
    *(Optional flags: `--delay MS` to change step speed, `--cell_size PX` for cell size)*

2.  **Without Visualization (Text Mode):**
    ```bash
    python main.py puzzles/4x4_easy.json
    ```

3.  **With Computer Vision:**
    ```bash
    python -m src.vision_solver --size 4
    ```
    *(Optional: specify puzzle size with `--size N` where N is between 3 and 9)*

4.  To run basic tests:
    ```bash
    python -m unittest discover tests
    ```

## Future Enhancements (TODO)

*   Implement more efficient solvers (Forward Checking, Arc Consistency).
*   Implement heuristics (MRV, LCV).
*   Add puzzle generation capabilities.
*   Develop a puzzle difficulty estimator.
*   Explore ML integration.
*   Add more comprehensive unit tests.