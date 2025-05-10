# KenKen Puzzle Solver and Generator

## Overview

This project is a comprehensive KenKen puzzle application designed to generate, solve, and allow manual play of KenKen puzzles. KenKen is a mathematical and logical puzzle similar to Sudoku, where players fill a grid with numbers while satisfying arithmetic constraints in outlined cages.

The application features a user-friendly graphical interface built with Tkinter, supports multiple solving algorithms including classical constraint satisfaction methods and supervised machine learning models, and provides tools for puzzle generation and visualization.

## Features

- **Puzzle Generation:** Create KenKen puzzles of sizes ranging from 3x3 to 9x9 with valid solutions.
- **Puzzle Solving:** Solve puzzles using various algorithms:
  - Backtracking
  - Minimum Remaining Values (MRV)
  - Least Constraining Value (LCV)
  - Combined heuristics
  - Supervised solvers using Decision Tree, Random Forest, and KMeans clustering.
- **Manual Play:** Interactive GUI to play puzzles manually with input validation and solution checking.
- **Visualization:** Visualize the solving process and puzzle grids.
- **Performance Tracking:** Record solver accuracy, elapsed time, and nodes visited for analysis.
- **Extensible Design:** Modular codebase for easy extension and experimentation.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. Clone or download the repository:

```bash
git clone <repository-url>
cd kenken_project
```

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Dependencies

The project depends on several libraries including but not limited to:

- `pygame` for graphical components
- `scikit-learn`, `tensorflow`, `torch` for machine learning solvers
- `numpy`, `pandas`, `seaborn`, `matplotlib` for data processing and visualization
- `opencv-python`, `pytesseract`, `pymupdf` for image processing (optional)
- `joblib` for model persistence

## Usage

Run the main application GUI:

```bash
python main.py
```

### GUI Overview

The main window provides three primary options:

1. **Solve:**  
   - Select puzzle size (3x3 to 9x9).  
   - Choose a puzzle file from the `puzzles/` directory.  
   - Select a solving algorithm (backtracking, MRV, LCV, heuristics, or supervised solvers).  
   - View solver results including solution grid, accuracy (if solution available), and performance metrics.

2. **Generate:**  
   - Enter desired puzzle size (3 to 9).  
   - Generate a new KenKen puzzle with a valid solution.  
   - Puzzle and solution are saved automatically to the `puzzles/` directory.

3. **Play:**  
   - Select a puzzle to play manually.  
   - Input numbers into the grid with real-time validation.  
   - Validate your solution against the correct answer.  
   - Clear inputs and retry as needed.

### Command Line Usage (Advanced)

While the GUI is the primary interface, the modular codebase allows for programmatic puzzle generation, solving, and analysis via the modules in the `src/` directory.

## Project Structure

```
kenken_project/
│
├── main.py                  # Main application entry point with GUI
├── requirements.txt         # Python dependencies
├── accuracy_results.csv     # Records solver accuracy and performance metrics
├── puzzles/                 # JSON files of KenKen puzzles and solutions
├── pseudocode/              # Pseudocode and notes related to puzzle solving
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── accuracy_recorder.py
│   ├── cage.py
│   ├── generator.py
│   ├── model_comparison.png
│   ├── puzzle.py
│   ├── solver.py
│   ├── supervised_solver.py
│   ├── utils.py
│   ├── validate_grid.py
│   └── visualizer.py
├── tests/                   # Unit tests for various components
└── venvcw/                  # Virtual environment (optional)
```

## Troubleshooting

- Ensure all dependencies are installed correctly.
- If puzzles fail to load, verify JSON file integrity in the `puzzles/` directory.
- For issues with supervised solvers, confirm compatible versions of TensorFlow and PyTorch are installed.
- Use the terminal or command prompt to view error messages when running `main.py`.
