import csv
from pathlib import Path

class AccuracyRecorder:
    def __init__(self, accuracy_filepath="accuracy_results.csv", nodes_filepath="nodes_results.csv"):
        # Make sure we use the root project directory for the CSV files
        current_dir = Path(__file__).parent  # src directory
        project_root = current_dir.parent     # project root directory
        self.accuracy_filepath = project_root / accuracy_filepath
        self.nodes_filepath = project_root / nodes_filepath
        
        print(f"Initializing AccuracyRecorder with files:\n  Accuracy: {self.accuracy_filepath}\n  Nodes: {self.nodes_filepath}")
        
        # Write header to accuracy file if it does not exist
        if not self.accuracy_filepath.exists():
            print(f"Creating new accuracy CSV file at: {self.accuracy_filepath}")
            with open(self.accuracy_filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["model_type", "puzzle_size", "puzzle_file", "accuracy", "elapsed_time"])

        # Write header to nodes file if it does not exist
        if not self.nodes_filepath.exists():
            print(f"Creating new nodes CSV file at: {self.nodes_filepath}")
            with open(self.nodes_filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["method", "puzzle_size", "puzzle_file", "nodes_visited", "elapsed_time"])
    
    def record(self, method: str, puzzle_size: int, puzzle_file: str = "", accuracy: float = None, elapsed_time: float = None, nodes_visited: int = None):
        """Record metrics to appropriate files based on method type"""
        # ML methods go to accuracy_results.csv
        if method in ['Decision Tree', 'Random Forest', 'KMeans']:
            with open(self.accuracy_filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([method, puzzle_size, puzzle_file, accuracy, elapsed_time])
        
        # All methods' nodes visited data goes to nodes_results.csv
        with open(self.nodes_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([method, puzzle_size, puzzle_file, nodes_visited, elapsed_time])
