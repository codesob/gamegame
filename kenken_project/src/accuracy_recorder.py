import csv
from pathlib import Path

class AccuracyRecorder:
    def __init__(self, filepath="accuracy_results.csv"):
        self.filepath = Path(filepath)
        # Write header if file does not exist
        if not self.filepath.exists():
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["model_type", "puzzle_size", "puzzle_file", "accuracy", "elapsed_time", "nodes_visited"])

    def record(self, model_type: str, puzzle_size: int, puzzle_file: str = "", accuracy: float = None, elapsed_time: float = None, nodes_visited: int = None):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [model_type, puzzle_size, puzzle_file, accuracy]
            if elapsed_time is not None:
                row.append(elapsed_time)
            else:
                row.append("")
            if nodes_visited is not None:
                row.append(nodes_visited)
            else:
                row.append("")
            writer.writerow(row)
