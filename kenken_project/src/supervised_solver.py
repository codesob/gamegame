import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from .puzzle import Puzzle

from .utils import calculate_accuracy

class SupervisedSolver:
    def __init__(self, puzzle: Puzzle, solution_grid=None):
        self.puzzle = puzzle
        self.size = puzzle.size
        self.solution_grid = solution_grid
        self.model = None

    def extract_features_and_labels(self):
        """
        Extract features and labels from the puzzle grid for supervised learning.
        Uses solution_grid if provided, else uses puzzle grid.
        """
        X = []
        y = []
        for r in range(self.size):
            for c in range(self.size):
                features = [r, c]
                X.append(features)
                if self.solution_grid is not None:
                    y.append(self.solution_grid[r][c])
                else:
                    y.append(self.puzzle.get_cell_value(r, c))
        return np.array(X), np.array(y)

    def train_decision_tree(self):
        X, y = self.extract_features_and_labels()
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)

    def train_random_forest(self):
        X, y = self.extract_features_and_labels()
        self.model = RandomForestClassifier()
        self.model.fit(X, y)

    def train_kmeans(self, n_clusters=None):
        X, _ = self.extract_features_and_labels()
        if n_clusters is None:
            n_clusters = self.size  # default clusters
        self.model = KMeans(n_clusters=n_clusters)
        self.model.fit(X)

    def predict(self, row, col):
        if self.model is None:
            raise ValueError("Model not trained yet")
        features = np.array([[row, col]])
        if isinstance(self.model, KMeans):
            # For clustering, return cluster label
            return self.model.predict(features)[0]
        else:
            return self.model.predict(features)[0]

    def solve(self):
        """
        Example solve method that predicts values for all cells.
        """
        solved_grid = [[0]*self.size for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                solved_grid[r][c] = self.predict(r, c)
        return solved_grid

    def validate_solution(self, solver_instance) -> bool:
        """
        Validate the predicted solution grid using the provided Solver instance's validation method.
        """
        solved_grid = self.solve()
        is_valid = solver_instance.validate_solution(solved_grid)
        if is_valid:
            print("SupervisedSolver: Predicted solution is valid.")
        else:
            print("SupervisedSolver: Predicted solution is invalid.")
        return is_valid

    def calculate_accuracy(self):
        """
        Calculate accuracy of the model predictions against the true solution grid.
        Returns accuracy as a float between 0 and 1.
        """
        if self.solution_grid is None:
            raise ValueError("No solution grid provided for accuracy calculation.")
        predicted_grid = self.solve()
        return calculate_accuracy(self.solution_grid, predicted_grid)
