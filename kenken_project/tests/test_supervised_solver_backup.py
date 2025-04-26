import unittest
import os
import json
from src.puzzle import Puzzle
from src.supervised_solver import SupervisedSolver
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSupervisedSolver(unittest.TestCase):
    def setUp(self):
        self.puzzles_dir = os.path.join(os.path.dirname(__file__), '..', 'puzzles')
        self.puzzle_files = [
            "kenken_3x3_20250420_164117.json",
            "kenken_4x4_20250420_191233.json",
            "kenken_5x5_20250420_191321.json"
        ]

    def load_puzzle_and_solution(self, filename):
        filepath = os.path.join(self.puzzles_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        puzzle_data = data.get('puzzle')
        solution = data.get('solution')
        puzzle = Puzzle.from_dict(puzzle_data)
        return puzzle, solution

    def test_decision_tree_on_real_puzzles(self):
        for filename in self.puzzle_files:
            with self.subTest(puzzle=filename):
                puzzle, solution = self.load_puzzle_and_solution(filename)
                solver = SupervisedSolver(puzzle, solution_grid=solution)
                solver.train_decision_tree()
                solved = solver.solve()
                self.assertEqual(len(solved), puzzle.size)
                self.assertEqual(len(solved[0]), puzzle.size)
                for row in solved:
                    for val in row:
                        self.assertTrue(0 <= val <= puzzle.size)

    def test_random_forest_on_real_puzzles(self):
        for filename in self.puzzle_files:
            with self.subTest(puzzle=filename):
                puzzle, solution = self.load_puzzle_and_solution(filename)
                solver = SupervisedSolver(puzzle, solution_grid=solution)
                solver.train_random_forest()
                solved = solver.solve()
                self.assertEqual(len(solved), puzzle.size)
                self.assertEqual(len(solved[0]), puzzle.size)
                for row in solved:
                    for val in row:
                        self.assertTrue(0 <= val <= puzzle.size)

    def test_kmeans_on_real_puzzles(self):
        for filename in self.puzzle_files:
            with self.subTest(puzzle=filename):
                puzzle, solution = self.load_puzzle_and_solution(filename)
                solver = SupervisedSolver(puzzle, solution_grid=solution)
                solver.train_kmeans(n_clusters=puzzle.size)
                solved = solver.solve()
                self.assertEqual(len(solved), puzzle.size)
                self.assertEqual(len(solved[0]), puzzle.size)
                for row in solved:
                    for val in row:
                        self.assertTrue(0 <= val < puzzle.size)

if __name__ == "__main__":
    unittest.main()
