import unittest
import numpy as np
from maze_game import MazeGame, State

class TestMazeGame(unittest.TestCase):
    def setUp(self):
        # Use a fixed seed for reproducibility
        self.maze_size = 5
        self.maze_game = MazeGame(size=self.maze_size, obstacle_ratio=0.2, seed=42)

    def test_initialization(self):
        """Test if maze is initialized correctly"""
        self.assertEqual(self.maze_game.size, self.maze_size)
        self.assertEqual(self.maze_game.start, (0, 0))
        self.assertEqual(self.maze_game.goal, (self.maze_size-1, self.maze_size-1))
        self.assertIsInstance(self.maze_game.maze, np.ndarray)
        self.assertEqual(self.maze_game.maze.shape, (self.maze_size, self.maze_size))

    def test_start_goal_are_free(self):
        """Test if start and goal positions are free of obstacles"""
        self.assertEqual(self.maze_game.maze[0, 0], 0)  # Start
        self.assertEqual(self.maze_game.maze[self.maze_size-1, self.maze_size-1], 0)  # Goal

    def test_get_state(self):
        """Test if get_state returns correct State object"""
        state = self.maze_game.get_state((0, 0))
        self.assertIsInstance(state, State)
        self.assertEqual(state.position, (0, 0))
        self.assertEqual(len(state.distances), 4)  # Should have distances in 4 directions

    def test_get_neighbors(self):
        """Test if get_neighbors returns valid neighboring states"""
        state = self.maze_game.get_state((1, 1))
        neighbors = self.maze_game.get_neighbors(state)
        
        # Check if all neighbors are State objects
        self.assertTrue(all(isinstance(n, State) for n in neighbors))
        
        # Check if all neighbors are adjacent
        for neighbor in neighbors:
            x1, y1 = state.position
            x2, y2 = neighbor.position
            manhattan_dist = abs(x1 - x2) + abs(y1 - y2)
            self.assertEqual(manhattan_dist, 1)

    def test_is_goal(self):
        """Test goal state detection"""
        non_goal_state = self.maze_game.get_state((0, 0))
        goal_state = self.maze_game.get_state((self.maze_size-1, self.maze_size-1))
        
        self.assertFalse(self.maze_game.is_goal(non_goal_state))
        self.assertTrue(self.maze_game.is_goal(goal_state))

    def test_heuristic(self):
        """Test if heuristic returns correct Manhattan distance"""
        state = self.maze_game.get_state((0, 0))
        expected_heuristic = 2 * (self.maze_size - 1)  # Manhattan distance to goal
        self.assertEqual(self.maze_game.heuristic(state), expected_heuristic)

    def test_get_cost(self):
        """Test if movement cost is calculated correctly"""
        state1 = self.maze_game.get_state((0, 0))
        state2 = self.maze_game.get_state((0, 1))
        self.assertEqual(self.maze_game.get_cost(state1, state2), 1.0)

if __name__ == '__main__':
    unittest.main()
