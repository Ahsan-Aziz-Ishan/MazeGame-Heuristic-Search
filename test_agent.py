import unittest
from maze_game import MazeGame, State
from agent import SearchAgent, Node

class TestSearchAgent(unittest.TestCase):
    def setUp(self):
        # Create a simple 3x3 maze for testing
        self.maze_size = 3
        self.game = MazeGame(size=self.maze_size, obstacle_ratio=0.0, seed=42)
        self.agent = SearchAgent(self.game)
        self.start_state = self.game.get_state((0, 0))

    def test_agent_initialization(self):
        """Test if agent is initialized correctly"""
        self.assertIsInstance(self.agent.game, MazeGame)

    def test_get_path(self):
        """Test path reconstruction"""
        # Create a simple path of nodes
        state1 = self.game.get_state((0, 0))
        state2 = self.game.get_state((0, 1))
        state3 = self.game.get_state((1, 1))
        
        node1 = Node(state1, None, 0, 0)
        node2 = Node(state2, node1, 1, 0)
        node3 = Node(state3, node2, 2, 0)
        
        path = self.agent.get_path(node3)
        
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0].position, (0, 0))
        self.assertEqual(path[1].position, (0, 1))
        self.assertEqual(path[2].position, (1, 1))

    def test_best_first_search(self):
        """Test if best first search finds a path"""
        path, metrics = self.agent.best_first_search(self.start_state)
        
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0].position, (0, 0))
        self.assertEqual(path[-1].position, (self.maze_size-1, self.maze_size-1))
        
        # Check metrics
        self.assertIn('nodes_expanded', metrics)
        self.assertIn('max_frontier', metrics)

    def test_astar_search(self):
        """Test if A* search finds a path"""
        path, metrics = self.agent.astar_search(self.start_state)
        
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0].position, (0, 0))
        self.assertEqual(path[-1].position, (self.maze_size-1, self.maze_size-1))

    def test_greedy_best_first_search(self):
        """Test if greedy best first search finds a path"""
        path, metrics = self.agent.greedy_best_first_search(self.start_state)
        
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0].position, (0, 0))
        self.assertEqual(path[-1].position, (self.maze_size-1, self.maze_size-1))

    def test_uniform_cost_search(self):
        """Test if uniform cost search finds a path"""
        path, metrics = self.agent.uniform_cost_search(self.start_state)
        
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0].position, (0, 0))
        self.assertEqual(path[-1].position, (self.maze_size-1, self.maze_size-1))

    def test_node_comparison(self):
        """Test if nodes are compared correctly"""
        node1 = Node(self.start_state, None, 1, 2)  # f_cost = 3
        node2 = Node(self.start_state, None, 2, 2)  # f_cost = 4
        
        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)

if __name__ == '__main__':
    unittest.main()
