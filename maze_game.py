import numpy as np
from typing import Tuple, List, Dict, Set
from dataclasses import dataclass

@dataclass
class State:
    position: Tuple[int, int]
    distances: Tuple[int, int, int, int]  # distances to obstacles (up, right, down, left)

class MazeGame:
    def __init__(self, size: int, obstacle_ratio: float = 0.3, seed: int = None):
        """
        Initialize maze with given size and obstacle ratio
        
        Args:
            size: Size of the square maze
            obstacle_ratio: Ratio of obstacles to total cells (0 to 1)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # Initialize maze with obstacles
        n_obstacles = int(size * size * obstacle_ratio)
        self.maze = np.zeros((size, size), dtype=int)
        obstacle_positions = set()
        
        # Ensure start and goal positions are free
        while len(obstacle_positions) < n_obstacles:
            pos = (np.random.randint(0, size), np.random.randint(0, size))
            if pos != self.start and pos != self.goal:
                obstacle_positions.add(pos)
                
        for pos in obstacle_positions:
            self.maze[pos] = 1
            
    def get_state(self, position: Tuple[int, int]) -> State:
        """Get the state representation at given position"""
        x, y = position
        
        # Calculate distances to obstacles in each direction
        up = 0
        for i in range(x-1, -1, -1):
            if self.maze[i, y] == 1:
                break
            up += 1
            
        right = 0
        for j in range(y+1, self.size):
            if self.maze[x, j] == 1:
                break
            right += 1
            
        down = 0
        for i in range(x+1, self.size):
            if self.maze[i, y] == 1:
                break
            down += 1
            
        left = 0
        for j in range(y-1, -1, -1):
            if self.maze[x, j] == 1:
                break
            left += 1
            
        return State(position=position, distances=(up, right, down, left))
    
    def get_neighbors(self, state: State) -> List[State]:
        """Get valid neighboring states"""
        x, y = state.position
        neighbors = []
        
        # Check all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            
            # Check if position is valid and not an obstacle
            if (0 <= new_x < self.size and 
                0 <= new_y < self.size and 
                self.maze[new_x, new_y] == 0):
                neighbors.append(self.get_state((new_x, new_y)))
                
        return neighbors
    
    def is_goal(self, state: State) -> bool:
        """Check if state is goal state"""
        return state.position == self.goal
    
    def heuristic(self, state: State) -> float:
        """
        Manhattan distance heuristic
        Note: This is admissible as it never overestimates the actual cost
        """
        x, y = state.position
        gx, gy = self.goal
        return abs(x - gx) + abs(y - gy)
    
    def get_cost(self, state1: State, state2: State) -> float:
        """Get cost of moving from state1 to state2"""
        # In this simple maze, all moves cost 1
        return 1.0
