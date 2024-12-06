from maze_game import MazeGame, State
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import heapq

@dataclass
class Node:
    state: State
    parent: Optional['Node']
    g_cost: float  # Cost from start to current node
    h_cost: float  # Heuristic estimate from current to goal
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class SearchAgent:
    def __init__(self, game: MazeGame):
        self.game = game
        
    def get_path(self, node: Node) -> List[State]:
        """Reconstruct path from start to current node"""
        path = []
        current = node
        while current:
            path.append(current.state)
            current = current.parent
        return path[::-1]
    
    def best_first_search(self, start_state: State) -> Tuple[Optional[List[State]], Dict]:
        """
        Best-First Search implementation
        Returns: (path, metrics)
        """
        metrics = {'nodes_expanded': 0, 'max_frontier': 0}
        
        start_node = Node(start_state, None, 0, self.game.heuristic(start_state))
        frontier = [start_node]
        explored = set()
        
        while frontier:
            metrics['max_frontier'] = max(metrics['max_frontier'], len(frontier))
            current = heapq.heappop(frontier)
            
            if self.game.is_goal(current.state):
                return self.get_path(current), metrics
                
            state_tuple = (current.state.position, current.state.distances)
            if state_tuple in explored:
                continue
                
            explored.add(state_tuple)
            metrics['nodes_expanded'] += 1
            
            for neighbor_state in self.game.get_neighbors(current.state):
                if (neighbor_state.position, neighbor_state.distances) not in explored:
                    neighbor_node = Node(
                        neighbor_state,
                        current,
                        float('inf'),  # g_cost not used in Best-First
                        self.game.heuristic(neighbor_state)
                    )
                    heapq.heappush(frontier, neighbor_node)
                    
        return None, metrics
    
    def astar_search(self, start_state: State) -> Tuple[Optional[List[State]], Dict]:
        """
        A* Search implementation
        Returns: (path, metrics)
        """
        metrics = {'nodes_expanded': 0, 'max_frontier': 0}
        
        start_node = Node(start_state, None, 0, self.game.heuristic(start_state))
        frontier = [start_node]
        explored = set()
        g_scores = {(start_state.position, start_state.distances): 0}
        
        while frontier:
            metrics['max_frontier'] = max(metrics['max_frontier'], len(frontier))
            current = heapq.heappop(frontier)
            
            if self.game.is_goal(current.state):
                return self.get_path(current), metrics
                
            state_tuple = (current.state.position, current.state.distances)
            if state_tuple in explored:
                continue
                
            explored.add(state_tuple)
            metrics['nodes_expanded'] += 1
            
            for neighbor_state in self.game.get_neighbors(current.state):
                neighbor_tuple = (neighbor_state.position, neighbor_state.distances)
                tentative_g = g_scores[state_tuple] + self.game.get_cost(current.state, neighbor_state)
                
                if neighbor_tuple not in g_scores or tentative_g < g_scores[neighbor_tuple]:
                    g_scores[neighbor_tuple] = tentative_g
                    neighbor_node = Node(
                        neighbor_state,
                        current,
                        tentative_g,
                        self.game.heuristic(neighbor_state)
                    )
                    heapq.heappush(frontier, neighbor_node)
                    
        return None, metrics
    
    def greedy_best_first_search(self, start_state: State) -> Tuple[Optional[List[State]], Dict]:
        """
        Greedy Best-First Search implementation
        Similar to Best-First but with a different heuristic approach
        Returns: (path, metrics)
        """
        metrics = {'nodes_expanded': 0, 'max_frontier': 0}
        
        def greedy_heuristic(state: State) -> float:
            # Combine Manhattan distance with obstacle awareness
            manhattan = self.game.heuristic(state)
            # Add penalty for being close to obstacles
            obstacle_penalty = sum(1.0 / (d + 1) for d in state.distances)
            return manhattan + obstacle_penalty * 0.5
        
        start_node = Node(start_state, None, 0, greedy_heuristic(start_state))
        frontier = [start_node]
        explored = set()
        
        while frontier:
            metrics['max_frontier'] = max(metrics['max_frontier'], len(frontier))
            current = heapq.heappop(frontier)
            
            if self.game.is_goal(current.state):
                return self.get_path(current), metrics
                
            state_tuple = (current.state.position, current.state.distances)
            if state_tuple in explored:
                continue
                
            explored.add(state_tuple)
            metrics['nodes_expanded'] += 1
            
            for neighbor_state in self.game.get_neighbors(current.state):
                if (neighbor_state.position, neighbor_state.distances) not in explored:
                    neighbor_node = Node(
                        neighbor_state,
                        current,
                        float('inf'),
                        greedy_heuristic(neighbor_state)
                    )
                    heapq.heappush(frontier, neighbor_node)
                    
        return None, metrics
    
    def uniform_cost_search(self, start_state: State) -> Tuple[Optional[List[State]], Dict]:
        """
        Uniform Cost Search implementation
        Returns: (path, metrics)
        """
        metrics = {'nodes_expanded': 0, 'max_frontier': 0}
        
        start_node = Node(start_state, None, 0, 0)  # h_cost = 0 for UCS
        frontier = [start_node]
        explored = set()
        g_scores = {(start_state.position, start_state.distances): 0}
        
        while frontier:
            metrics['max_frontier'] = max(metrics['max_frontier'], len(frontier))
            current = heapq.heappop(frontier)
            
            if self.game.is_goal(current.state):
                return self.get_path(current), metrics
                
            state_tuple = (current.state.position, current.state.distances)
            if state_tuple in explored:
                continue
                
            explored.add(state_tuple)
            metrics['nodes_expanded'] += 1
            
            for neighbor_state in self.game.get_neighbors(current.state):
                neighbor_tuple = (neighbor_state.position, neighbor_state.distances)
                tentative_g = g_scores[state_tuple] + self.game.get_cost(current.state, neighbor_state)
                
                if neighbor_tuple not in g_scores or tentative_g < g_scores[neighbor_tuple]:
                    g_scores[neighbor_tuple] = tentative_g
                    neighbor_node = Node(
                        neighbor_state,
                        current,
                        tentative_g,
                        0  # h_cost = 0 for UCS
                    )
                    heapq.heappush(frontier, neighbor_node)
                    
        return None, metrics
