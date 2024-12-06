# Maze Runner with Heuristic Search

This project implements a Maze Runner game with various heuristic search algorithms including Best-First Search and A* Search.

## Project Structure
- `maze_game.py`: Contains the Game module for Maze Runner implementation
- `agent.py`: Contains the Agent module with different search algorithms
- `main.ipynb`: Jupyter notebook for running experiments and generating statistics
- `requirements.txt`: Project dependencies
- `test_maze_game.py`: Unit tests for the maze game implementation
- `test_agent.py`: Unit tests for the search algorithms

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the tests:
```bash
python -m pytest test_maze_game.py test_agent.py
```

3. Run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

## Game Rules
- The game is played on an n×n grid
- Start position is (0,0)
- Goal position is (n-1,n-1)
- The agent can only perceive:
  - Its current position
  - Distance to obstacles in 4 directions (up, down, left, right)
- Movement is restricted to 4 directions (up, down, left, right)

## Implemented Search Algorithms
1. Best-First Search
2. A* Search
3. Greedy Best-First Search
4. Uniform Cost Search

## Testing
The project includes comprehensive unit tests:
- `test_maze_game.py`: Tests maze generation, state representation, and game mechanics
- `test_agent.py`: Tests all search algorithms and path finding capabilities

Run tests with detailed output:
```bash
python -m pytest -v
