import math
import numpy as np
import random
from collections import deque

# Phase 1 Core Functions


def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_valid_node(grid, node):
    """Check if node is traversable and within grid bounds"""
    x, y = node
    return (0 <= x < len(grid) and 
            0 <= y < len(grid[0]) and 
            grid[x][y] == 0)

def get_neighbors(grid, node):
    """Get traversable neighbors (4-direction movement)"""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional
    neighbors = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if is_valid_node(grid, (nx, ny)):
            neighbors.append((nx, ny))
    return neighbors

# Phase 2 Enhancements


def create_random_grid(width=50, height=50, obstacle_prob=0.3, seed=None):
    """
    Create a random grid with obstacles
    :param width: grid width
    :param height: grid height
    :param obstacle_prob: probability of obstacle
    :param seed: random seed for reproducibility
    :return: 2D grid (0 = free, 1 = obstacle)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    grid = np.zeros((height, width), dtype=np.uint8)
    obstacles = np.random.choice([0, 1], size=(height, width), 
                                p=[1-obstacle_prob, obstacle_prob])
    grid[obstacles == 1] = 1
    return grid.tolist()

def is_connected(grid, start, goal):
    """
    Check if path exists between start and goal using BFS
    :param grid: 2D grid map
    :param start: (x, y) start position
    :param goal: (x, y) goal position
    :return: True if path exists, False otherwise
    """
    rows, cols = len(grid), len(grid[0])
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
            
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = node[0] + dx, node[1] + dy
            
            if (0 <= nx < rows and 0 <= ny < cols and
                grid[nx][ny] == 0 and 
                (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))
                
    return False

def generate_valid_start_goal(grid, min_distance=20):
    """
    Generate valid start and goal positions
    :param grid: 2D grid
    :param min_distance: minimum straight-line distance
    :return: (start, goal) tuple
    """
    height, width = len(grid), len(grid[0])
    valid_positions = [(r, c) for r in range(height) for c in range(width) 
                       if grid[r][c] == 0]
    
    if not valid_positions:
        return None, None
    
    while True:
        start = random.choice(valid_positions)
        goal = random.choice(valid_positions)
        
       
        if (start != goal and 
            manhattan_distance(start, goal) >= min_distance and
            is_connected(grid, start, goal)):
            return start, goal