import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def create_grid(width=20, height=20, obstacle_prob=0.2):
    """Create a random grid with obstacles"""
    grid = np.zeros((height, width))
    obstacles = np.random.choice([0, 1], size=(height, width), p=[1-obstacle_prob, obstacle_prob])
    grid[obstacles == 1] = 1
    return grid

def get_neighbors(grid, node):
    """Get traversable neighbors (4-direction movement)"""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def astar(grid, start, goal):
    """A* pathfinding implementation"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}
    
    explored = set()
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        explored.add(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, explored
        
        for neighbor in get_neighbors(grid, current):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, explored  # No path found

def visualize(grid, path, explored, start, goal):
    """Create visualization of the pathfinding process"""
    # Create display grid: 0=free, 1=obstacle, 2=start, 3=goal, 4=path, 5=explored
    display = np.zeros_like(grid)
    display[grid == 1] = 1  # Obstacles
    
    # Mark explored nodes
    for node in explored:
        if display[node] == 0 and node != start and node != goal:
            display[node] = 5
    
    # Mark path
    if path:
        for node in path[1:-1]:  # Skip start and goal
            display[node] = 4
    
    # Mark start and goal
    display[start] = 2
    display[goal] = 3
    
    # Create colormap
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue', 'lightblue'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = plt.Normalize(vmin=0, vmax=5)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(display, cmap=cmap, norm=norm)
    
    # Create legend
    legend_labels = {
        'Start': 'green',
        'Goal': 'red',
        'Path': 'blue',
        'Explored': 'lightblue',
        'Obstacle': 'black',
        'Free': 'white'
    }
    patches = [plt.Rectangle((0,0),1,1, color=color) for color in legend_labels.values()]
    plt.legend(patches, legend_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("A* Pathfinding Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Create a 20x20 grid with 20% obstacles
    grid = create_grid(20, 20, 0.2)
    
    # Define start and goal positions
    start = (0, 0)
    goal = (19, 19)
    
    # Ensure start and goal are not obstacles
    grid[start] = 0
    grid[goal] = 0
    
    # Run A* algorithm
    path, explored = astar(grid, start, goal)
    
    # Print results
    if path:
        print(f"Found path with {len(path)} steps!")
        print(f"Explored {len(explored)} nodes")
    else:
        print("No path found!")
    
    # Visualize
    visualize(grid, path, explored, start, goal)

if __name__ == "__main__":
    main()