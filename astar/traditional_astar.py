#classic A star
import heapq
from .utils import manhattan_distance, get_neighbors

def traditional_astar(grid, start, goal, heuristic=manhattan_distance):
    rows = len(grid)
    cols = len(grid[0])
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    explored_nodes = set()
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        explored_nodes.add(current)
        
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored_nodes
        
        for neighbor in get_neighbors(grid, current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, explored_nodes  

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path