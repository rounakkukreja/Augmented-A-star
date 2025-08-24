# import heapq
# import math
# import numpy as np
# from model.model_utils import load_model

# class AIAStar:
#     def __init__(self):
#         # Load model and scaler
#         self.model, self.scaler = load_model()
        
#         # Pre-allocate feature array for batch prediction
#         self.feature_array = np.empty((100, 6))  # Batch size 100
#         self.current_batch = 0
#         self.cache = {}

#     def predict_batch(self):
#         """Process queued predictions in bulk"""
#         if self.current_batch == 0:
#             return []
        
#         # Scale and predict
#         batch_features = self.feature_array[:self.current_batch]
#         scaled = self.scaler.transform(batch_features)
#         predictions = self.model.predict(scaled)
        
#         # Store in cache
#         for i in range(self.current_batch):
#             node = tuple(self.feature_array[i, :2].astype(int))
#             goal = tuple(self.feature_array[i, 2:4].astype(int))
#             self.cache[(node, goal)] = predictions[i]
        
#         self.current_batch = 0
#         return predictions

#     def get_heuristic(self, node, goal):
#         """Batch-optimized heuristic calculation"""
#         cache_key = (node, goal)
        
#         # Check cache first
#         if cache_key in self.cache:
#             return self.cache[cache_key]
        
#         # Calculate distances
#         dx = abs(node[0] - goal[0])
#         dy = abs(node[1] - goal[1])
#         manhattan = dx + dy
#         euclidean = math.sqrt(dx**2 + dy**2)
        
#         # Queue for batch prediction
#         self.feature_array[self.current_batch] = [
#             node[0], node[1], goal[0], goal[1],
#             manhattan, euclidean
#         ]
#         self.current_batch += 1
        
#         # Process batch if full
#         if self.current_batch >= 100:
#             self.predict_batch()
            
#         # Return temporary estimate (will be updated)
#         return manhattan  # Fallback

#     def find_path(self, grid, start, goal):
#         """Optimized pathfinding with batch prediction"""
#         self.cache = {}
#         self.current_batch = 0
        
#         open_set = []
#         heapq.heappush(open_set, (0, start))
        
#         g_score = {start: 0}
#         f_score = {start: self.get_heuristic(start, goal)}
#         came_from = {}
#         explored = set()
        
#         while open_set:
#             current = heapq.heappop(open_set)[1]
#             explored.add(current)
            
#             if current == goal:
#                 # Process remaining predictions
#                 self.predict_batch()
#                 return self.reconstruct_path(came_from, current), explored
            
#             for neighbor in self.get_neighbors(grid, current):
#                 tentative_g = g_score[current] + 1
                
#                 if neighbor not in g_score or tentative_g < g_score[neighbor]:
#                     came_from[neighbor] = current
#                     g_score[neighbor] = tentative_g
#                     f_score[neighbor] = tentative_g + self.get_heuristic(neighbor, goal)
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
#         return None, explored

#     @staticmethod
#     def get_neighbors(grid, node):
#         """Optimized neighbor finding"""
#         x, y = node
#         neighbors = []
#         if x > 0 and grid[x-1][y] == 0:
#             neighbors.append((x-1, y))
#         if x < len(grid)-1 and grid[x+1][y] == 0:
#             neighbors.append((x+1, y))
#         if y > 0 and grid[x][y-1] == 0:
#             neighbors.append((x, y-1))
#         if y < len(grid[0])-1 and grid[x][y+1] == 0:
#             neighbors.append((x, y+1))
#         return neighbors

#     def reconstruct_path(self, came_from, current):
#         path = [current]
#         while current in came_from:
#             current = came_from[current]
#             path.append(current)
#         return path[::-1]
    



#SECOND APPROACH

import heapq
import math
from .utils import get_neighbors, is_valid_node
from model.model_utils import load_model

class AIAStar:
    def __init__(self):
        # Load trained model and scaler
        self.model, self.scaler = load_model()
        # Create prediction cache
        self.heuristic_cache = {}
    
    def predict_heuristic(self, node, goal):
        """Cached heuristic prediction"""
        cache_key = (node, goal)
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]
        
        # Calculate distances directly (no numpy)
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        manhattan = dx + dy
        euclidean = math.sqrt(dx**2 + dy**2)
        
        # Prepare feature vector
        features = [
            node[0], node[1], 
            goal[0], goal[1],
            manhattan,
            euclidean
        ]
        
        # Predict with AI model
        scaled_features = self.scaler.transform([features])
        prediction = self.model.predict(scaled_features)[0]
        
        # Cache result
        self.heuristic_cache[cache_key] = prediction
        return prediction
    
    def find_path(self, grid, start, goal):
        """Optimized pathfinding with caching"""
        # Clear cache for new search
        self.heuristic_cache = {}
        
        # Initialize search
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        g_score = {start: 0}
        f_score = {start: self.predict_heuristic(start, goal)}
        came_from = {}
        explored_nodes = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            explored_nodes.add(current)
            
            if current == goal:
                return self.reconstruct_path(came_from, current), explored_nodes
            
            for neighbor in get_neighbors(grid, current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.predict_heuristic(neighbor, goal)
                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, explored_nodes
    
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path