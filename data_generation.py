import json
import random
import numpy as np
from tqdm import tqdm
from astar.traditional_astar import traditional_astar
from astar.utils import (
    create_random_grid,
    generate_valid_start_goal,
    manhattan_distance,
    euclidean_distance  
)

def generate_dataset(num_samples=10000, grid_size=50, min_path_length=30, seed=42):
    """
    Generate training dataset from synthetic maps
    :param num_samples: total samples to generate
    :param grid_size: grid dimensions (grid_size x grid_size)
    :param min_path_length: minimum path length to include
    :param seed: random seed
    """
    dataset = []
    sample_count = 0
    
    with tqdm(total=num_samples, desc="Generating training data") as pbar:
        while sample_count < num_samples:
            grid = create_random_grid(grid_size, grid_size, obstacle_prob=0.3, seed=seed+sample_count)
            start, goal = generate_valid_start_goal(grid, min_distance=min_path_length)
            
            if start is None or goal is None:
                continue 
                
            path, explored = traditional_astar(grid, start, goal)
            if not path or len(path) < min_path_length:
                continue 
            
            for i, node in enumerate(path):
                g_value = i 
                h_star = len(path) - i - 1 
                manhattan = manhattan_distance(node, goal)
                euclidean = euclidean_distance(node, goal)
                
                dataset.append({
                    "node": node,
                    "goal": goal,
                    "g_value": g_value,
                    "h_star": h_star,
                    "manhattan": manhattan,
                    "euclidean": euclidean
                })
                sample_count += 1
                pbar.update(1)
                
                if sample_count >= num_samples:
                    break
    
    with open("data/train_paths.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nGenerated {len(dataset)} training samples")

if __name__ == "__main__":
    generate_dataset(
        num_samples=10000, 
        grid_size=50,
        min_path_length=30,
        seed=42
    )