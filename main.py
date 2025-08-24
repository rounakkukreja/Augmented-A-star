import numpy as np
from astar.utils import create_random_grid, generate_valid_start_goal
from evaluation.visualization import visualize_comparison
from evaluation.benchmark import benchmark, print_comparison

def main():
   
    GRID_SIZE = 50
    OBSTACLE_PROB = 0.35
    SEED = 42
    MIN_PATH_LENGTH = 40

    
    print("Generating test grid...")
    grid = create_random_grid(GRID_SIZE, GRID_SIZE, 
                            obstacle_prob=OBSTACLE_PROB, 
                            seed=SEED)
    
    
    start, goal = generate_valid_start_goal(grid, MIN_PATH_LENGTH)
    while start is None or goal is None:
        SEED += 1
        grid = create_random_grid(GRID_SIZE, GRID_SIZE, 
                                obstacle_prob=OBSTACLE_PROB, 
                                seed=SEED)
        start, goal = generate_valid_start_goal(grid, MIN_PATH_LENGTH)

    print(f"\nStart: {start}, Goal: {goal}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}, Obstacles: {OBSTACLE_PROB*100}%")

    
    print("\nRunning benchmarks...")
    results = benchmark(grid, start, goal)
    
    print_comparison(results)
    visualize_comparison(
        grid,
        results["traditional"]["path"],
        results["ai"]["path"],
        results["traditional"]["explored_set"],  
        results["ai"]["explored_set"]            
    )

if __name__ == "__main__":
    main()
