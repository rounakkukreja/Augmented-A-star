import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def visualize_comparison(grid, trad_path, ai_path, trad_explored, ai_explored):
    """
    Visualize comparison between traditional and AI A*
    
    Args:
        grid: 2D numpy array of the map
        trad_path: Traditional A* path
        ai_path: AI A* path  
        trad_explored: Set of nodes explored by traditional A*
        ai_explored: Set of nodes explored by AI A*
    """
    grid = np.array(grid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create colormap
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue', 'lightblue'])
    norm = plt.Normalize(0, 5)
    
    # Prepare visualization grids
    def prepare_vis_grid(path, explored):
        vis_grid = np.zeros_like(grid)
        vis_grid[grid == 1] = 1  # Obstacles
        
        # Mark explored nodes (if not path/start/goal)
        for node in explored:
            if vis_grid[node] == 0:
                vis_grid[node] = 5
                
        # Mark path
        if path:
            for node in path[1:-1]:
                vis_grid[node] = 4
            vis_grid[path[0]] = 2  # Start
            vis_grid[path[-1]] = 3  # Goal
        return vis_grid
    
    # Create visualization grids
    trad_vis = prepare_vis_grid(trad_path, trad_explored)
    ai_vis = prepare_vis_grid(ai_path, ai_explored)
    
    # Plot traditional A*
    ax1.imshow(trad_vis, cmap=cmap, norm=norm)
    ax1.set_title(f"Traditional A*\nExplored: {len(trad_explored)} nodes")
    ax1.axis('off')
    
    # Plot AI A*
    ax2.imshow(ai_vis, cmap=cmap, norm=norm)
    ax2.set_title(f"AI-Augmented A*\nExplored: {len(ai_explored)} nodes") 
    ax2.axis('off')
    
    # Create unified legend
    legend_elements = [
        Patch(facecolor='green', label='Start'),
        Patch(facecolor='red', label='Goal'),
        Patch(facecolor='blue', label='Path'),
        Patch(facecolor='lightblue', label='Explored'),
        Patch(facecolor='black', label='Obstacle'),
        Patch(facecolor='white', label='Free')
    ]
    
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              ncol=6,
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("evaluation/comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_heatmap(explored_set, filename):
    """Optional: Save exploration heatmap"""
    heatmap = np.zeros_like(grid)
    for node in explored_set:
        heatmap[node] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Exploration Density')
    plt.title('Node Exploration Heatmap')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()