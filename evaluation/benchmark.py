import time
from astar.traditional_astar import traditional_astar
from astar.ai_astar import AIAStar

def benchmark(grid, start, goal):
    """Compare traditional vs AI-enhanced A*"""
    results = {}
    
    # Traditional A*
    start_time = time.perf_counter()
    trad_path, trad_explored_set = traditional_astar(grid, start, goal)
    trad_time = time.perf_counter() - start_time
    
    results["traditional"] = {
        "path": trad_path,
        "explored_set": trad_explored_set,  # Actual set of nodes
        "explored_count": len(trad_explored_set),
        "time": trad_time,
        "path_length": len(trad_path) if trad_path else 0
    }
    
    # AI-enhanced A*
    ai_astar = AIAStar()
    start_time = time.perf_counter()
    ai_path, ai_explored_set = ai_astar.find_path(grid, start, goal)
    ai_time = time.perf_counter() - start_time
    
    results["ai"] = {
        "path": ai_path,
        "explored_set": ai_explored_set,  # Actual set of nodes
        "explored_count": len(ai_explored_set),
        "time": ai_time,
        "path_length": len(ai_path) if ai_path else 0
    }
    
    return results

def print_comparison(results):
    """Print performance comparison"""
    trad = results["traditional"]
    ai = results["ai"]
    
    print("\n" + "="*50)
    print("Performance Comparison: Traditional A* vs AI-A*")
    print("="*50)
    print(f"{'Metric':<20} | {'Traditional A*':<15} | {'AI-A*':<10} | Improvement")
    print(f"{'Path Length':<20} | {trad['path_length']:<15} | {ai['path_length']:<10} | "
          f"{'Same' if trad['path_length'] == ai['path_length'] else 'DIFFERENT!'}")
    print(f"{'Nodes Explored':<20} | {trad['explored_count']:<15} | {ai['explored_count']:<10} | "
          f"{trad['explored_count']/ai['explored_count']:.1f}x fewer")
    print(f"{'Time (ms)':<20} | {trad['time']*1000:<15.2f} | {ai['time']*1000:<10.2f} | "
          f"{trad['time']/ai['time']:.1f}x faster")
    print("="*50)
    
    # Quality check
    if trad['path_length'] != ai['path_length']:
        print("\n⚠️ WARNING: Path lengths differ! AI may be finding suboptimal paths")
    elif ai['path_length'] > 0:
        print(f"✅ AI found the same optimal path ({ai['path_length']} steps)")