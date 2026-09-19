"""
SafeRoutePlanner — Graph Cache & Dataset Management
===================================================
Provides two-tier caching:
1. In-memory hot cache for instant response to repeated queries.
2. Persistent disk cache (.pickle) for fast restarts without Overpass API lag.
3. Pre-bundled benchmark datasets for core demonstration corridors (e.g., Mumbai).
"""

import os
import time
import pickle
import numpy as np

# Cache directory configuration
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
import tempfile

CACHE_DIR = os.path.join(tempfile.gettempdir(), "saferoute_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_memory_cache = {}
CACHE_TTL = 86400  # 24 hours


def get_grid_key(lat: float, lon: float) -> str:
    """Generate a discrete cache key based on a 0.5-degree spatial grid."""
    return f"{round(lat, 1)}_{round(lon, 1)}"


def is_point_within_graph(G, lat: float, lon: float, buffer_deg: float = 0.005) -> bool:
    """
    Check if a geographic coordinate falls reliably within the node bounds of graph G.
    """
    if G is None or len(G.nodes) == 0:
        return False
    ys = [G.nodes[n]["y"] for n in G.nodes]
    xs = [G.nodes[n]["x"] for n in G.nodes]
    min_lat, max_lat = min(ys) + buffer_deg, max(ys) - buffer_deg
    min_lon, max_lon = min(xs) + buffer_deg, max(xs) - buffer_deg
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)


def load_graph_from_disk(filepath: str):
    """Safely load a serialized NetworkX graph from disk."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cached graph from {filepath}: {e}")
        return None


def save_graph_to_disk(G, filepath: str):
    """Serialize and save a scored NetworkX graph to disk."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Graph successfully persisted to disk: {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save graph to {filepath}: {e}")


def get_bundled_benchmark_graph():
    """Load the pre-computed benchmark graph if available."""
    benchmark_path = os.path.join(DATA_DIR, "mumbai_benchmark.pickle")
    if os.path.exists(benchmark_path):
        return load_graph_from_disk(benchmark_path)
    return None


def get_or_build_graph(lat: float, lon: float, radius_meters: int = 6000, ox_module=None):
    """
    Retrieves scored road graph via hierarchical strategy:
    1. In-memory active cache
    2. Bundled benchmark graph (if coordinate is enclosed)
    3. Persistent disk cache
    4. Live OSMnx build + multi-criteria scoring
    """
    grid_key = f"{get_grid_key(lat, lon)}_{radius_meters}"
    now = time.time()

    # 1. Check in-memory cache
    if grid_key in _memory_cache:
        entry = _memory_cache[grid_key]
        if now - entry["timestamp"] < CACHE_TTL:
            return entry["G"]

    # 2. Check bundled benchmark dataset
    benchmark_G = get_bundled_benchmark_graph()
    if benchmark_G and is_point_within_graph(benchmark_G, lat, lon):
        _memory_cache[grid_key] = {"G": benchmark_G, "timestamp": now}
        return benchmark_G

    # 3. Check persistent disk cache
    disk_path = os.path.join(CACHE_DIR, f"graph_{grid_key}.pickle")
    cached_G = load_graph_from_disk(disk_path)
    if cached_G and is_point_within_graph(cached_G, lat, lon):
        _memory_cache[grid_key] = {"G": cached_G, "timestamp": now}
        return cached_G

    # 4. Live fetch from OSMnx and score
    import osmnx as ox
    if ox_module:
        ox = ox_module

    from backend.engine.scoring import score_road_network

    print(f"Cache miss for key {grid_key} ({lat:.4f}, {lon:.4f}, radius={radius_meters}m) — fetching OSM network...")
    G = ox.graph_from_point((lat, lon), dist=radius_meters, network_type="drive")
    G = score_road_network(G, lat, lon, ox_module=ox)

    # Cache in memory and on disk
    _memory_cache[grid_key] = {"G": G, "timestamp": now}
    save_graph_to_disk(G, disk_path)

    return G
