import networkx as nx
import random
import numpy as np


def get_pagerank(G, alpha=0.85):
    """
    Calcola il PageRank per tutti i nodi.

    Args:
        G (nx.DiGraph): Il grafo (diretto o non diretto).
        alpha (float): Damping factor (default 0.85).

    Returns:
        dict: Dizionario {nodo_id: score_pagerank}
    """
    print(f"Calcolo PageRank su {len(G)} nodi...")
    return nx.pagerank(G, alpha=alpha)


def get_clustering_coefficient(G):
    return nx.clustering(G)

def get_approx_betweenness(G, k=100000, seed=42):
    return nx.betweenness_centrality(G, k=k, normalized=True, seed=seed)

def get_harmonic_centrality(G, k=1000, seed=42):
    
    """
    Calculates Approximated Harmonic Centrality using k pivots.
    
    Formula: H(u) = sum(1 / dist(u, v)) for v != u.
    
    Instead of calculating distances to ALL nodes (slow), we calculate 
    distances to k random 'pivot' nodes and scale the result.
    
    Strategy for Directed Graphs:
    We need dist(u, pivot). To find this efficiently for all u at once,
    we run a BFS starting from the 'pivot' on the REVERSED graph.
    
    Args:
        G (nx.DiGraph): The input graph.
        k (int): Number of pivots (samples). k=1000 is usually a good balance.
        seed (int): Random seed for reproducibility.
        
    Returns:
        dict: {node_id: harmonic_score}
        
        
        return nx.harmonic_centrality(G)
    """
    print(f"🔹 Calculating Approximated Harmonic Centrality with k={k} pivots...")
    
    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Validation: k cannot be larger than N
    if k > n:
        k = n
        
    # 1. Select k random pivots
    pivots = random.sample(nodes, k)
    
    # 2. Prepare for reverse BFS (Crucial for directed graphs)
    if G.is_directed():
        G_rev = G.reverse()
    else:
        G_rev = G 
        
    # Dictionary to accumulate scores
    scores = {node: 0.0 for node in nodes}
    
    # 3. Main Loop: BFS from each pivot
    for i, pivot in enumerate(pivots):
        if i % 100 == 0 and i > 0:
            print(f"   Processed {i}/{k} pivots...")
            
        # Calculate distances FROM all nodes TO the pivot
        # In the reversed graph, this is BFS starting from the pivot.
        # lengths = {node_u: distance_from_u_to_pivot}
        lengths = nx.single_source_shortest_path_length(G_rev, pivot)
        
        for u, dist in lengths.items():
            if dist > 0:  # Ignore self-loop (dist=0)
                scores[u] += 1.0 / dist
                
    # 4. Normalization / Scaling
    # Since we only summed over k nodes, we multiply by N/k to estimate 
    # what the sum would be if we checked all N nodes.
    factor = n / k
    for node in scores:
        scores[node] *= factor
        
    print("✅ Calculation complete.")
    return scores