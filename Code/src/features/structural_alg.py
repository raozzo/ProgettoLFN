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