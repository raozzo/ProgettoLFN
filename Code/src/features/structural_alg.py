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
