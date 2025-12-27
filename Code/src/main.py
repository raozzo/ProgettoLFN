import pickle
import pandas as pd
import features

if __name__ == "__main__":

    # Graph retrieval
    graph_path = "../data/processed/amazon_graph.pickle"
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Structural features computation
    #betweenness_scores_df = features.get_betweenness_centrality(G)
    harmonic_scores_df = features.get_harmonic_centrality(G, version="CPU")
    #page_rank_df = features.get_pagerank(G)
    #clustering_coefficient_df = features.get_clustering_coefficient(G)

    # Embeddings

    # Clustering

    # ...


