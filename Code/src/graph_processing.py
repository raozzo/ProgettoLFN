import os
import pickle
import re
import networkx as nx

def parse_amazon_graph_data(file_path, target_groups):
    """
    Reads the raw file and returns:
    1. A dictionary of node metadata {ASIN: {data}}
    2. An adjacency dictionary {ASIN: [neighbors]}
    """
    metadata = {}
    adjacency = {}

    current_asin = None
    current_meta = {}

    # Regex compilate
    asin_pattern = re.compile(r"^ASIN:\s+(\w+)")
    group_pattern = re.compile(r"^group:\s+(.+)")
    salesrank_pattern = re.compile(r"^salesrank:\s+(\d+)")
    similar_pattern = re.compile(r"^similar:\s+\d+\s+(.+)")

    print(f"Starting reading {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # New product (starts with "Id:")
                if line.startswith("Id:"):
                    # If valid, save the previous product
                    if current_asin and current_meta.get("group") in target_groups:
                        metadata[current_asin] = current_meta
                    # Reset
                    current_asin = None
                    current_meta = {}

                elif line.startswith("ASIN:"):
                    match = asin_pattern.match(line)
                    if match:
                        current_asin = match.group(1)
                        current_meta["ASIN"] = current_asin

                elif line.startswith("title:"):
                    current_meta["title"] = line[6:].strip()

                elif line.startswith("group:"):
                    match = group_pattern.match(line)
                    if match:
                        current_meta["group"] = match.group(1)

                elif line.startswith("salesrank:"):
                    match = salesrank_pattern.match(line)
                    if match:
                        current_meta["salesrank"] = int(match.group(1))

                elif line.startswith("similar:"):
                    match = similar_pattern.match(line)
                    if match and current_asin:
                        neighbors = match.group(1).split()
                        adjacency[current_asin] = neighbors

            # Save the last product
            if current_asin and current_meta.get("group") in target_groups:
                metadata[current_asin] = current_meta

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return metadata, adjacency


def create_graph_pickle(input_path, output_path, target_groups=None):
    """
    1. Reads the raw data.
    2. Builds the graph.
    3. Extracts the GCC.
    4. Saves the pickle.

    Returns: The path of the generated output file.
    """

    if target_groups is None:
        target_groups = {"Book", "DVD", "Video", "Music"}

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found in {input_path}")
    else:

        # Create the output folder if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        nodes_dict, edges_dict = parse_amazon_graph_data(input_path, target_groups)
        print(f"Found {len(nodes_dict)} valid nodes")

        # Graph construction
        print("Building the directed NetworkX graph...")
        G = nx.DiGraph()

        # Add nodes
        for asin, data in nodes_dict.items():
            G.add_node(asin, **data)

        # Add edges
        edge_count = 0
        for u, neighbors in edges_dict.items():
            if u in nodes_dict:
                for v in neighbors:
                    if v in nodes_dict:
                        G.add_edge(u, v)
                        edge_count += 1

        print(f"Graph construction completed.")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")

        # Largest Weakly Connected Component
        print("Extracting the Largest Weakly Connected Component...")
        connected_components = sorted(
            nx.weakly_connected_components(G), key=len, reverse=True
        )

        if connected_components:
            giant_component_nodes = connected_components[0]
            G_giant = G.subgraph(giant_component_nodes).copy()

            print(f"LWCC extraction completed.")
            print(f"LWCC nodes: {G_giant.number_of_nodes()}")
            print(f"LWCC edges: {G_giant.number_of_edges()}")

            # Save the graph in .pickle format
            print(f"Saving to {output_path}...")
            with open(output_path, "wb") as f:
                pickle.dump(G_giant, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Done!")
        else:
            print(
                "Error: the graph seems empty or connected components are missing."
            )