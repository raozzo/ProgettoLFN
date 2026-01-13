import os
import time
import pandas as pd
import networkx as nx

def get_node2vec_embeddings(G, version = 'cuda', embedding_dim=128,
                            walk_length=80, context_size=10, walks_per_node=10, p=1, q=2,
                            epochs_GPU=100, patience_GPU=3, batch_size_GPU=128):

    """
    Computes Node2Vec embeddings using either the CPU ('node2vec' library) or a GPU accelerated version
    ('PyTorch Geometric' library) of the node2vec algorithm.

    Args:
        G (nx.Graph): Input NetworkX graph.
        version (str): Version of the Node2Vec algorithm to use ('cpu' or 'cuda').
        embedding_dim (int): Dimension of the output embedding vectors.
        walk_length (int): Length of each random walk.
        context_size (int): Window size for the skip-gram model.
        walks_per_node (int): Number of random walks generated per node.
        p (float): Return parameter (likelihood of returning to the immediate source).
        q (float): In-out parameter (likelihood of moving away from the source).
        epochs_GPU (int): Maximum number of training epochs for (only used on GPU version).
        patience_GPU (int): Epochs to wait for loss improvement before early stopping for (only used on GPU version).
        batch_size_GPU (int): Number of nodes per training batch for (only used on GPU version).

    Returns:
        pd.DataFrame: A DataFrame containing the 'ASIN' (original node ID) and the
                      corresponding embedding vectors.
    """

    if version == 'cpu':
        return get_node2vec_emb_CPU(G, embedding_dim=embedding_dim, walk_length=walk_length, window = context_size,
                                    walks_per_node=walks_per_node, p=p, q=q)
    elif version == 'cuda':
        return get_node2vec_emb_GPU(G, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size,
                                    walks_per_node=walks_per_node, p=p, q=q, epochs=epochs_GPU, patience=patience_GPU,
                                    batch_size=batch_size_GPU)
    return None


def get_node2vec_emb_CPU(G, embedding_dim=128, walk_length=80, window=10,
                         walks_per_node=10, p=1, q=2):

    """
    Computes Node2Vec embeddings on the CPU using the standard `node2vec` library.

    Args:
        G (nx.Graph): Input NetworkX graph.
        embedding_dim (int): Dimension of the output embedding vectors.
        walk_length (int): Length of each random walk.
        window (int): Window size for the skip-gram model.
        walks_per_node (int): Number of random walks generated per node.
        p (float): Return parameter (likelihood of returning to the immediate source).
        q (float): In-out parameter (likelihood of moving away from the source).

    Returns:
        pd.DataFrame: DataFrame containing 'ASIN' (original ID) and embedding columns.
    """

    from node2vec import Node2Vec

    print("Starting embeddings computation on CPU using node2vec library")
    start_time = time.time()

    # Model configuration
    node2vec_model = Node2Vec(
        G,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=walks_per_node,
        workers=os.cpu_count(),
        p=p,
        q=q,
        quiet=False
    )

    # Model training
    # window: max nodes distance at which the algorith will try to predict relations
    # min_count: will consider also nodes that appear only 1 time
    model = node2vec_model.fit(window = window, min_count = 1, batch_words = 4)

    end_time = time.time()
    print(f"Total CPU time: {end_time - start_time:.2f} s")

    # Output DataFrame building
    df = pd.DataFrame(
        index=model.wv.index_to_key,
        data=model.wv.vectors
    )

    df.columns = [f"emb_{i}" for i in range(model.vector_size)]
    df = df.reset_index() # push the index (ASIN) to be a standard column
    df = df.rename(columns={'index': 'ASIN'}) # rename the "index" column

    return df

def get_node2vec_emb_GPU(G, embedding_dim=128,
                         walk_length=80, context_size=10, walks_per_node=10,
                         p=1, q=2, epochs=100, patience=3, batch_size=128):
    """
    Computes Node2Vec embeddings using GPU acceleration via PyTorch Geometric.

    This function preprocesses the input graph (attribute clearing and integer relabeling),
    trains the model using an early stopping mechanism, and re-maps the resulting vectors to
    the original node IDs.

    Args:
        G (nx.Graph): Input NetworkX graph.
        embedding_dim (int): Dimension of the output embedding vectors.
        walk_length (int): Length of each random walk.
        context_size (int): Window size for the skip-gram model.
        walks_per_node (int): Number of random walks generated per node.
        p (float): Return parameter (likelihood of returning to the immediate source).
        q (float): In-out parameter (likelihood of moving away from the source).
        epochs (int): Maximum number of training epochs.
        patience (int): Epochs to wait for loss improvement before early stopping.
        batch_size (int): Number of nodes per training batch.

    Returns:
        pd.DataFrame: A DataFrame containing the 'ASIN' (original node ID) and the
                      corresponding embedding vectors.
    """

    import torch
    from torch_geometric.nn import Node2Vec as torch_Node2Vec
    from torch_geometric.utils import from_networkx

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("CUDA available, using PyTorch on GPU.")
    if device == 'cpu':
        print("CUDA not available, using PyTorch on CPU.")

    start_time = time.time()

    # Remove attributes from the graph's nodes
    G_clean = G.copy()
    for n in G_clean.nodes():       # for each node's ID n in G_clean...
        G_clean.nodes[n].clear()    # access to the attributes dictionary

    # Create a graph copy with integer node's labels
    nodes_list = list(G_clean.nodes())
    node_mapping = {node: i for i, node in enumerate(nodes_list)}
    reverse_mapping = {i: node for i, node in enumerate(nodes_list)}
    G_int = nx.relabel_nodes(G_clean, node_mapping)

    # Convert the graph to PyTorch Geometric's Data object
    data = from_networkx(G_int)
    data = data.to(device)

    # Model configuration
    model = torch_Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True
    ).to(device)

    # Generates walks on-the-fly and set batches of 128 nodes
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)

    # set the appropriate optimizer since the matrix is sparse
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    # Training
    model.train()               # start training mode
    best_loss = float('inf')    # + infinity
    counter = 0

    print(f"Starting training")
    for epoch in range(epochs):
        total_loss = 0

        # for each nodes batch, read the pair of positive random walks and negative random walks
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch: {epoch+1:02d}, Loss: {total_loss / len(loader):.4f}")

        # Stopping logic
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0  # reset the counter since
        else:
            counter += 1
            print(f"   No improvement detected for {counter} epochs.")

        if counter >= patience:
            print(f"Stopping model training since there has been no improvement for {patience} epochs.")
            break

    end_time = time.time()
    print(f"Total GPU time: {end_time - start_time:.2f} s")

    # Embeddings extraction
    model.eval()                    # stop training mode
    with torch.no_grad():           # do not compute gradients
        z = model().cpu().numpy()   # bring values to CPU

    # Remap list index to the original node ID
    emb_dict = {nodes_list[i]: z[i] for i in range(len(nodes_list))}
    df = pd.DataFrame.from_dict(emb_dict, orient='index')

    df.columns = [f"emb_{i}" for i in range(df.shape[1])]
    df = df.reset_index() # push the index (ASIN) to be a standard column
    df = df.rename(columns={'index': 'ASIN'}) # rename the "index" column

    return df