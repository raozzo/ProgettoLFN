import platform
import sys
import random
from collections import deque
import networkx as nx
import numpy as np
import scipy.sparse as sp

from collections import defaultdict

#from datasketch import HyperLogLog
import hashlib
#NOTE spostato cupy solo in se GPU 
import time
import copy
import pandas as pd
import gc

def get_platform():
    """
    Detects hardware in use and returns (platform_name, library_to_use ).
    """
    if sys.platform == "darwin" and platform.processor() == "arm":
        try:
            import mlx.core as mx
            return "mlx", mx
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu", torch
    except ImportError:
        pass
    return "cpu", np

def get_pagerank (G, alpha = 0.85, tol= 1e-5, max_iter =1000):
    '''
    Compute PageRank Score 
    graph is the loaded pickle file 
    '''
    
    # Extract Adjacency Matrix
    if hasattr(G, 'adjacency'):
        node_labels = list(G.nodes())
        # Convert to Scipy CSC matrix 
        adj = nx.to_scipy_sparse_array(G, format='csc', dtype=np.float32)
    else:
        adj = G.tocsc()
        node_labels = list(range(adj.shape))
    
    #number of nodes 
    n_nodes = adj.shape[0]
    
    if __name__ == "__main__":
        display(n_nodes)
    
    #we now can calcualate the out degree of each node (that given the adjaceny matrix 
    # is simply the sum along colums )√
    out_degrees = np.array(adj.sum(axis=1)).flatten()
    is_sink = (out_degrees == 0)                 #another check if a node is a sink
    
    # Normalize transition probabilities: P_ij = 1 / out_degree(j) (ignoring sinks)
    norm_out_degrees = np.where(is_sink, 1.0, out_degrees)
    adj.data = adj.data / norm_out_degrees[np.repeat(np.arange(n_nodes), np.diff(adj.indptr))]
    
    platform_name, engine = get_platform()
    
    # PageRank uses the transpose for 'Pull' aggregation: r_next = alpha * (adj.T @ r)
    P_matrix = adj.T.tocsc()
    
    #NOTE change it to mlx 
    if platform_name == "mlx":
        import mlx.core as mx
        # Pre-calculate target indices using NumPy (MLX repeat doesn't support arrays yet)
        counts = np.diff(P_matrix.indptr)
        targets_np = np.repeat(np.arange(n_nodes), counts)
        
        # Move arrays to Unified Memory
        indices = mx.array(P_matrix.indices)
        data = mx.array(P_matrix.data)
        targets = mx.array(targets_np)
        sink_mask = mx.array(is_sink.astype(np.float32))
        
        r = mx.full((n_nodes,), 1.0 / n_nodes)
        teleport_v = (1.0 - alpha) / n_nodes

        @mx.compile
        def update_step(r_prev):
            # Weighted values to sum
            weighted = data * r_prev[indices]
            
            # CORRECT MLX SYNTAX: Use.at.add() for parallel scatter-add
            res = mx.zeros((n_nodes,))
            res = res.at[targets].add(weighted)
            
            # Sink correction
            sink_mass = mx.sum(r_prev * sink_mask)
            return (alpha * (res + sink_mass / n_nodes)) + teleport_v

        for i in range(max_iter):
            r_next = update_step(r)
            mx.eval(r_next) # Materialize the lazy graph
            
            if mx.sum(mx.abs(r_next - r)) < tol:
                print(f"Converged at iteration {i}")
                break
            r = r_next
    
        r = r / mx.sum(r)
        final_ranks = np.array(r)
    
    
    elif platform_name == "gpu ":
        import torch
        device = torch.device("cuda")
        P_torch = torch.sparse_csc_tensor(
            torch.from_numpy(P_matrix.indptr).to(torch.int64),
            torch.from_numpy(P_matrix.indices).to(torch.int64),
            torch.from_numpy(P_matrix.data).to(torch.float32),
            size=(n_nodes, n_nodes)
        ).to(device)
        
        sinks = torch.from_numpy(is_sink).to(device)
        pr = torch.full((n_nodes, 1), 1.0 / n_nodes, device=device)
        teleport_v = (1.0 - alpha) / n_nodes

        for i in range(max_iter):
            #Sparse MAtrix_vect multiplication (more efficient)
            pr_next = torch.sparse.mm(P_torch, r)
            sink_mass = torch.sum(r[sinks])
            pr_next = alpha * (pr_next + sink_mass / n_nodes) + teleport_v
            
            if torch.norm(pr_next - pr, p=1) < tol:
                print(f"Converged at iteration {i}")
                break
            pr = pr_next
        #NOTE cpu() transfer from VRAM ----> RAM     
        final_ranks = pr.cpu().numpy().flatten()
    
    else:
        #we create the pagerank vector (initialized as every node as probability 1/n)
        pr = np.full(n_nodes, 1.0/n_nodes)
        teleport_const = (1.0 - alpha) / n_nodes
        for i in range(max_iter):
            
            pr_next = P_matrix.dot(pr)
            sink_mass = np.sum(pr[is_sink])
            pr_next = alpha*(pr_next +sink_mass / n_nodes)+teleport_const
            
            # check if maximum difference is lower than tol, if yes not much imprvement so we break
            if np.linalg.norm(pr_next - pr,1)<tol:
                break
            #else we update the pagerank vector 
            pr = pr_next
            
        final_ranks = pr
        
        
    return dict(zip(node_labels, final_ranks))


def get_clustering_coefficient(G, M=100000):
    """
    Calculate the clustering coefficient for all nodes (i.e., the fraction of triangles around each node).
    We make use of the definition adapted for directed graphs.
    Implementation exploits Reservoir sampling.

    Args:
        G (nx.DiGraph): The directed graph.
        M (int): edge memory size.

    Returns:
        dict: Dictionary {node_id: clustering_coefficient}
    """

    # Initialization
    S = [] # list of sampled edges
    S_adj = defaultdict(set) # adjacency list for sampled edges (needed to reduce time complexity for computing N_u_S and N_v_S from O(M) to O(1))
    t = 0 # time step
    t_S = {v: 0 for v in G.nodes} # number of triangles around each node initialized to 0
    d = {v: 0 for v in G.nodes} # degree of each node initialized to 0

    # Initialize clustering coefficient dictionary to 0
    cc = {v: 0.0 for v in G.nodes}

    # Transform G.edges into a stream of edges sigma
    sigma = list(G.edges)
    random.shuffle(sigma) # shuffle edges to simulate random stream

    # Process each edge in the stream (1 pass)
    for (u, v) in sigma:
        t += 1

        # Update degree counts of u and v
        d[u] += 1
        d[v] += 1

        # Neighborhood of u intesected v contained in S
        N_u_S = S_adj[u]
        N_v_S = S_adj[v]
        N_uv_S = N_u_S & N_v_S

        # as we are using Reservoir sampling, we need to take into account of the fact that p=M/t is changing over time
        # so we cannot simply multiply by p^2 at the end, as the probability of each triangle to be included in S is not the same for all triangles
        # instead, we increment the triangle counts directly considering the probability
        # Explanation: a triangle is detected only when its third edge (u,v) arrives.
        # At time t, this requires the first two edges to already reside in the reservoir S.
        # The probability of a specific edge being in S at time t is p_t = M/(t-1).
        # Thus, the probability of both previous edges being present is eta_t = (M/(t-1)) * ((M-1)/(t-2)).
        # To maintain an unbiased estimator E[t_S] = true_triangles, we increment by the reciprocal of this probability.
        if t > M:
            weight = (t-1)*(t-2)/(M*(M-1))
        else:
            weight = 1

        # For each c in N_uv_S, increment triangle count for u,v,c
        for c in N_uv_S:
            t_S[u] += weight
            t_S[v] += weight
            t_S[c] += weight

        # Reservoir sampling

        # CASE 1: add edge (u,v) to S if |S| < M
        if t <= M:
            # in order to count triangles we store egdes in S_adj as they were undirected
            S_adj[u].add(v)
            S_adj[v].add(u)
            S.append((u,v))

        # CASE 2: add edge (u,v) to S with probability M/t, 
        # replacing a random edge in S, chosen uniformly at random with probability 1/M
        else:
            prob = random.uniform(0,1)
            if prob <= M / t:
                index = random.randint(0, M-1)

                # remove old edge from adjacency list
                old_u, old_v = S[index]
                S_adj[old_u].discard(old_v)
                S_adj[old_v].discard(old_u)

                # add new edge to adjacency list
                S_adj[u].add(v)
                S_adj[v].add(u)

                S[index] = (u,v)

    # Compute clustering coefficient for each node
    for v in G.nodes:
        # Avoid division by zero
        if d[v] >= 2:
            cc[v] = (t_S[v]) / (d[v] * (d[v] -1)) # we should not multiply by 2 (directed graph adaptation)

            # correction for sampling ???
            # p = M / t
            # cc[v] /= p**2
        else:
            cc[v] = 0.0

    return cc

def bfs_directed(G, s, t):
    """
    Perform BFS on a directed graph from a source node.
    Adapted to handle directed edges and to store all shortest paths.
    
    Args:
        G (nx.DiGraph): The directed graph.
        s: Source node.
        t: Target node.
    
    Returns:
        dict: Dictionary {node_id: list of shortest paths from s to t}
    """

    # Initializiation
    distance = {v: float('inf') for v in G.nodes()}
    ID = {v: 0 for v in G.nodes()} # 0 = unvisited, 1 = visited
    preds = {v: [] for v in G.nodes()} # predecessors list to store all parents forming a SP
    target_found_at_layer = float('inf')
    
    # Visiting source node s
    distance[s] = 0
    ID[s] = 1
    L = {0 : [s]} # list of lists L_i containing nodes at distance i from s
    i = 0

    # while (!L_i.isEmpty() and i < target_found_at_layer) do:
    # while not(L[i].isEmpty()) and i < target_found_at_layer:
    while i in L and L[i] and i < target_found_at_layer:
        L[i+1] = []
        for v in L[i]:
            for w in G.successors(v):
                # CASE 1: first time visiting w
                if ID[w] == 0:
                    distance[w] = i + 1
                    preds[w].append(v)
                    ID[w] = 1 # mark as visited
                    L[i+1].append(w)
                    if w == t:
                        target_found_at_layer = i + 1
                # CASE 2: found another shortest path to w (at the same shortest distance)
                elif distance[w] == i + 1:
                    preds[w].append(v)
        i += 1

    # Reconstruct all SPs from s to t backtracking using predecessors
    all_SPs = [] # List to store all shortest paths from s to t
    current_path = [t] # current path being explored

    # Helper function for backtracking
    def FindPaths(curr):
        if curr == s:
            all_SPs.append(list(reversed(current_path)))  # Append reversed path
            return
        for pred in preds[curr]:
            current_path.append(pred)
            FindPaths(pred)
            current_path.pop()
    
    if distance[t] != float('inf'):
        FindPaths(t)

    return all_SPs

# PROBLEM: In the slides the professor says that we should sample k couples (s,t) from VxV with s != t and then find all shortest paths between them.
# In NewtorkX implementation instead, they sample k source nodes and compute shortest paths from each of them to all other nodes.
# => to understand if we prefer to stick to the slides or to NetworkX implementation.
def get_approx_betweenness(G, k=10, seed=42):
    """
    Calculate the approximate betweenness centrality for all nodes using sampling.
    Implementation of Brande's algorithm adapted for directed graphs
    
    Args:
        G (nx.DiGraph): The directed graph.
        k (int): Number of samples for approximation.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary {node_id: approximate_betweenness_score}
    """
    # Set random seed for reproducibility and sample k nodes
    random.seed(seed)

    V = list(G.nodes())
    if k > len(V):
        k = len(V)

    # Initialize betweenness centrality dictionary to zero
    b = {v: 0.0 for v in V}

    # In this graph it happens many times that t is not reachable from s, 
    # so everytime  we find out that there are no paths between s and t (up to a limit number of trials equal to 10)
    # we resample a new (s,t) couple and do not increment the counter i
    i = 0
    max_trials = 0
    while i < k:

        # Choose uniformly at random (s,t) from V with s != t
        """
        check = False
        while check == False:
            s = random.choice(V)
            t = random.choice(V)
            if s != t:
                check = True
        """
        s, t = random.sample(V, 2)

        # find the index of s and t in V
        s_index = V.index(s)
        t_index = V.index(t)

        # Find all shortest paths from s to all other nodes using BFS adapted for directed graphs and store them in a list P_st
        P_st = bfs_directed(G, s, t)
        if (len(P_st) == 0):
            print("No path found between s and t, iteration: ", i)
            if max_trials < 10:
                max_trials += 1
                continue
        print ("s = ", s_index, ", t = ", t_index, ", #SPs = ", len(P_st), ", len SP = ", max([len(p) for p in P_st]) if P_st else 0)
        print("------------------------------")

        # Choose a shortest path uniformly at random from P_st
        if P_st: # Check if there are any paths
            path = random.choice(P_st)

            # For each node v in the chosen path (but s and t), increment b[v] by 1/k
            for v in path:
                if v != s and v != t:
                    b[v] += 1.0 / k

        i += 1

    count = 0
    for v in b:
        if b[v] > 0:
            count += 1
    print(f"Total nodes with non-zero betweenness: {count} out of {len(V)}")

    return b
    #return nx.betweenness_centrality(G, k=k, normalized=True, seed=seed)

def get_harmonic_centrality(G, p=10, version="CPU_opt"):
    """
    Calcola la Harmonic Centrality approssimata usando HyperBall.

    :param G: Oggetto NetworkX DiGraph (350k nodi).
    :param p: Precisione dei registri (p=10 -> 2^10 registri = errore ~3%).
    :param version: Variante dell'implementazione con cui calcolare la Harmonic Centrality
                    (Valori accettati: "CPU", "GPU").
    """
    if version == "CPU":
        return harmonic_v2_CPU(G, p)
    elif version == "GPU":
        import cupy as cp
        return harmonic_v3_GPU(G, p)
    return None

def harmonic_v2_CPU(G, p=10):

    # DEFINIZIONE STRUTTURE BASE PER TRATTARE IL GRAFO NETWORKX
    print(f"--- FASE 1: Setup CPU e Hashing ---")

    G_rev = G.reverse()
    nodes = list(G_rev.nodes())
    n_nodes = len(nodes)

    m = 1 << p
    """
    Numero di registri che compongono ciascun contatore: m = 2^p
    """

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    """
    Mappatura nodo -> indice 0..N-1: dizionario del tipo (k, v) = (node, i), con i risultato dell'enumerazione dell' array di nodi
    """

    edges = np.array([(node_to_idx[u], node_to_idx[v]) for u, v in G_rev.edges()], dtype=np.int32)
    """
    Array di coppie (u_index, v_index), una per ogni edge del tipo (u, v) in G_rev
    """

    # DEFINIZIONE MATRICE DEI CONTATORI
    M_A = np.zeros((n_nodes, m), dtype=np.uint8)
    """
    Matrice [n_nodes x m] dei contatori;
    Dato che un registro deve contenere il numero di leading zeroes di un hash (64 bits), il massimo valore inseribile in ciascun registro sara' < 64 (dato che i primi b bits dell' hash servono a individuare il registro corretto tra gli m), quindi 1 byte (uint8) e' sufficiente.
    """

    # CALCOLO DEGLI HASH
    print("Calcolo hash iniziali su CPU...")
    for i, node in enumerate(nodes):
        # Hash del nodo (crea hash con algoritmo md5 e trasforma il risultato in stringa hex, poi converte in intero a partire da base 16 verso base 10)
        h = int(hashlib.md5(str(node).encode('utf8')).hexdigest(), 16)

        # AND binario tra l' hash h e il numero m-1 = (2^10 - 1) = 1023 = sequenza di zeri seguiti da 10 valori '1'
        # il risultato corrisponde agli ultimi 10 bit di h, che selezionano il registro in cui scrivere il numero di leading zeroes
        j = h & (m - 1)

        # Right shift per rimuovere da h gli ultimi 10 bit estratti in precedenza
        w = h >> p

        # Conteggio del numero di trailing zeroes della porzione di hash rimanente (+1)
        rho = 1
        while (w & 1) == 0 and rho < 32: # while l'ultimo bit di w è uno '0':
            w >>= 1
            rho += 1
        M_A[i, j] = rho

    # STRUTTURE DATI HYPERBALL E PREALLOCAZIONE
    M_B = M_A.copy()
    M_float = np.empty((n_nodes, m), dtype=np.float32)
    sources = edges[:, 1] # Array monodimensionale ottenuto prendendo SOLO (tutta) la colonna 1 di edges
    targets = edges[:, 0] # Array monodimensionale ottenuto prendendo SOLO (tutta) la colonna 0 di edges

    # Coppia di arrays per memorizzare le cardinalità al passo t-1 e t
    # Inizialmente ogni nodo ha cardinalità 1 (se stesso)
    prev_cardinality = np.ones(n_nodes, dtype=np.float32)
    harmonic_centrality = np.zeros(n_nodes, dtype=np.float32)

    alpha_m = 0.7213 / (1 + 1.079 / m)
    factor = alpha_m * (m ** 2)

    # PULIZIA RAM
    del G_rev, edges
    gc.collect()

    print("Dati pronti. Inizio loop CPU.")


    # -------------------------------------------------- #
    # LOOP PRINCIPALE HYPERBALL                          #
    # -------------------------------------------------- #
    t = 0
    changed = True

    while changed:
        # Time
        start_time = time.time()

        t += 1
        changed = False

        m_src = M_A if t % 2 == 1 else M_B
        m_target = M_B if t % 2 == 1 else M_A

        # PROPAGAZIONE IN AVANTI DEI CONTATORI
        m_target[:] = m_src[:]
        if len(sources) > 0:
            np.maximum.at(m_target, targets, m_src[sources])

        # Conversione in float sfruttando la memoria già allocata (M_float)
        np.copyto(M_float, m_target, casting='safe')

        # PRIMA STIMA - MEDIA ARMONICA (in-place)
        np.multiply(M_float, -1.0, out=M_float)
        np.power(2.0, M_float, out=M_float)
        sum_regs = np.sum(M_float, axis=1)
        estimate_raw = factor / sum_regs

        # CORREZIONE DELLA STIMA TRAMITE LINEAR COUNTING
        estimate = estimate_raw.copy()
        num_zeros = np.sum(m_target == 0, axis=1) # conta quanti registri sono rimasti a 0 per ciascun nodo
        threshold = 2.5 * m
        mask_lc = (estimate_raw <= threshold) & (num_zeros > 0) # Maschera booleana che determina quali nodi devono usare la correzione linear counting

        if np.any(mask_lc):
            # Formula: m * log(m / V)
            # Calcoliamo solo per i nodi nella maschera
            V = num_zeros[mask_lc].astype(np.float32)
            estimate[mask_lc] = m * np.log(m / V)

        # VERIFICA MODIFICHE RISPETTO ALLA STIMA DI CARDINALITA' PRECEDENTE
        diff = estimate - prev_cardinality
        mask_changed = diff > 0.001

        if np.any(mask_changed):
            changed = True
            harmonic_centrality[mask_changed] += diff[mask_changed] * (1.0 / t)
            prev_cardinality[mask_changed] = estimate[mask_changed]

        # Print finale
        elapsed = time.time() - start_time
        active_nodes = int(np.sum(mask_changed))
        print(f"Fine t={t}. Tempo CPU: {elapsed:.4f}s. Nodi attivi: {active_nodes}")

        if t > 1000:
            break

    # =========================================================================
    # FASE 4: RECUPERO RISULTATI
    # =========================================================================
    print("Calcolo finito. Restituzione dati...")
    return pd.DataFrame({
        'ASIN': nodes,
        'HarmonicCentrality': harmonic_centrality
    })


def harmonic_v3_GPU(G, p=10):
    # =========================================================================
    # FASE 1: SETUP SU CPU
    # =========================================================================
    print(f"--- FASE 1: Setup CPU e Hashing ---")

    G_rev = G.reverse()
    nodes = list(G_rev.nodes())
    n_nodes = len(nodes)

    m = 1 << p
    """
    Numero di registri che compongono ciascun contatore: m = 2^p
    """

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    """
    Mappatura nodo -> indice 0..N-1: dizionario del tipo (k, v) = (node, i), con i risultato dell'enumerazione dell' array di nodi
    """

    edges = np.array([(node_to_idx[u], node_to_idx[v]) for u, v in G_rev.edges()], dtype=np.int32)
    """
    Array di coppie (u_index, v_index), una per ogni edge del tipo (u, v) in G_rev
    """

    M_cpu = np.zeros((n_nodes, m), dtype=np.int32)
    """
    Matrice [n_nodes x m] dei contatori;
    Dato che un registro deve contenere il numero di leading zeroes di un hash (64 bits), il massimo valore inseribile in ciascun registro sara' < 64 (dato che i primi b bits dell' hash servono a individuare il registro corretto tra gli m), quindi 1 byte (uint8) e' sufficiente.
    """

    print("Calcolo hash iniziali su CPU...")

    # Pre-calcolo hash per ogni nodo per inizializzare M
    for i, node in enumerate(nodes):
        # Hash del nodo (crea hash con algoritmo md5 e trasforma il risultato in stringa hex, poi converte in intero a partire da base 16 verso base 10)
        h = int(hashlib.md5(str(node).encode('utf8')).hexdigest(), 16)

        # AND binario tra l' hash h e il numero m-1 = (2^10 - 1) = 1023 = sequenza di zeri seguiti da 10 valori '1'
        # il risultato corrisponde agli ultimi 10 bit di h, che selezionano il registro in cui scrivere il numero di leading zeroes
        j = h & (m - 1)

        # Right shift per rimuovere da h gli ultimi 10 bit estratti in precedenza
        w = h >> p

        # Conteggio del numero di trailing zeroes della porzione di hash rimanente (+1)
        rho = 1
        while (w & 1) == 0 and rho < 32: # while l'ultimo bit di w è uno '0':
            w >>= 1
            rho += 1
        M_cpu[i, j] = rho

    # =========================================================================
    # FASE 2: TRASFERIMENTO SU GPU
    # =========================================================================
    print(f"--- FASE 2: Spostamento dati su GPU, pre-allocazione ---")
    M_A = cp.asarray(M_cpu)         # Matrice corrente
    M_B = M_A.copy()                # Matrice per il prossimo passo (Double Buffering)
    M_float = cp.empty_like(M_A, dtype=cp.float32) # Buffer per i calcoli float

    if len(edges) > 0:
        sources_gpu = cp.asarray(edges[:, 1])
        """ Array monodimensionale ottenuto prendendo SOLO (tutta) la colonna 1 di edges """
        targets_gpu = cp.asarray(edges[:, 0])
        """ Array monodimensionale ottenuto prendendo SOLO (tutta) la colonna 0 di edges """
    else:
        sources_gpu = cp.array([], dtype=cp.int32)
        targets_gpu = cp.array([], dtype=cp.int32)

    # Coppia di arrays per memorizzare le cardinalità al passo t-1 e t
    # Inizialmente ogni nodo ha cardinalità 1 (se stesso)
    prev_cardinality_gpu = cp.ones(n_nodes, dtype=cp.float32)
    harmonic_centrality_gpu = cp.zeros(n_nodes, dtype=cp.float32)

    alpha_m = 0.7213 / (1 + 1.079 / m)
    factor = alpha_m * (m ** 2)

    del M_cpu, edges
    gc.collect()

    print(f"Dati pronti. VRAM inizialmente impegnata: ~{((M_A.nbytes * 3) / 1024**2):.2f} MB")

    # =========================================================================
    # FASE 3: LOOP PRINCIPALE SU GPU
    # =========================================================================
    t = 0
    changed = True

    while changed:
        t += 1
        changed = False

        # Time
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()


        m_src = M_A if t % 2 == 1 else M_B
        m_target = M_B if t % 2 == 1 else M_A


        # PROPAGAZIONE IN AVANTI DEI CONTATORI
        # se il numero di archi e' > 0, calcola il massimo tra il registro del nodo sorgente e i registri dei nodi target
        m_target[:] = m_src[:]
        if len(sources_gpu) > 0:
            # Aggiorna i nodi target con il massimo
            cp.maximum.at(m_target, targets_gpu, m_src[sources_gpu])

        # CONVERSIONI DELLA MATRICE DEI CONTATORI
        M_float[:] = m_target.astype(cp.float32)

        # PRIMA STIMA - MEDIA ARMONICA
        cp.multiply(M_float, -1.0, out=M_float)
        cp.exp2(M_float, out=M_float)
        sum_regs = cp.sum(M_float, axis=1)
        estimate_raw = factor / sum_regs

        # CORREZIONE DELLA STIMA TRAMITE LINEAR COUNTING
        num_zeros = cp.sum(m_target == 0, axis=1) # conta quanti registri sono rimasti a 0 per ciascun nodo (dato che True e' considerato 1)
        estimate = estimate_raw.copy()

        threshold = 2.5 * m
        # Maschera booleana che determina quali nodi devono usare la correzione linear counting. un elemento del vettore e' true solo se entrambe le condizioni sono verificate (AND bitwise)
        mask_lc = (estimate_raw <= threshold) & (num_zeros > 0)

        if cp.any(mask_lc):
            # Formula: m * log(m / V)
            # Calcoliamo solo per i nodi nella maschera
            V = num_zeros[mask_lc].astype(cp.float32)
            estimate[mask_lc] = m * cp.log(m / V)

        # VERIFICA MODIFICHE RISPETTO ALLA STIMA DI CARDINALITA' PRECEDENTE
        diff = estimate - prev_cardinality_gpu
        mask_changed = diff > 0.001

        if cp.any(mask_changed):
            changed = True
            harmonic_centrality_gpu[mask_changed] += diff[mask_changed] * (1.0 / t)
            prev_cardinality_gpu[mask_changed] = estimate[mask_changed]

        # Time
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start_time

        active_nodes = int(cp.sum(mask_changed))
        print(f"Fine t={t}. Tempo GPU: {elapsed:.4f}s. Nodi attivi: {active_nodes}")

        if t > 1000:
            break

    # =========================================================================
    # FASE 4: RECUPERO RISULTATI
    # =========================================================================
    print("Calcolo terminato. Recupero dati dalla GPU...")
    return pd.DataFrame({
        'ASIN': nodes,
        'HarmonicCentrality': harmonic_centrality_gpu.get()
    })