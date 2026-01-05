import random
from collections import deque
import networkx as nx
import numpy as np

#from datasketch import HyperLogLog
import hashlib
#NOTE spostato cupy solo in se GPU 
import time
import copy
import pandas as pd
import gc


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


def get_clustering_coefficient(G, M=1000):
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
        N_u_S = {x for (x, y) in S if y==u} | {y for (x, y) in S if x==u}
        N_v_S = {x for (x, y) in S if y==v} | {y for (x, y) in S if x==v}
        N_uv_S = N_u_S & N_v_S

        # For each c in N_uv_S, increment triangle count for u,v,c
        for c in N_uv_S:
            t_S[u] += 1
            t_S[v] += 1
            t_S[c] += 1

        # Reservoir sampling

        # CASE 1: add edge (u,v) to S if |S| < M
        if t <= M:
            S.append((u,v))

        # CASE 2: add edge (u,v) to S with probability M/t, 
        # replacing a random edge in S, chosen uniformly at random with probability 1/M
        else:
            prob = random.uniform(0,1)
            if prob <= M / t:
                index = random.randint(0, M-1)
                S[index] = (u,v)

    # Compute clustering coefficient for each node
    for v in G.nodes:
        cc[v] = (t_S[v]) / (d[v] * (d[v] -1)) # we should not multiply by 2 (directed graph adaptation)

        # correction for sampling ???
        # p = M / t
        # cc[v] /= p**2

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
    # Initialize structures
    distance = {v: float('inf') for v in G.nodes()}
    predecessors = {v: [] for v in G.nodes()}

    distance[s] = 0
    Q = deque([s])
    found_target_distance = float('inf')

    # BFS traversal to build predecessors map
    # We stop exploring paths longer than the shortest path to t
    # This ensures we only find shortest paths
    while Q:
        v = Q.popleft()

        # If we reached the target, record the distance
        if distance[v] >= found_target_distance:
            continue

        # Explore successors of v (outgoing edges) 
        for w in G.successors(v):
            # CASE 1: first time visiting w (standard BFS)
            if distance[w] == float('inf'):
                distance[w] = distance[v] + 1
                predecessors[w].append(v)
                Q.append(w)
                if w == t:
                    found_target_distance = distance[w]
            # CASE 2: found another shortest path to w, which was already visited
            elif distance[w] == distance[v] + 1:
                predecessors[w].append(v)
    
    # Reconstruct all SPs from s to t backtracking using predecessors
    all_SPs = [] # List to store all shortest paths

    # Helper function for backtracking
    def FindPaths(v, current_path):
        if v == s:
            all_SPs.append(current_path[::-1])  # Append reversed path
            return
        for pred in predecessors[v]:
            FindPaths(pred, current_path + [pred])

    if distance[t] != float('inf'):
        FindPaths(t, [t])

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

    for i in range(k):

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

        # Find all shortest paths from s to all other nodes using BFS adapted for directed graphs and store them in a list P_st
        P_st = bfs_directed(G, s, t)

        # Choose a shortest path uniformly at random from P_st
        if P_st: # Check if there are any paths
            path = random.choice(P_st)

            # For each node v in the chosen path (but s and t), increment b[v] by 1/k
            for v in path:
                if v != s and v != t:
                    b[v] += 1.0 / k

    # Normalize the betweenness scores for directed graphs: divide by (n-1)(n-2)
    # Multiply by n/k to scale the scores based on the number of samples ???
    n = len(V)
    normalization_factor = 1 / (n * (n - 1))
    for v in b:
        b[v] *= normalization_factor

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