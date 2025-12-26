import networkx as nx
import numpy as np

from datasketch import HyperLogLog
import hashlib
import cupy as cp
import time
import copy


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

def get_harmonic_centrality(G, p=10, version=2):
    """
    Calcola la Harmonic Centrality approssimata usando HyperBall.

    Args:
        :param G: Oggetto NetworkX DiGraph (350k nodi).
        :param p: Precisione dei registri (p=10 -> 2^10 registri = errore ~3%).
        :param version: Variante dell'implementazione con cui calcolare la Harmonic Centrality
                        (1 -> CPU, 2 -> CPU (improved), 3 -> CPU+GPU).
    """
    if version == 1:
        return harmonic_v1_CPU(G, p)
    elif version == 3:
        return harmonic_v3_GPU(G, p)
    else:
        return harmonic_v2_CPU(G, p)


def harmonic_v1_CPU(G, p=10):
    # =========================================================================
    # FASE 1: PREPARAZIONE E INIZIALIZZAZIONE
    # =========================================================================
    # Per calcolare la centralità "in entrata" (quanto sono importante),
    # dobbiamo contare chi può raggiungere ME. HyperBall propaga "in avanti",
    # quindi lavoriamo sul grafo trasposto (archi invertiti).
    print(f"--- FASE 1: Inversione grafo e Inizializzazione HLL (p={p}) ---")
    G_rev = G.reverse()
    nodes = list(G_rev.nodes())

    # =========================================================================
    # FASE 2: COSTRUZIONE DELLA MATRICE DI CONTATORI
    # =========================================================================

    # Dizionario per i contatori attuali: {nodo: HyperLogLog}
    # Inizialmente ogni nodo conosce solo se stesso (distanza 0).
    counters = {}
    for node in nodes:
        hll = HyperLogLog(p=p)
        # HLL richiede input in bytes. Convertiamo l'ID del nodo.
        node_id_encoded = str(node).encode('utf-8')
        # Aggiungo il nodo stesso al contatore (inizialmente l'insieme dei nodi raggiungibili contiene solo se stesso
        hll.update(node_id_encoded)
        # Aggiungo il contatore al dizionario che associa ogni nodo ad un contatore HyperLogLog
        counters[node] = hll

    # Dizionari per memorizzare i risultati
    # per ogni nodo in nodes, aggiungi al dizionario la coppia chiave-valore (node:0.0)
    harmonic_centrality = {node: 0.0 for node in nodes}

    # Memorizziamo la cardinalità al passo precedente (N_{t-1}).
    # Al tempo t=0, ogni nodo raggiunge solo se stesso, quindi count = 1.
    prev_cardinality = {node: 1.0 for node in nodes}

    print("Inizializzazione completata.")

    # =========================================================================
    # FASE 2: LOOP PRINCIPALE (Espansione della 'Palla') [cite: 771, 778]
    # =========================================================================
    # Iteriamo per t = 1, 2, ... fino a che le stime non cambiano più (stabilizzazione).
    # t rappresenta la distanza (raggiungibili in t passi).
    t = 0
    changed = True

    while changed:
        t += 1
        changed = False
        print(f"--- Inizio Iterazione t={t} ---")
        start_time = time.time()

        # Buffer per i nuovi contatori del passo t
        next_counters = {}

        # =====================================================================
        # FASE 3: UNIONE DEI CONTATORI (Propagazione) [cite: 775, 780]
        # =====================================================================
        # Logica: I nodi che posso raggiungere in t passi sono l'unione di:
        # 1. Quelli che raggiungevo già (me stesso e i vecchi percorsi)
        # 2. Quelli che raggiungono i miei vicini al passo t-1.

        for node in nodes:
            # Copiamo il contatore attuale del nodo (stato t-1)
            # NOTA: copy è necessario perché HLL è mutabile
            hll_new = copy.copy(counters[node])

            # Uniamo con i contatori dei vicini (successori nel grafo trasposto, G_rev.neighbors() = G.successors())
            # Questo simula il passaggio di informazioni "indietro" nel grafo originale
            neighbors = list(G_rev.neighbors(node))

            if neighbors:
                for neighbor in neighbors:
                    # L'operazione di unione HLL è molto veloce (bit-wise OR dei registri)
                    hll_new.merge(counters[neighbor])

            # Salviamo il nuovo stato
            next_counters[node] = hll_new

        # =====================================================================
        # FASE 4: AGGIORNAMENTO METRICA (Calcolo Harmonic)
        # =====================================================================
        # Formula: H(x) = sum [ (N_t - N_{t-1}) / t ]
        # Dove (N_t - N_{t-1}) è la stima dei nodi trovati ESATTAMENTE a distanza t.

        current_change_count = 0

        for node in nodes:
            old_count = prev_cardinality[node]
            new_count = next_counters[node].count()

            # Se la stima è aumentata, abbiamo trovato nuovi nodi a distanza t
            if new_count > old_count:
                delta = new_count - old_count

                # Aggiungiamo il contributo alla centralità armonica
                harmonic_centrality[node] += delta * (1.0 / t)

                # Aggiorniamo la memoria per il prossimo passo
                prev_cardinality[node] = new_count

                # Segnaliamo che c'è stato un cambiamento nel sistema
                changed = True
                current_change_count += 1

        # =====================================================================
        # FASE 5: CHIUSURA ITERAZIONE E CONTROLLO CONVERGENZA [cite: 833]
        # =====================================================================
        # Scambiamo i buffer: i contatori 'next' diventano quelli attuali per il prossimo t
        counters = next_counters

        elapsed = time.time() - start_time
        print(f"Fine t={t}. Nodi aggiornati: {current_change_count}. Tempo: {elapsed:.2f}s")

        # Sicurezza per evitare loop infiniti in grafi patologici,
        # anche se HLL converge tipicamente entro il diametro effettivo del grafo.
        if t > 1000: # Cutoff arbitrario, aumentabile
            print("Raggiunto limite massimo iterazioni.")
            break

    return harmonic_centrality

def harmonic_v2_CPU(G, p=10):
    # =========================================================================
    # FASE 1: SETUP
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

    M = np.zeros((n_nodes, m), dtype=np.int32)
    """
    Matrice [n_nodes x m] dei contatori;
    Dato che un registro deve contenere il numero di leading zeroes di un hash (64 bits), il massimo valore inseribile in ciascun registro sara' < 64 (dato che i primi b bits dell' hash servono a individuare il registro corretto tra gli m), quindi 1 byte (uint8) e' sufficiente (qui usiamo int32 per sicurezza e compatibilità).
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
        M[i, j] = rho

    # =========================================================================
    # FASE 2: PREPARAZIONE DATI
    # =========================================================================
    print(f"--- FASE 2: Preparazione vettori NumPy ---")

    if len(edges) > 0:
        sources = edges[:, 1]
        """ Array monodimensionale ottenuto prendendo SOLO (tutta) la colonna 1 di edges """
        targets = edges[:, 0]
        """ Array monodimensionale ottenuto prendendo SOLO (tutta) la colonna 0 di edges """
    else:
        sources = np.array([], dtype=np.int32)
        targets = np.array([], dtype=np.int32)

    # Coppia di arrays per memorizzare le cardinalità al passo t-1 e t
    # Inizialmente ogni nodo ha cardinalità 1 (se stesso)
    prev_cardinality = np.ones(n_nodes, dtype=np.float32)
    harmonic_centrality = np.zeros(n_nodes, dtype=np.float32)

    alpha_m = 0.7213 / (1 + 1.079 / m)
    factor = alpha_m * (m ** 2)

    print("Dati pronti. Inizio loop CPU.")

    # =========================================================================
    # FASE 3: LOOP PRINCIPALE SU CPU
    # =========================================================================
    t = 0
    changed = True

    while changed:
        t += 1
        changed = False

        # Time
        start_time = time.time()

        M_prev = M.copy()

        # PROPAGAZIONE IN AVANTI DEI CONTATORI
        # se il numero di archi e' > 0, calcola il massimo tra il registro del nodo sorgente e i registri dei nodi target
        if len(sources) > 0:
            # Prendi i registri dei nodi sorgente
            source_registers = M_prev[sources]
            # Aggiorna i nodi target con il massimo
            # np.maximum.at è l'equivalente NumPy di cp.maximum.at
            np.maximum.at(M, targets, source_registers)

        # CONVERSIONI DELLA MATRICE DEI CONTATORI
        M_float = M.astype(np.float32)
        M_bool = M == 0 # nuova matrice booleana che contiene True se il valore corrispondente in M e' 0

        # PRIMA STIMA - MEDIA ARMONICA
        reg_pow = np.power(2.0, -M_float)
        sum_regs = np.sum(reg_pow, axis=1)

        # Gestione divisione per zero (se sum_regs è 0, anche se improbabile con inizializzazione corretta)
        with np.errstate(divide='ignore'):
            estimate_raw = factor / sum_regs

        # CORREZIONE DELLA STIMA TRAMITE LINEAR COUNTING
        num_zeros = np.sum(M_bool, axis=1) # conta quanti registri sono rimasti a 0 per ciascun nodo
        estimate = estimate_raw.copy()
        threshold = 2.5 * m
        # Maschera booleana che determina quali nodi devono usare la correzione linear counting
        mask_lc = (estimate_raw <= threshold) & (num_zeros > 0)

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

        # Time
        elapsed = time.time() - start_time

        active_nodes = int(np.sum(mask_changed))
        print(f"Fine t={t}. Tempo CPU: {elapsed:.4f}s. Nodi attivi: {active_nodes}")

        if t > 1000:
            break

    # =========================================================================
    # FASE 4: RECUPERO RISULTATI
    # =========================================================================
    print("Calcolo finito. Restituzione dati...")

    final_centrality = harmonic_centrality
    result = {nodes[i]: final_centrality[i] for i in range(n_nodes)}
    return result

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
    print(f"--- FASE 2: Spostamento dati su GPU ---")

    M_gpu = cp.asarray(M_cpu)

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

    print("Dati in VRAM. Inizio loop GPU.")

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

        M_gpu_prev = M_gpu.copy()

        # PROPAGAZIONE IN AVANTI DEI CONTATORI
        # se il numero di archi e' > 0, calcola il massimo tra il registro del nodo sorgente e i registri dei nodi target
        if len(sources_gpu) > 0:
            # Prendi i registri dei nodi sorgente
            source_registers = M_gpu_prev[sources_gpu]
            # Aggiorna i nodi target con il massimo
            cp.maximum.at(M_gpu, targets_gpu, source_registers)

        # CONVERSIONI DELLA MATRICE DEI CONTATORI
        M_gpu_float = M_gpu.astype(cp.float32)
        M_gpu_bool = M_gpu == 0 # nuova matrice booleana che contiene True se il valore corrispondente in M_gpu e' 0

        # PRIMA STIMA - MEDIA ARMONICA
        reg_pow = cp.power(2.0, -M_gpu_float)
        sum_regs = cp.sum(reg_pow, axis=1)
        estimate_raw = factor / sum_regs

        # CORREZIONE DELLA STIMA TRAMITE LINEAR COUNTING
        num_zeros = cp.sum(M_gpu_bool, axis=1) # conta quanti registri sono rimasti a 0 per ciascun nodo (dato che True e' considerato 1)
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
    print("Calcolo finito. Recupero dati dalla GPU...")

    final_centrality = harmonic_centrality_gpu.get()
    result = {nodes[i]: final_centrality[i] for i in range(n_nodes)}
    return result
