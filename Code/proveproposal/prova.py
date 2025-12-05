import gzip
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyvis.network import Network
import math
import random


def build_graph_from_meta(file_path, limit=None):
    """
    Costruisce il grafo e i metadati direttamente da amazon-meta.txt.gz
    limit: (opzionale) numero massimo di righe da leggere per test rapidi.
    """
    print(f"--- Inizio parsing di {file_path} ---")

    G = nx.Graph()  # Usa DiGraph() se vuoi archi diretti
    asin_to_id = {}  # Mappa per convertire ASIN in ID numerici
    temp_edges = {}  # Memoria temporanea per gli archi (ID -> [lista ASIN])

    current_id = -1
    current_asin = ""

    # Variabili di stato
    reading_reviews = False

    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i > limit:
                break

            line = line.strip()

            # Gestione ID (Inizio nuovo nodo)
            if line.startswith("Id:"):
                current_id = int(line.split()[1])
                reading_reviews = False
                continue

            # Gestione ASIN
            if line.startswith("ASIN:"):
                current_asin = line.split()[1]
                asin_to_id[current_asin] = current_id
                G.add_node(current_id, label=current_asin)  # Crea il nodo
                continue

            # Gestione Titolo
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip()
                nx.set_node_attributes(G, {current_id: title}, "title")
                continue

            # Gestione Gruppo (La tua ground-truth)
            if line.startswith("group:"):
                group = line.split(":", 1)[1].strip()
                nx.set_node_attributes(G, {current_id: group}, "group")
                continue

            # Gestione Similar (Gli archi)
            if line.startswith("similar:"):
                parts = line.split()
                # parts[0] è "similar:", parts[1] è il conteggio, dal 2 in poi sono ASIN
                if len(parts) > 2:
                    similar_asins = parts[2:]
                    temp_edges[current_id] = similar_asins
                continue

    print(f"Parsing completato. Nodi trovati: {G.number_of_nodes()}")
    print("Costruzione degli archi (conversione ASIN -> ID)...")

    # Fase 2: Creazione Archi
    edge_count = 0
    for source_id, targets in temp_edges.items():
        for target_asin in targets:
            if target_asin in asin_to_id:
                target_id = asin_to_id[target_asin]
                # Aggiungiamo l'arco solo se entrambi i nodi esistono nel dataset
                G.add_edge(source_id, target_id)
                edge_count += 1
            else:
                # Nota: Amazon-meta ha riferimenti a ASIN che non hanno una entry completa
                # Li ignoriamo per avere un grafo pulito
                pass

    print(f"Archi creati: {edge_count}")
    return G


def scale_size(val, min_val, max_val, min_size=5, max_size=30):
    """Funzione helper per scalare logaritmicamente la dimensione dei nodi."""
    if val <= 0:
        return min_size
    log_val = math.log10(val)
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)

    if log_max == log_min:
        return min_size

    normalized = (log_val - log_min) / (log_max - log_min)
    return min_size + normalized * (max_size - min_size)


def visualize_ego_network(
    G, hub_node_id, max_neighbors=150, filename="amazon_ego_network.html"
):
    """
    Estrae l'ego-network di un nodo hub e crea una visualizzazione Pyvis
    interattiva, colorata per gruppo e dimensionata per PageRank.
    """
    print(f"\n--- Creazione visualizzazione per l'hub: {hub_node_id} ---")

    # Estrai vicini e limita il numero per leggibilità
    neighbors = list(G.neighbors(hub_node_id))
    if len(neighbors) > max_neighbors:
        neighbors = random.sample(neighbors, max_neighbors)

    ego_nodes = neighbors + [hub_node_id]
    ego_graph = G.subgraph(ego_nodes)

    print(
        f"Sottografo 'ego' creato: {ego_graph.number_of_nodes()} nodi, {ego_graph.number_of_edges()} archi."
    )

    # Inizializza Pyvis
    net = Network(height="800px", width="100%", notebook=False, cdn_resources="in_line")

    # Trova min/max PageRank *solo nel sottografo* per una scala visiva migliore
    pr_scores = nx.get_node_attributes(ego_graph, "pagerank")
    if not pr_scores:
        print("Errore: Attributo 'pagerank' non trovato.")
        return

    min_pr = min(pr_scores.values()) if pr_scores else 0.00001
    max_pr = max(pr_scores.values()) if pr_scores else 0.00001

    # Mappa colori
    groups = set(nx.get_node_attributes(ego_graph, "group").values())
    colors = [
        "#FF5733",
        "#33FF57",
        "#3357FF",
        "#FF33A1",
        "#33FFF6",
        "#F6FF33",
        "#A133FF",
    ]
    color_map = {group: colors[i % len(colors)] for i, group in enumerate(groups)}
    default_color = "#999999"

    # Aggiungi nodi al network Pyvis
    for node_id in ego_graph.nodes():
        attrs = G.nodes[node_id]  # Ottieni tutti gli attributi

        pr_score = attrs.get("pagerank", min_pr)
        group = attrs.get("group", "Unknown")
        title = attrs.get("title", "N/A")

        # Dimensiona in base al PageRank
        node_size = scale_size(pr_score, min_pr, max_pr)
        color = color_map.get(group, default_color)

        # Il popup che appare al passaggio del mouse
        title_popup = f"""
        --- Prodotto ---<br>
        <b>Titolo:</b> {title}<br>
        <b>Gruppo:</b> {group}<br>
        <b>ID:</b> {node_id}<br>
        <b>PageRank:</b> {pr_score:.2e}
        """

        # Styling speciale per il nodo HUB
        if node_id == hub_node_id:
            net.add_node(
                node_id,
                label=title,
                color="#FF0000",
                size=node_size * 1.5,
                title=title_popup,
            )
        else:
            net.add_node(
                node_id, label=title, color=color, size=node_size, title=title_popup
            )

    # Aggiungi gli archi
    net.add_edges(ego_graph.edges())
    net.repulsion(node_distance=150, central_gravity=0.01, spring_length=150)
    net.show_buttons(filter_=["physics"])

    # Salva il file HTML
    net.save_graph(filename)
    print(f"Visualizzazione interattiva salvata in: '{filename}'")


# --- 1. ESECUZIONE ---
# Imposta limit=100000 per fare una prova veloce, o toglilo per tutto il file
G = build_graph_from_meta("amazon-meta.txt.gz")

# Pulizia: Rimuoviamo nodi isolati (senza archi) per migliorare il clustering
print("Rimozione nodi isolati...")
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Nodi rimanenti: {G.number_of_nodes()}")

# --- 2. CALCOLO METRICHE (Feature Engineering) ---
print("\n--- Calcolo Metriche Topologiche ---")

# A. PageRank (Importanza globale)
print("Calcolo PageRank...")
pr = nx.pagerank(G, alpha=0.85)

# B. Clustering Coefficient (Connettività locale)
print("Calcolo Clustering Coefficient...")
cc = nx.clustering(G)

# C. Degree Centrality (Popolarità semplice)
deg = nx.degree_centrality(G)

# D. Betweenness Centrality
# ATTENZIONE: Su grafi grandi (>10k nodi) è LENTISSIMO.
# Usiamo un'approssimazione (k=sampling) o lo saltiamo per ora.
print("Calcolo Betweenness (stimata su campione k=100)...")
bet = nx.betweenness_centrality(G, k=100)


# ===== INSERISCI QUESTO CODICE QUI SOTTO =====
print("Aggiunta metriche come attributi del nodo...")
nx.set_node_attributes(G, pr, "pagerank")
nx.set_node_attributes(G, cc, "clustering_coeff")
nx.set_node_attributes(G, deg, "degree_centrality")
nx.set_node_attributes(G, bet, "betweenness")
# ===============================================

# --- 3. CREAZIONE DATAFRAME PER CLUSTERING ---
print("\n--- Preparazione Dati per Clustering ---")

# Estraiamo i dati in liste ordinate per ID nodo
nodes = list(G.nodes())
data = {
    "node_id": nodes,
    "pagerank": [pr[n] for n in nodes],
    "clustering_coeff": [cc[n] for n in nodes],
    "degree": [deg[n] for n in nodes],
    "betweenness": [bet[n] for n in nodes],
    # Ground Truth (target)
    "category_group": [G.nodes[n].get("group", "Unknown") for n in nodes],
    "title": [G.nodes[n].get("title", "Unknown") for n in nodes],
}

df = pd.DataFrame(data)

# Filtriamo via i gruppi 'Unknown' o categorie irrilevanti se vuoi
df_clean = df[df["category_group"] != "Unknown"]

print(f"Dataset pronto per il clustering: {df_clean.shape}")
print(df_clean.head())

# --- 4. NORMALIZZAZIONE E EXPORT ---
# Seleziona solo le colonne numeriche per il clustering
feature_cols = ["pagerank", "clustering_coeff", "degree", "betweenness"]
X = df_clean[feature_cols].values

# Standardizza (Media 0, Std 1) - Fondamentale per K-Means!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Salviamo il risultato per usarlo in un altro script o notebook
output_file = "amazon_features_matrix.csv"
# Salviamo sia le feature scalate che le etichette originali
df_final = pd.DataFrame(X_scaled, columns=feature_cols)
df_final["node_id"] = df_clean["node_id"].values
df_final["category_group"] = df_clean["category_group"].values
df_final.to_csv(output_file, index=False)

print(f"\nFile salvato: {output_file}")
print("Ora puoi usare questo CSV per K-Means, Graphlets o Node2Vec.")

# --- 5. VISUALIZZAZIONE SOTTOGRAFO (Nuova Sezione) ---
print("\n--- Avvio Visualizzazione Sottografo ---")
if G.number_of_nodes() > 0:
    # Trova il nodo hub (grado più alto)
    degrees = dict(G.degree())
    hub_node = max(degrees, key=degrees.get)
    print(
        f"Nodo 'hub' identificato (grado più alto): {hub_node} (Grado: {degrees[hub_node]})"
    )

    # Ottieni il titolo dell'hub per un nome file più carino
    hub_title = (
        G.nodes[hub_node]
        .get("title", f"hub_{hub_node}")
        .replace(" ", "_")
        .replace("/", "_")
    )
    safe_title = "".join(c for c in hub_title if c.isalnum() or c == "_")[:50]

    output_viz_file = f"network_viz_{safe_title}.html"

    # Chiama la funzione di visualizzazione
    visualize_ego_network(G, hub_node, max_neighbors=12000, filename=output_viz_file)
else:
    print("Grafo vuoto, nessuna visualizzazione da generare.")

print("\nScript completato.")
