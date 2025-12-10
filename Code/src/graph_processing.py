import networkx as nx
import pandas as pd
import pickle
import os
import re
import sys

# --- CONFIGURAZIONE ---
# Percorsi
RAW_DATA_PATH = "../data/amazon-meta.txt"
# Modificato: usiamo l'estensione standard .pickle
OUTPUT_PATH = "../data/processed/amazon_graph.pickle"

# Creiamo la cartella di output se non esiste
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Gruppi da mantenere (come da proposta)
TARGET_GROUPS = {"Book", "DVD", "Video", "Music"}

print("Configurazione completata. Output sarà: ", OUTPUT_PATH)


# --- FUNZIONE DI PARSING ---
def parse_amazon_graph_data(file_path):
    """
    Legge il file raw e restituisce:
    1. Un dizionario di metadati dei nodi {ASIN: {data}}
    2. Un dizionario di adiacenza {ASIN: [neighbors]}
    """
    metadata = {}
    adjacency = {}

    current_asin = None
    current_meta = {}

    # Regex compilate per velocità
    id_pattern = re.compile(r"^Id:\s+(\d+)")
    asin_pattern = re.compile(r"^ASIN:\s+(\w+)")
    title_pattern = re.compile(r"^title:\s+(.+)")
    group_pattern = re.compile(r"^group:\s+(.+)")
    salesrank_pattern = re.compile(r"^salesrank:\s+(\d+)")
    similar_pattern = re.compile(r"^similar:\s+\d+\s+(.+)")

    print(f"Inizio lettura file: {file_path}")

    # Encoding 'utf-8' con 'ignore' per evitare crash su caratteri strani nel dataset
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # NUOVO PRODOTTO (Inizia con Id:)
            if line.startswith("Id:"):
                # Salva il prodotto precedente se valido e del gruppo giusto
                if current_asin and current_meta.get("group") in TARGET_GROUPS:
                    metadata[current_asin] = current_meta

                # Reset variabili
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

        # Salva l'ultimo prodotto alla fine del file
        if current_asin and current_meta.get("group") in TARGET_GROUPS:
            metadata[current_asin] = current_meta

    return metadata, adjacency


# --- ESECUZIONE PARSING ---
# Nota: Assicurati che RAW_DATA_PATH punti al file corretto sul tuo PC
if not os.path.exists(RAW_DATA_PATH):
    print(f"ERRORE: File non trovato in {RAW_DATA_PATH}")
else:
    nodes_dict, edges_dict = parse_amazon_graph_data(RAW_DATA_PATH)
    print(f"Nodi validi trovati (filtrati per gruppo): {len(nodes_dict)}")

    # --- COSTRUZIONE DEL GRAFO ---
    print("Costruzione del grafo NetworkX (DIRETTO)...")
    G = nx.DiGraph()  # Grafo DIRETTO (DiGraph)

    # 1. Aggiungi i nodi
    for asin, data in nodes_dict.items():
        G.add_node(asin, **data)

    # 2. Aggiungi gli archi (solo se entrambi i nodi esistono)
    edge_count = 0
    for u, neighbors in edges_dict.items():
        if u in nodes_dict:
            for v in neighbors:
                if v in nodes_dict:
                    G.add_edge(u, v)
                    edge_count += 1

    print(f"Grafo costruito.")
    print(f"Nodi Totali: {G.number_of_nodes()}")
    print(f"Archi Totali: {G.number_of_edges()}")

    # --- ESTRAZIONE COMPONENTE GIGANTE (LWCC) ---
    # Per grafi diretti, usiamo la componente DEBOLMENTE connessa più grande
    print("Estrazione della Componente Debolmente Connessa Gigante (LWCC)...")
    # Nota: per DiGraph si usa weakly_connected_components
    connected_components = sorted(
        nx.weakly_connected_components(G), key=len, reverse=True
    )

    if connected_components:
        giant_component_nodes = connected_components[0]
        G_giant = G.subgraph(giant_component_nodes).copy()

        print(f"GCC estratta.")
        print(f"Nodi GCC: {G_giant.number_of_nodes()}")
        print(f"Archi GCC: {G_giant.number_of_edges()}")

        # --- SALVATAGGIO ---
        print(f"Salvataggio in {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(G_giant, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Fatto! File .pickle generato con successo.")
    else:
        print("Errore: Il grafo sembra essere vuoto o non avere componenti connesse.")
