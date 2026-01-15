# ProgettoLFN

---

## Da citare nella midterm review:

* utilizziamo la componente connessa più grande del grafo, ignoriamo le altre
* Al posto della closeness centrality usiamo la harmonic, più adatta a gestire i nodi irraggiungibili.
* Valutiamo se usare accelerazione cuda cupy (dato che in cpu single-core ci ha messo 56min)
* cosa facciamo con lo score rw
* 

---

## Link utili 
* [Cartella Drive](https://drive.google.com/drive/folders/14T74Tn7I2E42TKAConzHNnDavb_Vng_T?usp=drive_link)
* [SNAP Dataset](https://snap.stanford.edu/data/amazon-meta.html)
* [Network Repository](https://networkrepository.com/)

---

# README

# Learning From Networks: Comparative analysis of structural, learned, and hybrid feature representations for graph clustering

This project explores and compares structural, learned (embeddings), and hybrid feature representations for clustering a graph based on the Amazon product dataset.

## 🚀 Guide through the Structure of the Project

To get a comprehensive overview of the final results and the comparative analysis, start with the main notebook.

### 1. General Results and Comparative Analysis
All comparisons between representations (Structural, Learned, Hybrid), confusion matrices, and evaluation metrics can be found here:
* 📊 **[`/Code/notebooks/print.ipynb`](./Code/notebooks/print.ipynb)**: The main project report and final interactive dashboard.

---

### 2. Handmade Implementations
The primary graph centrality metrics were implemented manually to study their algorithmic behavior. The source code is located in:
* 🛠️ **[`/Code/src/features/`](./Code/src/features/)**: Contains manual implementations of the structural features.

Detailed tests and step-by-step results for these implementations are documented in their dedicated notebooks:
* [Betweenness Centrality](./Code/notebooks/betweenness_centrality.ipynb)
* [Clustering Coefficient](./Code/notebooks/clustering_coefficient.ipynb)
* [Harmonic Centrality](./Code/notebooks/harmonic_centrality.ipynb)
* [PageRank](./Code/notebooks/pagerank.ipynb)

---

### 3. Additional Specialized Notebooks
* **[`embeddings.ipynb`](./Code/notebooks/embeddings.ipynb)**: Generation of learned features using node embedding algorithms.
* **[`scoring.ipynb`](./Code/notebooks/scoring.ipynb)**: Calculation of a helpfulness-weighted *Review Score* to add a qualitative dimension to the nodes.

---

### 4. Data Structure and Preprocessing
Data generated throughout the different project phases is organized as follows:
* **Computed Features**: [`/Code/data/processed/`](./Code/data/processed/) (CSV files for the various centrality scores).
* **Clustering Results**: [`/Code/data/processed/results/`](./Code/data/processed/results/) (Model outputs and evaluation metrics).

The data handling is managed by:
* **[`graph_processing.py`](./Code/src/graph_processing.py)**: Script dedicated to loading and cleaning the original Amazon metadata graph.
* **[`utils.py`](./Code/src/utils.py)**: Helper functions (e.g., the `load_or_compute` logic) used across the project, particularly in `print.ipynb`.

---

## 👥 Authors
* Chiara Frizzarin
* Leonardo Gusson
* Luca Rao