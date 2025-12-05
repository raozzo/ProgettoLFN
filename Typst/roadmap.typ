#import "@preview/cheq:0.3.0": checklist


// --- Setup & Styling ---
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "1",
)
#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
)

#set text(
  font: "Linux Libertine",
  size: 11pt,
  lang: "en"
)

#let accent = rgb("#2563eb")

#show heading: it => [
  #set text(fill: accent)
  #it
]

// --- Header ---
#align(center)[
  #text(size: 20pt, weight: "bold", fill: accent)[Project Implementation Plan] \
  #v(0.5em)
  #text(size: 14pt)[Comparative Analysis of Graph Clustering Features] \
  #v(1em)
  *Team:* Leonardo Gusson, Luca Rao, Chiara Frizzarin
]

#line(length: 100%, stroke: 1pt + gray)
#v(2em)

// --- Roles ---
= 1. Role Division

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 10pt,
  align: center,
  fill: (_, row) => if row == 0 { accent.lighten(90%) } else { white },
  
  [*Chiara* \ _Data & Integration_],
  [*Luca* \ _Graph Engineering_],
  [*Leonardo* \ _ML & Experiments_],
  
  [Data ingestion \ Cleaning \ Hybrid Vectors],
  [Structural Metrics \ PageRank, BC, CC \ Approximations],
  [Node2Vec \ Dimensionality Reduction \ Clustering],
)

// --- Tech Stack ---
= 2. Recommended Tech Stack
- *Graph Library:* `igraph` (C++) or `networkx`
- *Embeddings:* `node2vec` or `gensim`
- *ML:* `scikit-learn`
- *Data:* `pandas`, `numpy`

// --- Roadmap ---
= 3. Implementation Roadmap

== Phase 1: Setup & Data Ingestion (Chiara)
#show: checklist
- [x] **Repo Setup:** GitHub repo with `.gitignore`
- [ ] **Parser:** Extract ASIN, Group, and Edges from `amazon-meta.txt`
- [ ] **Filter:** Keep only Book, DVD, Video, Music groups
- [ ] **Graph:** Build Undirected Graph object

== Phase 2: Feature Engineering (Parallel)
*Structural (Luca)*
- [ ] **PageRank:** Compute and normalize
- [ ] **Clustering Coeff:** Compute and normalize
- [ ] **Approx. Betweenness:** Sampling-based BC ($k=1000$)
- [ ] **Approx. Closeness:** Sampling-based CC

*Topological (Leonardo)*
- [ ] **Node2Vec:** Setup random walks
- [ ] **Training:** Train for $d=128$ dimensions
- [ ] **Storage:** Save to `.npy`

== Phase 3: Hybridization (Chiara)
- [ ] **Merge:** Master DataFrame indexed by Node ID
- [ ] **Hybrid Vector:** Concatenate Structural (6 dims) + Embedding (128 dims)
- [ ] **Reduce:** Apply UMAP/PCA

== Phase 4: Experiments (Leonardo)
- [ ] **Cluster:** K-Means ($k=4$) on all 3 datasets
- [ ] **Validate:** Compute ARI and NMI scores
- [ ] **Visualize:** t-SNE scatter plots
- [ ] **Profile:** Measure execution time
