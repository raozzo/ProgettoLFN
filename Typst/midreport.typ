// --- 1. SETUP & DEFINITIONS ---

#let project(title: "", subtitle: "", authors: (), body) = {
  set document(author: authors, title: title)
  
  // Page setup
  set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, left: 2cm, right: 2cm),
  )
  
  // Fonts: Times New Roman is standard and safe
  set text(font: "Times New Roman", lang: "en", size: 11pt)
  set par(justify: true, first-line-indent: 0pt)

  // Heading styling
  set heading(numbering: "1.1")
  show heading: set text(font: "Arial", weight: "bold", fill: black)
  show link: set text(fill: blue)

  // Title Block
  align(center)[
    #text(font: "Arial", weight: 700, 1.75em, title) \
    #v(0.5em, weak: true)
    #text(1.2em, subtitle) \
    #v(1em, weak: true)
    #text(style: "italic", authors.join(", "))
  ]
  v(2em)

  body
}

// Custom Helpers
#let mygray(body) = text(fill: luma(40%), body)

#let newcontent(body) = block(
  stroke: (left: 3pt + blue),
  inset: (left: 10pt, y: 5pt),
  width: 100%,
  breakable: true,
  body
)

// Define math variables to fix "unknown variable" and make them look good
// This tells Typst to treat "rw" as a single italic symbol, not r * w
#let rw = math.italic("rw")
#let cc = math.italic("cc")
#let sr = math.italic("sr")
#let st = math.upright("st") // For the subscript "st"

// --- 2. THE REPORT CONTENT ---

#show: doc => project(
  title: "Comparative analysis of structural, learned, and hybrid feature representations for graph clustering",
  subtitle: "Learning From Networks - Project Proposal",
  authors: ("Leonardo Gusson", "Luca Rao", "Chiara Frizzarin"),
  doc
)

= Motivation <sec:motivation>

#mygray[
  This project aims at solving a question: is it possible to use a co-purchasing network to find out the actual categories of products considered? And which kind of features representation works better: structural, learned or hybrid?
]

The analysis is based on the "Amazon product co-purchasing network metadata" which was collected back in 2006 by crawling Amazon website.

From this dataset a directed graph $G=(V,A)$ is going to be built:
- *Nodes*: Each node $v in V$ represents a unique product in the Amazon dataset ($|V|=548,552$);
- *Arcs*: An edge $(u,v) in A$ exists if product $v$ is often co-purchased after $u$ ($|A|=1,788,725$).

#v(0.5em)
Each node comes with a set of information following this format:
- *Id*: Product id (number 0, ..., 548551)
- *ASIN*: Amazon Standard Identification Number
- *title*: Name/title of the product
- *group*: Product group (Book, DVD, Video or Music)
- *salesrank*: Amazon Salesrank
- *similar*: number $n in [0,5]$ of co-purchased products followed by a list of their ASINs (people who buy X also buy Y) (e.g. 2 B0001500VS B000002WA3)
- *categories*: Location in product category hierarchy to which the product belongs (separated by |, category id in [])
- *reviews*: Product review information: time, user id, rating, total number of votes on the review, total number of helpfulness votes (how many people found the review to be helpful)

#newcontent[
  A deeper analysis of the directed graph $G=(V,A)$ built from the Amazon co-purchasing dataset pointed out that weakly connected components are significantly heterogeneous. Specifically, a main connected component was identified, containing 334,843 nodes out of a total of 542,664 valid nodes. The subsequent connected components were found to be negligible in size (for instance, the second and third components contained only 222 and 184 nodes, respectively). Therefore, for this problem we will use the largest connected component of the graph to simplify the implementation of the algorithms.
  
  #text(fill: blue)[
    co-purchasing graph pre-processing:
    We decided to retain only products belonging to the categories 'Book', 'DVD', 'Video', and 'Music', along with their ASIN, title, group, and salesrank attributes. Subsequently, the largest connected component of the resulting directed graph was extracted and saved as a serialized NetworkX object in the .pickle format (chosen for efficiency) for downstream analysis.
  ]
]

All this information and the dataset can be found at #link("https://snap.stanford.edu/data/amazon-meta.html")[Stanford Large Network Dataset Collection].

= Method <sec:methods>

The objective of this project is to compare the effectiveness and performance of three different features sets to cluster a product co-purchasing graph, knowing that the number of clusters (i.e. product categories) is four.

== Structural + semantic features

We are going to combine the following structural centrality scores in a feature vector:

- *PageRank* $p(v)$ allows us to calculate the popularity of each product, in terms of important products co-purchased with other important products.
- *Closeness Centrality* $c(v)$ measures the importance of a product in the graph in terms of "trendsetters", i.e. it measures the closeness of a node to the others.
- *Betweenness Centrality* $b(v)$ represents crucial information for discovering key products that introduce customers to new categories.
- *Clustering Coefficient* $cc(v)$ identifies how a product's neighbors are interconnected, indicating whether it belongs to specialized kits of products or connects different groups.

#v(0.5em)
Then we compute a score $rw(v)$ analyzing the rating of the reviews, and a score $sr(v)$ based on the Amazon "salesrank", thus obtaining for each node $v$: 
$ arrow(F)_(#st)(v) = [p(v), c(v), b(v), cc(v), rw(v), sr(v)] $

#newcontent[
  where $rw(v)$ is a weighted review score, for each product $v$. The weight $w_i$ for an individual review $i$ is determined by its helpfulness votes (already in the dataset):
  
  $ w_i = "helpful_votes"_i + 1 $
  
  We add 1 to the weight to ensure that reviews with zero helpful votes (e.g., new reviews) are not discarded but still contribute minimally. The final score for a product is the weighted average of its individual review ratings $r_i$:
  
  $ rw(v) = (sum_(i=1)^(N) (r_i times w_i)) / (sum_(i=1)^(N) w_i) $
  
  where $N$ is the total number of reviews for product $v$.
]

== Node embeddings

As a second, parallel approach, we will move beyond single-score metrics and try to cluster the graph only using topological features through the use of *embeddings*:
$ arrow(E)(v) = [e_1, e_2, dots, e_d] $
where the dimension $d$ of the vector will be defined during the test phase (probably 64, 128 or 256).

== Hybrid Approach

The two approaches described above are powerful but capture fundamentally different types of information. Each has "blind spots" that the other can cover: indeed, centrality scores are interpretable and capture global roles while graph embeddings are excellent at capturing local context and semantic similarity. A hybrid approach can leverages the strengths of both.

To tackle this we will construct a multi-dimensional feature vector, $arrow(F)(v)$, for each product $v$. This vector will serve as the input for downstream machine learning models. All features will be normalized (e.g. using Min-Max scaling or Z-score standardization) to bring them into a common range.

The vector for a product $v$ is defined as a concatenation of its feature sets:
// Using sym.frown for the concatenation symbol
$ arrow(F)(v) = [ arrow(E)(v)^#sym.frown arrow(F)_(#st)(v) ] $

= Intended experiments <sec:experiments>

== Implementation

Given the graph size, we will implement approximated versions of pagerank, closeness centrality, betweenness centrality and clustering coefficient ourselves. On the other hand, the learned features vector $arrow(E)$ will be computed using the _Node2Vec_ algorithm, whose dimension $d$ will be defined during the implementation phase.

To identify the optimal node representation for clustering, we will conduct three independent experiments using K-means with $k=4$.
1. In the first approach we will tests the efficacy of classic, explicit graph metrics by feeding the standardized (e.g. using the Z-score) vector $arrow(F)_(#st)$ into K-means.
2. In the second approach we will evaluate the learned representation by first applying a dimensionality reduction technique (e.g. PCA or UMAP) due to the high dimensionality of embedding vectors, and finally we will use the reduced vector as K-means (or one of its approximate versions) input.
3. The same pipeline will be applied to the hybrid feature vector.

== Machines used
We have access to the following machine, we will use the fastest one:
- Macbook Air M2 (8Gb RAM)
- Laptop Intel Core Ultra 9 (32Gb RAM)
- Intel Core i7-6600U (8 Gb RAM)

== Experiments
To evaluate the three distinct approaches, we will perform a quantitative comparison based on two criteria: clustering quality and computational efficiency.
For quality, we will measure how well the resulting k-means clusters align with the four ground truth categories using evaluation metrics such as the ARI (Adjusted Rand Index) or the NMI (Normalized Mutual Information).
For efficiency, we will measure the total execution time required for each full pipeline.

= Additional details

In this first part of the project we collaborated often in-person and came up with a proposal through out an equally-participated brainstorming. We first looked at the suggested large networks and tried to think what we could have focused on and which techniques we could have applied. After that we consulted Gemini for better understanding if our ideas were enough challenging, but still doable (and also for proof-reading). The formalization and the writing of the project proposal were done in presence, similarly to the other phases, and the work split is not so rigorous because again we discussed and tried to write together.

Anyway it can be stated that:

- *Chiara:* Developed the project motivation (@sec:motivation) and led the dataset research and selection.
- *Luca:* Authored the core methodology (@sec:methods).
- *Leonardo:* Designed the experimental setup and validation plan (@sec:experiments).

#text(fill: blue)[
  In this second phase of the project we decided to split the workload into 3 parts regarding the implementations of the 4 methods considered. In this way each member is assigned a structural centrality score to implement:
  - *Luca:* PageRank.
  - *Leonardo:* Closeness Centrality.
  - *Chiara:* Betweenness Centrality and Clustering Coefficient.
]


