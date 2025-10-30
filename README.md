# ProgettoLFN

## Link utili 
[Cartella Drive](https://drive.google.com/drive/folders/14T74Tn7I2E42TKAConzHNnDavb_Vng_T?usp=drive_link)  

[SNAP](https://snap.stanford.edu/)  

[Network Repository](https://networkrepository.com/)

# Idea Finale:

1. implementazione da 0 di page rank modificato con nodi pesati e diretti 
2. utilizzare le reviews e categoria per [Paper](https://arxiv.org/html/2508.14059v1)

?. [amazon-2008](https://networkrepository.com/amazon-2008.php) no info 


## Formulazione alternativa del problema 
in termini di confronto di metodi di classificazione (quindi serve ML)

### Classificazione Induttiva di Prodotti (Embedding vs. Centralità)
Questa è l'idea più forte perché sfrutta i paper sugli embedding e l' idea su PageRank, usando le categorie 
ground-truth del dataset.

**Titolo Proposto** 
"Inductive Product Classification in Co-Purchase Networks: Comparing GraphSAGE against Centrality and Transductive 
Embeddings"

**Paper di Riferimento**
Network embeddings.pdf (principalmente GraphSAGE) e la tua idea di PageRank.

**Contesto**
Il dataset ha le categorie dei prodotti (es. "Libri", "Elettronica"). Possiamo formulare un task di node classification: 
"Data la rete di co-acquisti, riesci a predire la categoria di un prodotto?"

**L'Implementazione (Il Confronto)**
Si implementano e confrontano diversi metodi per generare feature per i nodi, da dare poi in pasto a un classificatore 
(es. Random Forest o MLP):
* Baseline 1 (La tua idea): Si implementa il PageRank Pesato. Si usano i punteggi di PageRank, e magari altre centralità
  (weighted in/out-degree), come feature del nodo.
* Baseline 2 (Classica): Si implementa un metodo transductive come Node2Vec. Bisogna modificare le random walk per
  rispettare i pesi degli archi (è più probabile seguire un arco con peso maggiore).
* Tecnica Avanzata (dal Paper): Si implementa GraphSAGE (da Network embeddings.pdf). GraphSAGE è induttivo. Per le
  feature iniziali dei nodi (che GraphSAGE richiede), si possono usare le feature della Baseline 1 (PageRank, degree).

**La Valutazione (Il Test Chiave)**
Si addestrano i modelli su un sottoinsieme di categorie (es. "Libri" e "Musica").
Si valuta la loro capacità di classificare prodotti in categorie mai viste durante l'addestramento (es. "Videogiochi").

**Ipotesi**
PageRank e Node2Vec (transductive) falliranno o andranno male, perché imparano un embedding per i nodi visti. GraphSAGE 
(inductive) riuscirà a generalizzare, perché impara una funzione per aggregare i vicini.