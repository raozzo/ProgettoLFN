# ProgettoLFN

## Link utili 
[Cartella Drive](https://drive.google.com/drive/folders/14T74Tn7I2E42TKAConzHNnDavb_Vng_T?usp=drive_link)  

[SNAP](https://snap.stanford.edu/)  
[SNAP Dataset](https://snap.stanford.edu/data/amazon-meta.html) 

[Network Repository](https://networkrepository.com/)

# Idea Finale:

L'idea centrale è: "Possiamo usare la struttura della rete di co-acquisto per scoprire le categorie reali dei prodotti? E quali feature (classiche vs. apprese) funzionano meglio?"

Proposta di Progetto: 
## "Clustering Strutturale vs. Appreso sulla Rete di Co-Acquisto Amazon"

Questo progetto confronta l'efficacia di diverse rappresentazioni dei nodi (prodotti) per il compito di clustering, usando le categorie dei prodotti come validazione.

### Fase 1: Data Preparation e Analisi Esplorativa (Il Grafo)

1. Costruzione del Grafo: Carica il dataset. I nodi sono i prodotti. Crea un arco (edge) tra due prodotti se sono "frequently co-purchased" (come definito nel dataset). Questo sarà un grafo non diretto e non pesato (o pesato, se usi il numero di co-acquisti, ma iniziamo semplice).
2. Ground-Truth: Estrai le categorie dei prodotti (es. "Book", "Music", "DVD") dal file di metadati. Questo sarà il tuo ground-truth per la validazione finale del clustering.

### Fase 2: Feature Engineering Strutturale (Le metriche classiche)

Per ogni nodo (prodotto) nel grafo, calcola un vettore di feature basato sulla topologia della rete. Questo ci dice il "ruolo" strutturale di un prodotto.

* Centralità (Le tue metriche):
    1. PageRank: Misura la "popolarità" o "autorevolezza" di un prodotto. Un prodotto è importante se è co-acquistato con altri prodotti importanti.
    2. Betweenness Centrality: Identifica i prodotti "ponte" (broker). Un prodotto con alta betweenness collega cluster di prodotti diversi (es. un libro di "storia" che collega il cluster "storia" e il cluster "politica").
    3. Closeness Centrality: Misura quanto un prodotto è "vicino" (in termini di "salti" di co-acquisto) a tutti gli altri prodotti. Identifica prodotti "centrali" nel catalogo.
* Connettività Locale:
    1. Clustering Coefficient (locale): Misura quanto i "vicini" (prodotti co-acquistati) di un prodotto sono co-acquistati anche tra loro. Un alto C.C. significa che un prodotto appartiene a una "cricca" o community molto fitta (es. il "Signore degli Anelli 1" sarà co-acquistato con "SdA 2" e "SdA 3", che sono anche co-acquistati tra loro).
* Schemi Locali (Graphlets):
    1. Questo è più avanzato e ottimo per una magistrale. Invece di contare solo i triangoli (come il C.C.), usi i graphlets (piccoli sottografi indotti) per creare un "vettore di firme topologiche" per ogni nodo (spesso chiamato Graphlet Degree Vector o Orbit counts). Questo descrive in modo molto ricco la topologia locale di un nodo (es. "questo nodo partecipa a 3 'code' e 2 'quadrati'").

Alla fine di questa fase, ogni prodotto `v` è rappresentato da un vettore: `Feat_Strutturali(v) = [PageRank(v), Betweenness(v), Closeness(v), C.C.(v), Graphlet_Vector(v)]`

### Fase 3: Feature Engineering Appreso (Graph Embeddings)

Qui creiamo una rappresentazione diversa dei nodi, non basata su metriche predefinite, ma "appresa" dalla struttura della rete.

* Random Walks: Applica un algoritmo come Node2Vec (o DeepWalk) al tuo grafo. 
* Training: Questi algoritmi eseguono "passeggiate casuali" sul grafo (simulando un utente che clicca su "prodotti simili") e poi usano un modello tipo Word2Vec per imparare un vettore (embedding) per ogni prodotto.
* Risultato: Ottieni un vettore denso (es. a 128 dimensioni) per ogni prodotto. La cosa fondamentale è che prodotti che si trovano in "quartieri" di rete simili avranno vettori simili nello spazio latente.

Alla fine di questa fase, ogni prodotto v è rappresentato da un altro vettore: `Feat_Appresi(v) = [Embedding_dim_1, ..., Embedding_dim_128]`

### Fase 4: Clustering Comparativo (Il Cuore del Progetto)

Ora hai due (o tre) modi per rappresentare i tuoi prodotti. Il tuo obiettivo è vedere quale rappresentazione cattura meglio la reale divisione in categorie.

Esegui un algoritmo di clustering (es. K-Means o DBSCAN) in tre scenari diversi:
* Clustering A (Solo Appreso): Esegui K-Means solo sui vettori Feat_Appresi(v) (gli embeddings di Node2Vec).
* Clustering B (Solo Strutturale): Esegui K-Means solo sui vettori Feat_Strutturali(v). (Ricorda di normalizzare queste feature!).
* Clustering C (Ibrido): Crea un "super-vettore" concatenando i due: Feat_Ibrido(v) = [Feat_Strutturali(v), Feat_Appresi(v)] ed esegui K-Means su questo.

### Fase 5: Valutazione e Analisi (La Tesi)
Questa è la parte più importante. Come fai a sapere quale clustering è "migliore"? Usi il ground-truth!

**Metrica di Valutazione**: Confronta i cluster che hai trovato (A, B, C) con le categorie reali dei prodotti (il tuo ground-truth dalla Fase 1). Usa metriche di clustering "esterno" come:
    * Adjusted Rand Index (ARI)
    * Normalized Mutual Information (NMI)

Domande di Ricerca a cui Rispondere:
1. Quale metodo (A, B, o C) produce i cluster che corrispondono meglio alle categorie reali dei prodotti?
2. La mia ipotesi: Il metodo A (solo Embeddings) o C (Ibrido) batterà nettamente il B (solo Strutturale). Dimostrarlo è un ottimo risultato.
3. I prodotti "ponte" (alta Betweenness) finiscono ai "bordi" dei cluster K-Means? O vengono messi in cluster "sbagliati"?
4. I prodotti "autorevoli" (alto PageRank) sono i centroidi dei loro cluster?
5. Il Clustering Coefficient medio all'interno dei cluster trovati è più alto della media globale del grafo? (Dovrebbe esserlo, se il clustering ha senso).

Strumenti Consigliati

* Python
* NetworkX: Per caricare il grafo e calcolare tutte le metriche della Fase 2 (PageRank, Betweenness, Closeness, C.C.).
* Libreria node2vec (o Pytorch Geometric): Per calcolare gli embeddings (Fase 3).
* Scikit-learn: Per K-Means e le metriche di valutazione (ARI, NMI) (Fase 4 e 5).