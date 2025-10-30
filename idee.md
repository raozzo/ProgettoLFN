
# Progetto 1: Analisi dell'"Ecosistema Amazon" (Analisi di Centralità e Community)

**Domanda Chiave:** Quali prodotti sono i "pilastri" dell'ecosistema di acquisto e quali sono le "comunità" di prodotti naturali?

**Concetti Utilizzati:**
* PageRank
* Betweenness Centrality
* Rilevamento di Comunità (es. algoritmo di Louvain)
* Clustering Coefficient (per l'analisi delle comunità)

**Come Farlo:**

1.  **Trova Prodotti "Gateway" (PageRank):**
    * Esegui `PageRank` sul grafico. PageRank è stato progettato per i link web, ma qui è perfetto. Un prodotto con un PageRank elevato non è solo un prodotto acquistato spesso, ma un prodotto a cui "puntano" (vengono co-acquistati con) altri prodotti importanti.
    * **Ipotesi:** I prodotti con il PageRank più alto non saranno prodotti di nicchia e costosi, ma articoli "fondamentali" o "gateway". Pensa a "Cavi HDMI", "Libri di testo fondamentali" o accessori popolari. Questi sono i prodotti che introducono le persone a interi ecosistemi di acquisto.

2.  **Trova Prodotti "Ponte" (Betweenness Centrality):**
    * Esegui la `Betweenness Centrality`. Un prodotto con alta "betweenness" è un prodotto che collega gruppi di prodotti altrimenti disparati.
    * **Ipotesi:** Questi sono prodotti "ponte". Esempio: una "GoPro" (fotocamera) potrebbe collegare la comunità "Accessori da sci", la comunità "Accessori per droni" e la comunità "Accessori per immersioni". È l'hub che unisce questi cluster.

3.  **Scopri le Comunità di Prodotti (Community Detection):**
    * Esegui un algoritmo di rilevamento di comunità (come Louvain o Girvan-Newman) per raggruppare i prodotti in cluster.
    * **L'Analisi (qui sta il bello):** Ora, usa i tuoi metadati! Per ogni community che hai trovato, guarda la `categoria` di prodotto dominante. Hai appena scoperto algoritmicamente "Elettronica", "Libri di cucina", "Attrezzi da giardino", ecc.
    * Puoi anche analizzare la *densità* di queste comunità. Il `Clustering Coefficient` medio all'interno di una comunità ti dice se si tratta di un "kit" stretto (tutti gli articoli vengono acquistati insieme) o di un gruppo più ampio.

---

### Progetto 2: "Kit vs. Utility" (Analisi della Struttura Locale)

**Domanda Chiave:** Possiamo distinguere tra prodotti che fanno parte di un "kit" (dove acquisti tutto il set) e prodotti "di utilità" (che vengono acquistati con molte cose non correlate)?

**Concetti Utilizzati:**
* Clustering Coefficient
* Graphlets (Analisi dei triangoli)

**Come Farlo:**

1.  **Calcola il Clustering Coefficient:** Calcola il coefficiente di clustering locale per ogni nodo (prodotto).
    * Ricorda: questo misura "quanti dei miei vicini sono anche vicini tra loro?"

2.  **Trova i "Kit" (Alto Clustering Coefficient):**
    * Estrai i prodotti con il *coefficiente di clustering più alto*.
    * **Ipotesi:** Questi sono prodotti "kit" o "hobby". Esempio: un "Kit per la preparazione del pane a lievitazione naturale". Se A è il libro, B è il cestino da lievitazione e C è la lama, è molto probabile che gli acquirenti comprino (A, B), (A, C) e (B, C). Questo crea un "triangolo" (graphlet) e un alto coefficiente di clustering.

3.  **Trova le "Utility" (Basso Clustering Coefficient):**
    * Estrai i prodotti con il *coefficiente di clustering più basso* (vicino a 0).
    * **Ipotesi:** Questi sono prodotti "hub" o "di utilità". Esempio: "Batterie Stilo AA". Vengono co-acquistate con "Controller per videogiochi", "Telecomandi" e "Torce elettriche". Ma le persone che comprano torce elettriche e controller per videogiochi non li comprano *insieme*. Quindi, i vicini del nodo "Batterie" non sono connessi tra loro.



---

### Progetto 3: Costruire un Sistema di Raccomandazione (ML su Grafi)

**Domanda Chiave:** Possiamo "imparare" il significato di un prodotto e trovare prodotti simili, usando solo la struttura del grafo?

**Concetti Utilizzati:**
* **Graph Embeddings** (specificamente `Node2Vec` o `GraphSAGE`)

**Come Farlo:**

1.  **Addestra un Modello di Embedding:**
    * Usa una libreria come `Node2Vec`. Questo modello esegue "passeggiate casuali" (random walks) sul tuo grafo di co-acquisto.
    * Impara un **vettore** (es. di 128 dimensioni) per ogni prodotto. I prodotti che si trovano in "quartieri" di rete simili (cioè, che vengono acquistati in contesti simili) finiranno per avere vettori simili.

2.  **Valida gli Embedding (La parte "magica"):**
    * Prendi un prodotto a caso (es. un libro di fantascienza specifico, Prodotto ID: 12345).
    * Trova il suo vettore imparato.
    * Calcola la **similarità coseno** (cosine similarity) tra quel vettore e tutti gli altri vettori di prodotto nel tuo set di dati.
    * Estrai i 5 prodotti più simili.

3.  **Analizza i Risultati:**
    * **Verifica:** I 5 prodotti più simili sono altri libri di fantascienza? O libri dello stesso autore? Se sì, il tuo modello ha "imparato" il concetto di genere e autore *senza mai aver letto una singola parola di testo*, ma solo analizzando con chi veniva co-acquistato.
    * **Visualizza:** Usa `t-SNE` o `UMAP` per ridurre i tuoi vettori da 128 a 2 dimensioni e tracciarli su un grafico a dispersione.
    * **Il Tocco Finale:** Colora i punti su quel grafico usando i metadati della `categoria`. Se il tuo embedding ha funzionato, dovresti vedere cluster chiari e colorati di "Elettronica" in un angolo, "Libri" in un altro e "Cucina" in un altro ancora. Questa è una potente visualizzazione che dimostra che il tuo modello ML ha catturato con successo la struttura del mercato.

---

### Progetto 4 (simile al 3)

**Titolo** Valutazione di GNN Induttivi (GraphSAGE) per la Predizione delle Categorie di Prodotti nel Grafo Co-acquisto di Amazon

**Motivazione**
Implementeremo e valuteremo un modello GNN induttivo, GraphSAGE , per predire la categoria di appartenenza dei prodotti (nodi) in un grande grafo di co-acquisto.
L'obiettivo è analizzare come un modello che apprende campionando e aggregando caratteristiche dal vicinato locale di un nodo possa generare embedding efficaci per nodi mai visti durante l'addestramento (apprendimento induttivo) , un requisito fondamentale per grafi dinamici come quelli dei prodotti e-commerce.

**Dati**
Utilizzeremo il network di co-acquisto di prodotti Amazon, disponibile pubblicamente tramite lo Stanford Network Analysis Project (SNAP).
Link al dataset: https://snap.stanford.edu/data/amazon0601.html
Questo grafo tratta i prodotti come nodi e crea un arco tra prodotti che sono stati acquistati frequentemente insieme.

**Metodo**
Problema: Il problema è definito come classificazione induttiva di nodi (inductive node classification). Data la struttura del grafo di co-acquisto, l'obiettivo è addestrare un modello GNN in grado di predire la categoria di un prodotto (nodo) anche se quel prodotto non era presente nel grafo durante la fase di addestramento.

**Algoritmi**
GraphSAGE : Come descritto nel paper "Inductive Representation Learning on Large Graphs", implementeremo questo framework. GraphSAGE apprende delle funzioni aggregatrici (es. Mean, LSTM, o Pooling aggregator ) che generano embedding per i nodi campionando e aggregando informazioni dai loro vicinati locali.

(Opzionale) Hierarchical Pooling: In una fase successiva, potremmo investigare l'applicazione di moduli di pooling gerarchico come DIFFPOOL (dal paper "Hierarchical Graph Representation Learning" ). Sebbene DIFFPOOL sia primariamente progettato per la classificazione dell'intero grafo (graph classification) , potremmo adattare il concetto per vedere se la creazione di rappresentazioni gerarchiche, che raggruppano (clusterizzano) nodi simili, possa migliorare l'accuratezza della classificazione dei singoli nodi.

**Esperimenti Previsti**
Utilizzeremo le implementazioni di GraphSAGE disponibili (es. quella menzionata nel paper: http://snap.stanford.edu/graphsage/ ) o reimplementeremo gli aggregatori in PyTorch/TensorFlow.

**Macchina per gli esperimenti**
Verrà utilizzata una macchina dotata di GPU (es. NVIDIA K80 ) o un cluster di calcolo universitario per gestire la dimensionalità del grafo Amazon.

**Esperimenti**
Addestreremo i modelli GraphSAGE (con i diversi aggregatori ) su un sottoinsieme di dati del grafo (es. prodotti fino a un certo anno, come nel paper ).
Valuteremo l'accuratezza della classificazione (utilizzando il Micro-F1 score ) su nodi non visti durante l'addestramento (es. prodotti apparsi nell'anno successivo ).
Confronteremo i risultati induttivi di GraphSAGE con baselines transduttive (che richiedono l'intero grafo in fase di training), come DeepWalk , per quantificare il guadagno dell'approccio induttivo rispetto ai metodi che apprendono embedding per nodi fissi.


# IDEA 2: studenti 
capire come funziona il network 

# IDEA 3: Rete stradale california

Il dataset roadNet-CA (~1.9M di nodi, ~2.7M di archi) è un eccellente banco di prova perché è un grafo sparse (a basso grado medio, tipico delle reti stradali) e sufficientemente grande da mettere in difficoltà algoritmi non scalabili.

### Progetto 1: Confronto di Efficienza e Scalabilità nel Community Detection su Larga Scala
**Titolo Proposto:** "Scalability and Quality of Graph Clustering: A Comparative Study of Overlapping and Non-Overlapping Algorithms on the California Road Network"

**Paper di Riferimento Principale:** Network clustering.pdf (per gli algoritmi di overlapping/hybrid conductance).

**Contesto:** Il clustering (community detection) è fondamentale. Nella rete stradale, le "comunità" possono rappresentare quartieri, distretti urbani o intere città. Il paper Network clustering.pdf introduce nuovi algoritmi nearly-linear-time per il clustering overlapping.

**L'Implementazione:**

* Baseline (Classici): Implementare o utilizzare implementazioni standard di algoritmi di clustering non-overlapping ampiamente noti (es. Louvain, Label Propagation). Questi sono noti per la loro velocità.
* Tecnica Avanzata (dal Paper): Implementare l'algoritmo di approssimazione per l'overlapping conductance descritto in Network clustering.pdf.
* Benchmark: Eseguire tutti gli algoritmi sul dataset roadNet-CA.

**La Valutazione:**

* Scalabilità (Tempo e Memoria): Misurare il tempo di esecuzione e l'utilizzo della memoria di picco per ciascun algoritmo. Il dataset roadNet-CA è abbastanza grande da stressare questi algoritmi. L'algoritmo "nearly-linear-time" del paper è davvero più veloce o ha un overhead costante elevato?
* Qualità (Metrica): Poiché non abbiamo una "ground truth" per le comunità stradali, la valutazione della qualità è difficile. Si possono usare metriche interne come la Modularity (per i non-overlapping) e la Conductance o altre metriche di overlapping (come la Overlapping Normalized Mutual Information se potessimo creare un benchmark sintetico simile).
* Analisi Qualitativa: Mappare geograficamente le comunità trovate. Le comunità di Louvain (nette) hanno senso (es. seguono i confini delle contee)? Le comunità overlapping identificate dal nuovo algoritmo si trovano in aree di transizione logiche (es. autostrade che collegano due città distinte)?

**Perché è una buona idea:** Testa direttamente un algoritmo "pratico" e "veloce" da uno dei paper (Network clustering.pdf) contro baseline consolidate su un benchmark reale e di grandi dimensioni, focalizzandosi sulla valutazione delle prestazioni (tempo, memoria, qualità).

--- 

### Progetto 2: Valutazione di Metodi di Embedding per la Predizione di Feature dei Nodi
**Titolo Proposto:** "Inductive vs. Transductive Embeddings for Node Feature Prediction in a Large-Scale Road Network"

**Paper di Riferimento Principale:** Network embeddings.pdf (GraphSAGE).

**Contesto:** Il dataset roadNet-CA è solo topologia. E se volessimo predire una caratteristica di un'intersezione (nodo)? Ad esempio, "questo nodo è un'intersezione autostradale?" o "questo nodo si trova in un'area ad alta densità?". Gli embedding dei nodi (vettori di feature) sono perfetti per questo.

**L'Implementazione:**

1. Creazione del Task:
   * Feature: Generare alcune feature di base per i nodi (es. degree del nodo, clustering coefficient locale).
   * Label (Ground Truth): Creare un task di classificazione. Ad esempio, identificare manualmente (o tramite dati GIS) un sottoinsieme di nodi come "svincoli autostradali" (label=1) e altri come "strade residenziali" (label=0). Questo sarà il nostro target di predizione.
2. Baseline (Transductive): Implementare un metodo di embedding classico "transduttivo" (es. DeepWalk o Node2Vec). Questi metodi imparano un embedding unico per ogni nodo nel grafo di addestramento.
3. Tecnica Avanzata (dal Paper):  Implementare GraphSAGE (da Network embeddings.pdf), che è induttivo. Apprende una funzione per aggregare le feature dei vicini.

**La Valutazione:**
1. Accuratezza (Setting Induttivo): Questo è il test chiave.
    * Addestrare entrambi i modelli su una porzione del grafo (es. la rete stradale della California del Nord).
    * Testare la loro capacità di predire le label dei nodi su una porzione completamente nuova e mai vista del grafo (es. la rete stradale di Los Angeles, o semplicemente un hold-out set).
    * Ipotesi: GraphSAGE dovrebbe funzionare molto meglio perché può generare embedding per nodi "unseen", mentre DeepWalk/Node2Vec non possono.
2. Efficienza: Misurare il tempo di addestramento e, soprattutto, il tempo di inferenza per generare l'embedding di un nuovo nodo (GraphSAGE dovrebbe essere veloce, i metodi transduttivi richiedono un riaddestramento).
3. Robustezza: Valutare come cambiano le prestazioni rimuovendo archi (simulando chiusure stradali). GraphSAGE dovrebbe essere più robusto ai cambiamenti topologici.

**Perché è una buona idea:** Confronta direttamente l'approccio induttivo (fiore all'occhiello di Network embeddings.pdf) con quello transduttivo su un compito pratico (classificazione dei nodi) e su un grafo di grandi dimensioni, evidenziando i vantaggi chiave (generalizzazione a nodi unseen, robustezza).

---

### Progetto 3: Analisi Comparativa di Algoritmi di Conteggio dei Motif
**Titolo Proposto:** "Finding Structural Patterns in the California Road Network: A Performance Benchmark of Motif Counting Algorithms"

**Paper di Riferimento Principale:** Network motifs and patterns.pdf (MOTIVO).

**Contesto:** I motif sono piccoli sottografi (pattern) che appaiono più frequentemente del previsto. Nella rete stradale, i motif possono rappresentare pattern di intersezione comuni: "incroci a 4 vie" (un quadrato C4), "cul-de-sac" (un nodo di grado 1 connesso a uno di grado più alto), "triangoli stradali" (rari ma possibili, C3), o pattern di svincolo più complessi (motif a 4 o 5 nodi).

**L'Implementazione:**

1. Baseline (Esatti): Utilizzare un algoritmo di conteggio esatto dei motif (es. algoritmo di Chiba-Nishizeki per triangoli (k=3), o altri metodi basati su enumerazione per k=4). Questi sono lenti ma accurati.
2. Tecnica Avanzata (dal Paper): Implementare l'algoritmo MOTIVO (da Network motifs and patterns.pdf), che utilizza color coding e sampling adattivo per stimare i conteggi dei motif.
3. Altro Campionamento: Implementare un metodo di campionamento più semplice (es. campionamento casuale di nodi o archi e conteggio dei motif attorno ad essi).

**La Valutazione:**

* Scalabilità e Velocità: Il conteggio esatto su un grafo da ~2M di nodi sarà proibitivo per k > 3 o 4. Misurare il tempo di esecuzione di tutti gli algoritmi per contare i motif di dimensione k=3, 4, 5.
* Accuratezza vs. Velocità (Trade-off):
  * Per k=3 (triangoli), dove il conteggio esatto è fattibile, confrontare l'accuratezza della stima di MOTIVO e del campionamento semplice rispetto al conteggio reale.
  * Dimostrare che MOTIVO (come descritto nel paper) ottiene un'accuratezza molto più elevata rispetto al campionamento semplice a parità di tempo di esecuzione.
*Analisi dei Risultati: Analizzare i conteggi dei motif trovati. Il grafo roadNet-CA è ricco di triangoli (C3)? O è dominato da quadrati (C4 - tipici blocchi urbani)? La distribuzione dei motif è diversa da quella che ci si aspetterebbe in un grafo casuale o in un social network?

**Perché è una buona idea:** Si concentra su un compito fondamentale (conteggio dei pattern) e testa le affermazioni di un paper (Network motifs and patterns.pdf) sulla velocità e l'accuratezza ("scales to much larger graphs") rispetto a metodi classici, utilizzando il dataset roadNet-CA come un perfetto caso di stress-test.