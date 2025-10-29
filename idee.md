
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


# IDEA 2: studenti 
capire come funziona il network 

# IDEA 3: 