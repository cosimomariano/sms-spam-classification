# Progetto del corso di Intelligenza Artificiale Scalabile

## SMS Spam Classification System

## Descrizione del problema
Lo scopo del progetto è effettuara la risoluzione di un task di Natural Language Processing (NLP) relativo alla classificazione automatizzata di messaggi SMS. L'obiettivo è produrre un sistema in grado di analizzare testo fornito dall'utente in input e discriminare in modo accurato se un messaggio appartiene alla categoria di spam (indesiderato) o ham (legittimo).

## Obiettivi
L'obiettivo principale del sistema è quello di identificare la natura di un SMS massimizzando la componente di affidabilita nelle predizioni ed ottimizzando i costi computazionali per la risoluzione del task. Ciò avviene attraverso:

1. Ingestione e pulizia dei dati in modalità batch.
2. Rappresentazione numerica del testo tramite tecnica TF-IDF.
3. Riuso di elaborazioni già effettuate per ridurre i tempi di esecuzione e le riesecuzioni ridondanti.

Il sistema restituisce una predizione in tempo reale esponendo anche i metadati associati al modello selezionato attraverso un'API REST (FastAPI).

## Pipeline del sistema
La pipeline del sistema è la seguente ed è strutturata come **Grafo Diretto Aciclico (DAG)**:

 - **Ingestione (Batching)**: lettura del dataset a blocchi (chunk, definiti tramite file di configurazione), validazione, uniformazione e normalizzazione delle colonne.
 - **Preprocessing**: applicazione di regole di normalizzazione e preprocessing per la rimozione di url, normalizzazione di numeri, eliminazione di caratteri speciali ed uniformazione degli spazi.
 - **Estrazione Feature e Vettorizzazione**: utilizzo della TF-IDF per la conversione del testo in una rappresentazione numerica sparsa ad alta dimensionalità incapsulata nella pipeline di classificazione.
 - **Training Parallelo (Fan-out)**: diramazione del workflow per l'addestramento concorrente e concorrente dei tre modelli candidati previsti da progetto: Random Forest, SVM, Naive Bayes.
 - **Model Selection (Fan-in)**: aggregazione delle metriche raccolte (F1-Score, Accuracy, ecc..) di tutti i modelli valutati e selezione del migliore.
 - **Deployment (Model Serving)**: esportazione dell'artefatto ottimale e generazione di un manifesto JSON per l'innesco automatico del microservizio di inferenza (model serving del modello migliore).

## Scelte progettuali principali
 1) Per gestire l'elaborazione out-of-core su dataset di grandi dimensioni è stato implementato il Batching Pattern nella fase di ingestion.
 2) Per gestire i casi in cui uno step sia già stato elaborato è stato implementato lo Step Memorization Pattern, formalizzato con la formula K = H(D,C,P,V[,R]), così da garantire il riuso corretto degli artefatti e un'invalidazione coerente in caso di modifiche a input, configurazione o versione della pipeline.
 3) Per garantire un disaccoppiamento tra documentazione e codice è stato adottato un approccio Contract-First: l'API è stata definita rigorosamente nel file openapi/openapi.yaml, per poi derivare da tale specifica i modelli di validazione e i relativi componenti applicativi.
 4) Per accelerare le fasi computazionali più onerose è stata introdotta l'esecuzione concorrente dei modelli candidati durante la fase di training.
 5) Per gestire il determinismo operativo sono stati fissati i seed in tutte le operazioni stocastiche.
 6) Per ottimizzare il servizio di inferenza è stato introdotto caching in-memory lato serving per il riuso del modello e del manifest già caricati.

## Argomenti del corso trattati
Il progetto copre direttamente i seguenti argomenti teorici trattati nel corso:

 - Workflow pattern (DAG e orchestrazione degli step).
 - Data Ingestion e Batching Pattern.
 - Caching Pattern.
 - Fan-out pattern per la parallelizzazione del training.
 - Fan-in pattern per l'aggregazione delle metriche.
 - Step Memorization Pattern basato su hashing strutturale.
 - Model Serving Pattern e containerizzazione del servizio.

## Struttura del progetto
Il progetto ha la seguente struttura modulare:

```
    |-- app/                 
    |   |-- core/            
    |   |-- generated/       
    |   |-- services/        
    |-- ml/                  
    |   |-- models/          
    |   |-- pipeline/        
    |-- openapi/             
    |-- scripts/  
```

## Come eseguire il progetto
Per eseguire il progetto, sono necessari:

 - Python 3.10+
 - Docker, Docker Compose (opzionale, per il serving)

**I comandi che seguono sono relativi al sistema operativo Windows**:

**Comandi manuali**:
1. Esecuzione della Pipeline di Machine Learning:
    Bash
    `python -m scripts.train_pipeline`

2. Avvio dell'API di classificazione (Locale):
    Bash
    `uvicorn app.main:app --host 0.0.0.0 --port 8000`

**Avvio automatico con docker compose**:
1. Avvio tramite Docker (Consigliato per la produzione):
    Bash
    `docker compose up --build`

## Output del sistema
Il sistema genera artefatti binari dei modelli .joblib e un file manifest strutturato "selected_model.json".
A livello di servizio, l'applicazione espone endpoint REST documentati (Swagger UI accessibile a /docs):

Qui di seguito riporto gli endpoint esposti:
    Metodo di censimento:
        **OPERATION nomeEndpoint**: Breve descrizione della funzione

Endpoints:
    **POST /predict**: Riceve un testo e restituisce la classificazione e il nome del modello responsabile della predizione.
    **GET /health**: Verifica la salute del servizio e lo stato del caricamento in memoria del modello (Caching in-memory).
    **GET /model-metadata**: Espone i dettagli di addestramento del modello correntemente servito, inclusi hash del dataset, metriche di performance e timestamp.

Inoltre, il sistema produce un logging strutturato in formato JSON, ideale per l'ingestione in sistemi di monitoraggio cloud(Kibana, Elastic, ecc..).

## Testing
Sono presenti test automatici eseguiti tramite il framework pytest per gli step principali della pipeline e per il comportamento del servizio.

## Note sul codice
Il codice è rigorosamente strutturato in servizi rispettando il pattern di **Single Responsibility**.
La naming convention di classi, metodi e variabili è in inglese.
I commenti sono presenti esclusivamente laddove necessari per via di logiche molto complesse da leggere.
I commit sono esclusivamente in lingua inglese e sono strutturati come segue:

   `[Iniziali Del Corso]: Breve descrizione sull'attività svolta`

Esempio:
   `[AIS]: Initial commit`

## Autore
    Cosimo Mariano