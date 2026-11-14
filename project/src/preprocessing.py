"""
Pulizia del dataset DDoS.

Lo script legge il CSV con i flussi di traffico di rete, elimina le colonne
inutili o inaffidabili, corregge i pochi valori anomali rimasti e verifica
che il risultato sia pronto per l'addestramento di un modello.

Ogni riga del dataset è un flusso (una sequenza di pacchetti scambiati tra
due indirizzi IP) e la colonna "label" indica se si tratta di traffico
normale ("benign") o di uno degli 11 tipi di attacco DDoS.

Uso:
  - stampa un riepilogo e restituisce il DataFrame già pulito (ddos).
"""

import pandas as pd
import numpy as np

# Percorso del file CSV iniziale (sporco).
PERCORSO_CSV = "ddos_dataset.csv"


# -----------------------------------------------------------------------------
# COLONNE DA RIMUOVERE
# -----------------------------------------------------------------------------
# Elenco delle colonne che non verranno usate dal modello. Per escluderne
# un'altra basta aggiungerla qui; per reintrodurla basta toglierla o commentarla.
#
# Il commento accanto a ogni nome spiega il motivo dell'esclusione:
#   ESCL = identificativi (IP, ID del flusso, orario): descrivono *chi* e
#          *quando*, non *come* si comporta il traffico. Tenerli farebbe
#          imparare al modello gli indirizzi dell'esperimento invece dei
#          pattern degli attacchi.
#   RID  = ridondanti: contengono un'informazione già presente in un'altra
#          colonna (ad esempio una media ricavabile da totali già inclusi).
#   ELIM = poco informative: quasi sempre costanti o duplicate, non aiutano
#          a distinguere le classi.
#   CORR = corrotte: contengono valori negativi impossibili dovuti a un
#          overflow del tool che ha estratto le feature. Questi valori
#          compaiono quasi solo negli attacchi, quindi tenerle porterebbe il
#          modello a riconoscere il difetto invece del traffico.
#
# Attenzione alle virgole: in Python due stringhe scritte una dopo l'altra
# senza virgola vengono unite in un'unica stringa, e la colonna non verrebbe
# rimossa. Per questo ogni riga termina con una virgola, anche l'ultima.
COLONNE_DA_RIMUOVERE = [
    "Flow ID",                      # ESCL
    "Destination IP",               # ESCL
    "Source IP",                    # ESCL
    "Timestamp",                    # ESCL
    # "Source Port",                # ESCL? porta scelta a caso dal sistema operativo:
    #                               #       decommentare per escluderla
    "Total Length of Bwd Packets",  # RID
    "Fwd Packet Length Min",        # RID
    "Fwd Packet Length Mean",       # RID
    "Min Packet Length",            # RID
    "Packet Length Mean",           # RID
    "Packet Length Std",            # RID
    "Packet Length Variance",       # RID
    "Average Packet Size",          # RID
    "Avg Fwd Segment Size",         # RID
    "Avg Bwd Segment Size",         # RID
    "Fwd Packets/s",                # RID
    "Fwd IAT Total",                # RID
    "Fwd IAT Mean",                 # RID
    "Fwd IAT Std",                  # RID
    "Fwd IAT Max",                  # RID
    "Fwd IAT Min",                  # RID
    "Bwd IAT Std",                  # RID
    "Idle Max",                     # RID
    "Idle Min",                     # RID
    "FIN Flag Count",               # ELIM
    "RST Flag Count",               # ELIM
    "PSH Flag Count",               # ELIM
    "ECE Flag Count",               # ELIM
    "Bwd PSH Flags",                # ELIM
    "Fwd URG Flags",                # ELIM
    "Bwd URG Flags",                # ELIM
    "Fwd Header Length.1",          # ELIM
    "Fwd Avg Bytes/Bulk",           # ELIM
    "Fwd Avg Packets/Bulk",         # ELIM
    "Fwd Avg Bulk Rate",            # ELIM
    "Bwd Avg Bytes/Bulk",           # ELIM
    "Bwd Avg Packets/Bulk",         # ELIM
    "Bwd Avg Bulk Rate",            # ELIM
    "Subflow Fwd Packets",          # ELIM
    "Subflow Fwd Bytes",            # RID
    "Subflow Bwd Packets",          # ELIM
    "Subflow Bwd Bytes",            # RID
    "act_data_pkt_fwd",             # RID
    "Fwd Header Length",            # CORR
    "min_seg_size_forward",         # CORR
]

# Colonne in cui il valore -1 è legittimo e non va considerato un errore.
# Indica che la finestra TCP non è disponibile, cosa normale per il traffico
# UDP (molto frequente negli attacchi DDoS). Il controllo finale sui valori
# negativi salta queste colonne.
COLONNE_CON_MENO_UNO_VALIDO = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]


# -----------------------------------------------------------------------------
# CARICAMENTO E PULIZIA
# -----------------------------------------------------------------------------
def carica_dataset(percorso=PERCORSO_CSV, verbose=True):
    """
    Legge il CSV, lo pulisce e restituisce un DataFrame pronto per il modello.

    Parametri:
      percorso: percorso del file CSV da leggere.
      verbose:  se True stampa la dimensione finale e quanti esempi ci sono
                per ogni classe.
    """

    # --- 1. Lettura del file -------------------------------------------------
    # low_memory=False fa leggere ogni colonna per intero prima di decidere il
    # suo tipo. Senza, pandas legge a blocchi e su colonne con valori misti
    # (come SimillarHTTP) può assegnare tipi diversi a parti diverse.
    ddos = pd.read_csv(percorso, sep=",", low_memory=False)

    # Rimuove eventuali spazi all'inizio e alla fine dei nomi delle colonne
    # (es. " Source IP" -> "Source IP"), così corrispondono a quelli della lista.
    ddos.columns = ddos.columns.str.strip()

    # Colonna indice. Nel CSV fornito ogni riga di dati ha un campo in più
    # rispetto all'intestazione (88 contro 87): è un indice senza nome, che
    # pandas usa automaticamente come indice delle righe, per cui non diventa
    # una colonna. Se invece il file avesse un'intestazione che inizia con un
    # nome vuoto (",Flow ID,..."), pandas creerebbe la colonna "Unnamed: 0":
    # verrebbe tenuta come feature numerica e, essendo diversa in ogni riga,
    # impedirebbe anche di trovare i duplicati. La si elimina solo se esiste.
    if "Unnamed: 0" in ddos.columns:
        ddos = ddos.drop(columns="Unnamed: 0")

    # --- 2. Rimozione delle colonne ------------------------------------------
    # Prima di rimuovere controlla che ogni nome della lista esista davvero
    # nel CSV: serve a scoprire subito refusi o virgole dimenticate, che
    # altrimenti passerebbero inosservati.
    mancanti = [c for c in COLONNE_DA_RIMUOVERE if c not in ddos.columns]
    if mancanti:
        print("ATTENZIONE - colonne non trovate nel CSV:", mancanti)

    # errors="ignore" evita che lo script si blocchi per un nome mancante:
    # l'avviso qui sopra basta a segnalarlo.
    ddos = ddos.drop(columns=COLONNE_DA_RIMUOVERE, errors="ignore")

    # --- 3. Correzione dei valori anomali ------------------------------------
    # Bwd Header Length è una lunghezza in byte, quindi non può essere negativa.
    # Nel dataset c'è un solo valore negativo (overflow): lo si trasforma in
    # NaN e si elimina la riga, perdendo un unico esempio.
    # Il controllo "in ddos.columns" serve se la colonna viene aggiunta a
    # COLONNE_DA_RIMUOVERE (è ridondante con Total Backward Packets): senza,
    # questo passo andrebbe in KeyError. Se la colonna è rimossa, la riga resta.
    if "Bwd Header Length" in ddos.columns:
        ddos.loc[ddos["Bwd Header Length"] < 0, "Bwd Header Length"] = np.nan
        ddos = ddos.dropna(subset=["Bwd Header Length"])

    # SimillarHTTP contiene 0 nella quasi totalità delle righe e, nelle poche
    # restanti, l'URL della richiesta HTTP. Il testo dell'URL non è utile al
    # modello (sono indirizzi specifici dell'esperimento), quindi la colonna
    # diventa binaria: 1 se il flusso contiene una richiesta HTTP, 0 altrimenti.
    # La conversione in stringa serve perché lo 0 compare sia come numero sia
    # come testo "0".
    ddos["SimillarHTTP"] = (ddos["SimillarHTTP"].astype(str).str.strip() != "0").astype(int)

    # --- 4. Rimozione dei duplicati ------------------------------------------
    # Righe identiche potrebbero finire sia nel training sia nel test set,
    # facendo sembrare il modello più bravo di quanto sia. reset_index
    # rinumera le righe da 0 dopo le eliminazioni.
    ddos = ddos.drop_duplicates().reset_index(drop=True)

    # --- 5. Controlli finali -------------------------------------------------
    # Ogni assert blocca lo script con un messaggio se la condizione è falsa.
    # Servono a garantire che il dataset sia davvero pulito prima di usarlo.
    numeriche = ddos.select_dtypes("number")

    # Nessun valore mancante.
    assert ddos.isna().sum().sum() == 0, "Ci sono ancora NaN"

    # Nessun valore infinito (tipici delle colonne "al secondo" quando la
    # durata del flusso è zero, perché si divide per zero).
    assert not np.isinf(numeriche).values.any(), "Ci sono valori infiniti"

    # Nessun valore negativo, escluse le colonne in cui -1 è legittimo.
    # In caso di errore il messaggio elenca le colonne colpevoli.
    negativi = (numeriche.drop(columns=COLONNE_CON_MENO_UNO_VALIDO, errors="ignore") < 0).sum()
    negativi = negativi[negativi > 0]
    assert negativi.empty, f"Valori negativi inattesi:\n{negativi}"

    # Tutte le feature devono essere numeriche: i modelli non accettano testo.
    # L'unica colonna di testo ammessa è la label.
    non_numeriche = [c for c in ddos.columns if c != "label" and c not in numeriche.columns]
    assert not non_numeriche, f"Colonne non numeriche: {non_numeriche}"

    # --- 6. Riepilogo --------------------------------------------------------
    if verbose:
        print(f"Shape finale: {ddos.shape}")    # (righe, colonne)
        print(ddos["label"].value_counts())     # numero di esempi per classe

    return ddos


# -----------------------------------------------------------------------------
# ESECUZIONE DIRETTA
# -----------------------------------------------------------------------------
# Questo blocco viene eseguito solo lanciando il file direttamente, non quando
# viene importato da un altro script.
if __name__ == "__main__":
    ddos = carica_dataset()

    # Separazione tra feature (X, gli input del modello) e target (y, la classe da prevedere).
    X = ddos.drop(columns=["label"])
    y = ddos["label"]
    print(f"Feature: {X.shape[1]}  |  Campioni: {X.shape[0]}")

    # Writing a DataFrame to CSV
    # index=False prevents pandas from writing the row labels as an empty-named column
    # (riga spostata dentro il blocco: fuori, veniva eseguita anche a ogni
    # "import preprocessing" e, non esistendo ancora "ddos", l'import falliva
    # con NameError)
    ddos.to_csv("formatted_data.csv", index=False)
