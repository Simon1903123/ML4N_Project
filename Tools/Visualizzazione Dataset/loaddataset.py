"""
ML4N - Project 8: prima occhiata "grezza" alla struttura del dataset.

COSA FA
    Carica ddos_dataset.csv, pulisce i nomi delle colonne (toglie gli spazi
    iniziali) ed elimina la colonna indice "Unnamed: 0". Poi analizza la
    struttura del dataset senza modificarlo.

OUTPUT A TERMINALE (non salvato su file)
    1.  Dimensioni del dataset (righe e colonne)
    2.  Prime 5 righe, trasposte (una feature per riga)
    3.  Ultime 5 righe, trasposte
    3b. Un flusso di esempio per ogni classe, 4 classi alla volta
    4.  Per ogni colonna: tipo, valori distinti, valori mancanti, esempio
    5.  Riepilogo dei tipi di dato
    6.  Colonne con valori NaN o infiniti
    7.  Colonne costanti (un solo valore)
    8.  Numero di righe duplicate
    9.  Numero e percentuale di flussi per classe
    10. Valori più frequenti di Protocol, IP, SimillarHTTP e Inbound
    11. Intervallo temporale coperto dal Timestamp
    12. Statistiche descrittive (describe) delle feature numeriche non costanti

    Per salvare l'output in un file di testo:
        python dataset_structure.py > struttura.txt

OUTPUT SU FILE
    dataset_sample.csv  (nella stessa cartella dello script)
        - primi 20 flussi di ogni classe: 240 righe x 87 colonne
        - separatore virgola, senza colonna indice
        - viene sovrascritto a ogni esecuzione
        - utile per guardare i dati grezzi in Excel o VS Code
          (in Excel: Dati -> Da testo/CSV, delimitatore virgola)
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ML4N_Project" / "ddos_dataset.csv"   # cambia se serve
if not CSV_PATH.exists():                                   # fallback: CSV accanto allo script
    CSV_PATH = BASE_DIR / "ddos_dataset.csv"

# mostra tutte le colonne senza troncarle
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()          # toglie gli spazi nei nomi
df = df.drop(columns=["Unnamed: 0"], errors="ignore")


def title(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


title("1. DIMENSIONI")
print(f"Righe (flussi): {df.shape[0]}   Colonne: {df.shape[1]}")

# Con 87 colonne la tabella non sta in larghezza: la trasponiamo
# (ogni riga = una feature, ogni colonna = un flusso), cosi si legge dall'alto in basso.
title("2. PRIME 5 RIGHE (trasposte: una feature per riga, un flusso per colonna)")
print(df.head().T.to_string())

title("3. ULTIME 5 RIGHE (trasposte)")
print(df.tail().T.to_string())

# un esempio per classe, molto utile per confrontarle a colpo d'occhio
title("3b. UN FLUSSO DI ESEMPIO PER OGNI CLASSE (trasposto)")
example = (df.groupby("label").head(1).set_index("label")
           .drop(columns=["Flow ID"]).sort_index().T)
# 12 classi non stanno in larghezza: stampa 4 classi alla volta
for i in range(0, example.shape[1], 4):
    print(example.iloc[:, i:i + 4].to_string())
    print()

# salva anche un campione da aprire in Excel / VS Code
sample_path = BASE_DIR / "dataset_sample.csv"
df.groupby("label").head(20).to_csv(sample_path, index=False)
print(f"\nCampione (20 flussi per classe) salvato in: {sample_path}")

title("4. ELENCO COLONNE, TIPO, VALORI DISTINTI, ESEMPIO")
info = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_unique": df.nunique(),
    "n_missing": df.isna().sum(),
    "example": df.iloc[0],
})
print(info.to_string())

title("5. TIPI DI DATO (RIEPILOGO)")
print(df.dtypes.value_counts())

title("6. VALORI MANCANTI E INFINITI")
num = df.select_dtypes(include="number")
n_inf = np.isinf(num).sum()
print("Colonne con NaN:", df.columns[df.isna().any()].tolist() or "nessuna")
print("Colonne con inf:", n_inf[n_inf > 0].to_dict() or "nessuna")

title("7. COLONNE COSTANTI (UN SOLO VALORE)")
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
for c in const_cols:
    print(f"  {c}  ->  valore unico: {df[c].iloc[0]}")

title("8. RIGHE DUPLICATE")
print(df.duplicated().sum())

title("9. FLUSSI PER CLASSE (label)")
counts = df["label"].value_counts()
print(pd.DataFrame({"flussi": counts, "%": (100 * counts / len(df)).round(2)}))

title("10. COLONNE CATEGORICHE / IDENTIFICATIVE")
for c in ["Protocol", "Source IP", "Destination IP", "SimillarHTTP", "Inbound"]:
    print(f"\n{c}  ({df[c].nunique()} valori distinti) - i 5 piu frequenti:")
    print(df[c].value_counts().head(5).to_string())

title("11. INTERVALLO TEMPORALE")
ts = pd.to_datetime(df["Timestamp"])
print(f"Da {ts.min()} a {ts.max()}  (durata: {ts.max() - ts.min()})")

title("12. STATISTICHE DESCRITTIVE DELLE FEATURE NUMERICHE")
print(num.drop(columns=const_cols, errors="ignore").describe().T.to_string())