"""
ML4N - Project 8: colonne costanti, colonne identiche e coppie di feature molto correlate.

COSA FA
    Carica ddos_dataset.csv, pulisce i nomi delle colonne (toglie gli spazi
    iniziali) ed elimina la colonna indice "Unnamed: 0". Poi cerca le feature
    inutili o ridondanti, senza modificare il dataset:
      1. colonne costanti (un solo valore in tutte le righe)
      2. colonne identiche (stessi valori riga per riga)
      3. coppie di feature con |correlazione di Pearson| > THRESHOLD (0.95)
      4. heatmap della matrice di correlazione

    Le analisi 2-4 usano solo le feature numeriche "di comportamento" (66):
    sono escluse le colonne costanti e gli identificativi
    (Flow ID, IP, Timestamp, porte, Protocol).

OUTPUT A TERMINALE
    - elenco delle colonne costanti e del loro valore
    - elenco delle coppie di colonne identiche
    - coppie con |correlazione| > THRESHOLD, ordinate dalla più alta

OUTPUT SU FILE (nella stessa cartella dello script, sovrascritti a ogni esecuzione)
    correlated_pairs.csv
        - TUTTE le coppie di feature (66 x 65 / 2 = 2.145 righe), non solo
          quelle sopra soglia
        - colonne: feature_1, feature_2, abs_corr (valore assoluto della
          correlazione), ordinate dalla più correlata alla meno correlata
    correlation_heatmap.png
        - matrice di correlazione 66 x 66 (circa 2160 x 1800 pixel)
        - colori: rosso = correlazione positiva, blu = negativa,
          bianco = nessuna correlazione (scala da -1 a +1)

NOTA
    La correlazione di Pearson misura solo relazioni lineari ed è sensibile
    ai valori estremi. Per relazioni monotone non lineari si può usare
    num.corr(method="spearman").
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD = 0.95   # soglia di |correlazione| oltre cui due feature sono "ridondanti"

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ML4N_Project" / "ddos_dataset.csv"
if not CSV_PATH.exists():
    CSV_PATH = BASE_DIR / "ddos_dataset.csv"

df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# 1. colonne costanti
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print(f"COLONNE COSTANTI ({len(const_cols)}):")
for c in const_cols:
    print(f"  {c:25s} sempre = {df[c].iloc[0]}")

# feature numeriche "di comportamento": tolgo costanti e identificativi
ID_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp",
           "Source Port", "Destination Port", "Protocol"]
num = df.select_dtypes(include="number").drop(columns=const_cols + ID_COLS, errors="ignore")

# 2. colonne identiche (stessi valori riga per riga)
print("\nCOLONNE IDENTICHE (duplicati esatti):")
cols = list(num.columns)
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        if num[a].equals(num[b]):
            print(f"  {a}  ==  {b}")

# 3. matrice di correlazione (Pearson) e coppie sopra soglia
corr = num.corr()
upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (upper.stack().sort_values(ascending=False)
         .rename("abs_corr").reset_index()
         .rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
strong = pairs[pairs["abs_corr"] > THRESHOLD]
print(f"\nCOPPIE CON |correlazione| > {THRESHOLD}: {len(strong)}")
print(strong.round(3).to_string(index=False))
pairs.to_csv(BASE_DIR / "correlated_pairs.csv", index=False)

# 4. heatmap
fig, ax = plt.subplots(figsize=(18, 15))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title("Pearson correlation matrix (constant and ID columns removed)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(BASE_DIR / "correlation_heatmap.png", dpi=120)
print("\nSalvati: correlated_pairs.csv, correlation_heatmap.png")
