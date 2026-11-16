# ML4N - Project 8: DDoS attacks detection and characterization

## Struttura

```
ML4N_Project/
├── Documentazione/                   <- traccia spiegata, guida alle feature, readme del dataset
└── project/
    ├── analisi_dataset.ipynb         <- Section 1.1: esplorazione del dataset e motivazione della pulizia
    ├── data/
    │   ├── ddos_dataset.csv          <- dataset originale (non modificarlo)
    │   └── ddos_pulito.csv           <- dataset pulito (notebook §9)
    ├── src/
    │   └── preprocessing.py          <- PULIZIA: carica_dataset() -> 42 feature + label
    ├── results/                      <- output di lavoro del notebook (csv, txt): non vanno nel report
    └── report/
        ├── figures/                  <- GENERATE dal notebook (.pdf per LaTeX, .png per anteprima)
        ├── tables/                   <- GENERATE dal notebook (.tex)
        └── Latex/                    <- SCRITTO A MANO: il report
            ├── main.tex              <- file principale: compilare QUESTO
            ├── .latexmkrc            <- file ausiliari in Latex/build/
            └── sections/             <- una sezione del report per file
```

## Uso

**Analisi (Section 1.1):** aprire `project/analisi_dataset.ipynb` in VS Code o Jupyter, scegliere un
kernel Python con pandas, numpy e matplotlib, e fare *Restart → Run All*. Il notebook rigenera
figure, tabelle, `results/` e `data/ddos_pulito.csv`.

**Dataset pulito negli script di addestramento:**

```python
from preprocessing import carica_dataset
ddos = carica_dataset("data/ddos_dataset.csv")
X, y = ddos.drop(columns="label"), ddos["label"]
```

## Regole

1. Il file `data/ddos_dataset.csv` non si modifica: è il dato grezzo.
2. Non modificare a mano i file in `report/figures` e `report/tables`: si rigenerano dal notebook.
3. In LaTeX si scrive solo il testo, nei file di `report/Latex/sections/` (una sezione per file).
4. Prima di lavorare: `git pull`. Dopo: `git add`, `git commit`, `git push`.

## Compilare il report

- **VS Code + LaTeX Workshop**: aprire `project/report/Latex/main.tex` e salvare.
  Grazie alla riga `% !TEX root = ../main.tex` si può salvare anche mentre si modifica una sezione.
- **Terminale**: `cd project/report/Latex` e `latexmk main.tex`.
- **Overleaf**: caricare l'intera cartella `project/report/` (servono anche `figures/` e `tables/`)
  e impostare `Latex/main.tex` come *Main document*.

Il PDF esce in `project/report/Latex/main.pdf`, i file ausiliari in `project/report/Latex/build/`.

## Requisiti

```
python -m pip install pandas numpy matplotlib jupyter
```
