# Formula 1 – predikcija Top 3 plasmana vozača

Ovaj repozitorijum sadrži Python deo seminarskog rada iz predmeta **Skladištenje podataka i otkrivanje znanja**.

Tema rada je analiza rezultata Formule 1 i predikcija da li će vozač završiti sezonu u **Top 3 plasmanu**.

## Skup podataka

Korišćen je javno dostupan Kaggle skup podataka:

**Formula 1 Races Results Dataset 1950 to 2024**
https://www.kaggle.com/datasets/lakshayjain611/f1-races-results-dataset-1950-to-2024

Korišćeni fajlovi:

* `drivers_updated.csv`
* `winners.csv`
* `fastest_laps_updated.csv`
* `teams_updated.csv`

## Cilj analize

Cilj Python dela rada je da se na osnovu podataka o vozačima, timovima, pobedama i najbržim krugovima napravi klasifikacioni model koji predviđa da li vozač završava sezonu u Top 3 plasmanu.

Ciljna promenljiva je:

* `Top3Plasman = 1` ako je vozač završio sezonu na poziciji 1, 2 ili 3
* `Top3Plasman = 0` ako vozač nije završio sezonu u Top 3

## Korišćeni algoritmi

U radu su primenjena dva klasifikaciona algoritma:

1. Logistička regresija
2. Random Forest

Modeli su poređeni korišćenjem metrika:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

## Priprema podataka

U pripremi podataka urađeno je:

* spajanje podataka iz više CSV fajlova
* izračunavanje broja pobeda po vozaču i sezoni
* izračunavanje broja najbržih krugova po vozaču i sezoni
* kreiranje ciljne promenljive `Top3Plasman`
* kodiranje kategoričkih promenljivih
* podela na trening i test skup
* kros-validacija modela

Iz modela su izbačene kolone `PozicijaVozaca` i `BodoviVozaca`, jer bi direktno otkrivale konačan plasman i dovele do curenja podataka.

## Rezultati

Logistička regresija ostvarila je bolje ukupne rezultate od Random Forest modela. Random Forest je imao nešto veći recall za klasu Top 3, ali je imao slabiju preciznost i niži F1-score.

Na osnovu poređenja modela, logistička regresija je izabrana kao bolji model za ovaj skup podataka.

## Unapređenja u odnosu na osnovni Kaggle dataset

U ovom radu nije samo prikazan dataset, već su dodata sledeća unapređenja:

* formiran je novi skup podataka za problem klasifikacije
* kreirana je ciljna promenljiva `Top3Plasman`
* povezani su podaci o vozačima, pobedama, najbržim krugovima i timovima
* primenjena su dva modela mašinskog učenja
* urađeno je poređenje modela pomoću više metrika
* prikazane su matrice konfuzije i važnost atributa
* urađena je interpretacija rezultata u kontekstu Formule 1

## Fajlovi u repozitorijumu

* `F1_python_analiza.py` – glavni Python kod
* `F1_python_analiza.ipynb` – Jupyter notebook verzija
* `requirements.txt` – potrebne Python biblioteke
* `rezultati_modela.csv` – tabela sa metrikama modela
* `slike_python/` – grafikoni dobijeni analizom
* CSV fajlovi – ulazni podaci korišćeni u analizi
