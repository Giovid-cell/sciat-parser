# SC-IAT Parser

Web app che trasforma un export grezzo Qualtrics/MinnoJS di uno SC-IAT in un
file Excel (`.xlsx`) con tre fogli: **Trial_Level**, **Quality_Report**,
**Summary_Stats**. Funziona per qualsiasi SC-IAT con questa struttura.

App online: _(incolla qui il link dopo il deploy)_

## Come si usa

1. Apri il link dell'app nel browser.
2. Carica il file `.csv` grezzo.
3. Controlla le impostazioni (colonna dati, colonna ID, righe da saltare).
4. Premi **"Carica e rileva i blocchi"** ed etichetta congruente/incongruente
   (di solito è già corretto).
5. Premi **"Genera Excel"** e scarica il file.

## Eseguire in locale (per sviluppatori)

```
pip install -r requirements.txt
streamlit run sciat_webapp.py
```

## Nota sulla privacy

Questo repository contiene **solo codice**. I dati dei partecipanti non vanno
mai caricati qui: il file `.gitignore` esclude tutti i `.csv`/`.xlsx`. I file
caricati nell'app vengono elaborati in memoria e non sono salvati nel repo.
