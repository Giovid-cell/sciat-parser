"""
SC-IAT Parser — Web App (Streamlit)
====================================
Interfaccia web sopra il parser `sciat_minnojs_parser.py`.

Trasforma un export grezzo Qualtrics/MinnoJS in un singolo file .xlsx con tre
fogli (Trial_Level, Quality_Report, Summary_Stats), per QUALSIASI SC-IAT con
quella struttura — non solo lo studio sull'aiuto.

L'utente:
  1. carica il CSV grezzo;
  2. sceglie la colonna che contiene i dati SC-IAT e la colonna ID;
  3. etichetta i blocchi rilevati come "congruente" / "incongruente"
     (il suggerimento è precompilato con la logica originale dello studio);
  4. scarica l'Excel.

Avvio:
    streamlit run sciat_webapp.py
"""

import io
import re
import tempfile
import contextlib
from pathlib import Path

import pandas as pd
import streamlit as st

from sciat_minnojs_parser import SCIATParser


# --------------------------------------------------------------------------- #
#  Parser configurabile: le condizioni congruente/incongruente non sono più    #
#  cablate in italiano, ma decise da una mappa fornita dall'utente.            #
# --------------------------------------------------------------------------- #
def _norm(text) -> str:
    """Normalizza un block text rimuovendo tutti gli spazi (chiave della mappa)."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", "", str(text))


class ConfigurableSCIATParser(SCIATParser):
    """Come SCIATParser, ma con mappa blocco->condizione iniettabile dall'esterno."""

    def __init__(self, *args, condition_map: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_map = condition_map or {}

    def _assign_condition(self, block_text) -> str:
        # Se l'utente ha definito una mappa, ha la precedenza assoluta.
        if self.condition_map:
            return self.condition_map.get(_norm(block_text), "unknown")
        # Altrimenti si comporta come l'originale (compatibilità studio "aiuto").
        return super()._assign_condition(block_text)

    def default_guess(self, block_text) -> str:
        """Suggerimento basato sulla logica italiana originale (per pre-compilare la UI)."""
        return SCIATParser._assign_condition(self, block_text)


# --------------------------------------------------------------------------- #
#  Utility                                                                      #
# --------------------------------------------------------------------------- #
def write_temp(uploaded) -> str:
    """Salva il file caricato in un temporaneo e restituisce il path."""
    suffix = Path(uploaded.name).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name


def read_columns(path: str, skip_rows: int) -> list[str]:
    """Legge solo le intestazioni per popolare i menu a tendina."""
    skip = [1, 2] if skip_rows == 2 else list(range(1, skip_rows + 1))
    df = pd.read_csv(
        path, dtype=str, encoding="utf-8-sig", sep=None, engine="python",
        header=0, skiprows=skip, nrows=200,
    )
    df.columns = df.columns.str.strip()
    return list(df.columns), df


def guess_iat_column(df: pd.DataFrame) -> str | None:
    """Indovina la colonna SC-IAT: la prima i cui valori contengono 'block'."""
    for c in df.columns:
        if df[c].astype(str).str.contains("block", na=False, case=False).any():
            return c
    return None


def distinct_block_texts(parser) -> list[str]:
    """Estrae le stringhe di blocco distinte dei blocchi di prova/critici (1-4)."""
    td = parser.trial_data
    block_col = "block" if "block" in td.columns else "Block"
    cond_col = "cond" if "cond" in td.columns else "BlockText"
    blocks = pd.to_numeric(td[block_col], errors="coerce")
    mask = blocks.isin([1, 2, 3, 4])
    texts = (
        td.loc[mask, cond_col].astype(str)
        .str.strip().str.replace(r"[\r\n]+", "", regex=True)
    )
    seen, out = set(), []
    for t in texts:
        if t and t.lower() != "nan" and _norm(t) not in seen:
            seen.add(_norm(t))
            out.append(t)
    return out


def build_excel(parser) -> bytes:
    """Costruisce il workbook a 3 fogli in memoria e restituisce i byte."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        parser.trial_data.to_excel(w, sheet_name="Trial_Level", index=False)
        parser.quality_report.to_excel(w, sheet_name="Quality_Report", index=False)
        parser.summary_data.to_excel(w, sheet_name="Summary_Stats", index=False)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  UI                                                                           #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="SC-IAT Parser", page_icon="🧠", layout="wide")
st.title("🧠 SC-IAT Parser — da dato grezzo a Excel")
st.caption(
    "Carica un export Qualtrics/MinnoJS e ottieni un file .xlsx con tre fogli: "
    "**Trial_Level**, **Quality_Report**, **Summary_Stats**. "
    "Funziona con qualsiasi SC-IAT con questa struttura."
)

uploaded = st.file_uploader("📂 File grezzo (.csv)", type=["csv"])

if uploaded is None:
    st.info("Carica un file CSV per iniziare.")
    st.stop()

path = write_temp(uploaded)

st.subheader("1 · Impostazioni colonne")
c1, c2, c3 = st.columns(3)
with c3:
    skip_rows = st.number_input(
        "Righe di metadati Qualtrics da saltare", min_value=0, max_value=5, value=2,
        help="Gli export Qualtrics hanno 2 righe di metadati sotto l'intestazione.",
    )

try:
    columns, head_df = read_columns(path, skip_rows)
except Exception as e:
    st.error(f"Impossibile leggere il file: {e}")
    st.stop()

iat_guess = guess_iat_column(head_df)
with c1:
    iat_column = st.selectbox(
        "Colonna con i dati SC-IAT",
        options=columns,
        index=columns.index(iat_guess) if iat_guess in columns else 0,
        help="La colonna che contiene il CSV annidato dell'esperimento (contiene 'block').",
    )
with c2:
    id_default = "ResponseId" if "ResponseId" in columns else columns[0]
    id_column = st.selectbox(
        "Colonna ID partecipante",
        options=columns,
        index=columns.index(id_default),
    )

if st.button("🔍 Carica e rileva i blocchi", type="primary"):
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            parser = ConfigurableSCIATParser(
                path, iat_column=iat_column, id_column=id_column, skip_rows=int(skip_rows),
            )
            parser.load_data().parse_all_participants()
        st.session_state["parser"] = parser
        st.session_state["blocks"] = distinct_block_texts(parser)
        st.session_state.pop("xlsx", None)
    except Exception as e:
        st.error(f"Errore durante il caricamento: {e}")
    with st.expander("Log di caricamento"):
        st.code(log.getvalue() or "(nessun output)")

# ---- Mappatura condizioni ------------------------------------------------- #
if "parser" in st.session_state and "blocks" in st.session_state:
    parser = st.session_state["parser"]
    blocks = st.session_state["blocks"]

    st.subheader("2 · Etichetta i blocchi")
    st.caption(
        "Per ogni configurazione di blocco rilevata, indica se è **congruente** o "
        "**incongruente**. Il suggerimento è precompilato; correggilo per il tuo SC-IAT."
    )

    options = ["congruent", "incongruent", "ignora"]
    labels = {"congruent": "congruente", "incongruent": "incongruente", "ignora": "ignora"}
    condition_map: dict[str, str] = {}

    if not blocks:
        st.warning("Nessuna configurazione di blocco rilevata (blocchi 1-4). Controlla le colonne scelte.")
    for i, bt in enumerate(blocks):
        guess = parser.default_guess(bt)
        default_idx = options.index(guess) if guess in ("congruent", "incongruent") else 0
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.code(bt, language=None)
        with col_b:
            choice = st.selectbox(
                "condizione", options, index=default_idx,
                format_func=lambda x: labels[x], key=f"blk_{i}",
                label_visibility="collapsed",
            )
        if choice != "ignora":
            condition_map[_norm(bt)] = choice

    out_name = f"{Path(uploaded.name).stem}_parsed.xlsx"

    if st.button("⚙️ Genera Excel", type="primary"):
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                parser.condition_map = condition_map
                parser.process_trials()
                parser.compute_quality_metrics()
                parser.compute_summary_statistics()
                xlsx = build_excel(parser)
            st.session_state["xlsx"] = xlsx
            st.session_state["xlsx_name"] = out_name
            st.success("Excel generato ✔")
        except Exception as e:
            st.error(f"Errore in elaborazione: {e}")
        with st.expander("Log di elaborazione"):
            st.code(log.getvalue() or "(nessun output)")

# ---- Anteprima + download ------------------------------------------------- #
if "xlsx" in st.session_state:
    parser = st.session_state["parser"]
    st.subheader("3 · Anteprima e download")

    st.download_button(
        "⬇️ Scarica .xlsx",
        data=st.session_state["xlsx"],
        file_name=st.session_state.get("xlsx_name", "SCIAT_parsed.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    t1, t2, t3 = st.tabs(["Trial_Level", "Quality_Report", "Summary_Stats"])
    with t1:
        st.caption(f"{len(parser.trial_data)} righe")
        st.dataframe(parser.trial_data.head(200), use_container_width=True)
    with t2:
        st.caption(f"{len(parser.quality_report)} partecipanti")
        st.dataframe(parser.quality_report, use_container_width=True)
    with t3:
        st.caption(f"{len(parser.summary_data)} partecipanti")
        st.dataframe(parser.summary_data, use_container_width=True)
