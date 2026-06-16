"""
SC-IAT Parser - Web App (Streamlit)
===================================
Converte un export grezzo Qualtrics/MinnoJS di uno SC-IAT in un file Excel con
tre fogli (Trial_Level, Quality_Report, Summary_Stats), per qualsiasi SC-IAT con
questa struttura.

Lo scoring del D-score e' interamente configurabile dalla barra laterale: ogni
ricercatore puo' scegliere la variante dell'algoritmo che preferisce.

Avvio:
    streamlit run sciat_webapp.py
"""

import io
import os
import re
import tempfile
import contextlib
from pathlib import Path

import pandas as pd
import streamlit as st

from sciat_minnojs_parser import SCIATParser
from sciat_scoring import ScoringConfig, score

try:
    import pyreadstat
    HAS_SAV = True
except Exception:
    HAS_SAV = False


# --------------------------------------------------------------------------- #
#  Parser con condizioni configurabili (congruente/incongruente per qualsiasi  #
#  stimolo, non solo lo studio sull'aiuto).                                     #
# --------------------------------------------------------------------------- #
def _norm(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", "", str(text))


class ConfigurableSCIATParser(SCIATParser):
    def __init__(self, *args, condition_map: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_map = condition_map or {}

    def _assign_condition(self, block_text) -> str:
        if self.condition_map:
            return self.condition_map.get(_norm(block_text), "unknown")
        return super()._assign_condition(block_text)

    def default_guess(self, block_text) -> str:
        return SCIATParser._assign_condition(self, block_text)


# --------------------------------------------------------------------------- #
#  Utility                                                                      #
# --------------------------------------------------------------------------- #
def write_temp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name


def read_columns(path: str, skip_rows: int):
    skip = [1, 2] if skip_rows == 2 else list(range(1, skip_rows + 1))
    df = pd.read_csv(
        path, dtype=str, encoding="utf-8-sig", sep=None, engine="python",
        header=0, skiprows=skip, nrows=200,
    )
    df.columns = df.columns.str.strip()
    return list(df.columns), df


def guess_iat_column(df: pd.DataFrame):
    for c in df.columns:
        if df[c].astype(str).str.contains("block", na=False, case=False).any():
            return c
    return None


def distinct_block_texts(parser) -> list[str]:
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


def workbook_xlsx(sheets: dict) -> bytes:
    """Workbook .xlsx con piu' fogli (dict nome -> DataFrame)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    buf.seek(0)
    return buf.getvalue()


def sheet_xlsx(df: pd.DataFrame, name: str) -> bytes:
    """Workbook .xlsx con un solo foglio."""
    return workbook_xlsx({name: df})


def _sanitize_varname(name: str) -> str:
    """Nome variabile valido per SPSS (lettere/numeri/underscore, max 64)."""
    n = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    if not n or not n[0].isalpha():
        n = "v_" + n
    return n[:64]


def sheet_sav(df: pd.DataFrame) -> bytes | None:
    """Esporta un DataFrame in formato SPSS .sav (None se pyreadstat manca)."""
    if not HAS_SAV:
        return None
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == bool:
            d[c] = d[c].astype(int)
        elif d[c].dtype == object:
            d[c] = d[c].fillna("").astype(str)
    d.columns = [_sanitize_varname(c) for c in d.columns]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sav")
    tmp.close()
    try:
        pyreadstat.write_sav(d, tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
    finally:
        os.unlink(tmp.name)
    return data


# --------------------------------------------------------------------------- #
#  Barra laterale: opzioni di scoring                                           #
# --------------------------------------------------------------------------- #
def scoring_sidebar() -> ScoringConfig:
    st.sidebar.header("Opzioni di scoring")
    st.sidebar.caption(
        "Imposta la variante dell'algoritmo. I valori predefiniti seguono "
        "Karpinski & Steinman (2006)."
    )

    st.sidebar.subheader("Blocchi inclusi nelle analisi")
    include_practice = st.sidebar.checkbox(
        "Includi i blocchi di pratica", value=False,
        help="Se disattivo, tutte le stime (D, tassi, esclusioni) usano solo i "
             "blocchi critici. Se attivo, includono anche i blocchi di pratica.",
    )

    st.sidebar.subheader("Filtri sui tempi di risposta")
    st.sidebar.caption("I trial fuori soglia non vengono eliminati: diventano NA.")
    remove_fast = st.sidebar.checkbox("Escludi risposte troppo rapide", value=True)
    fast_threshold = st.sidebar.number_input(
        "Soglia inferiore (ms)", value=350, step=10, disabled=not remove_fast,
    )
    remove_slow = st.sidebar.checkbox("Escludi risposte troppo lente", value=True)
    slow_threshold = st.sidebar.number_input(
        "Soglia superiore (ms)", value=1500, step=50, disabled=not remove_slow,
    )

    st.sidebar.subheader("Trattamento degli errori")
    error_mode_label = st.sidebar.radio(
        "Nelle medie di condizione",
        ["Penalita': media di blocco + ms", "Escludi i trial errati"],
    )
    error_mode = "penalty" if error_mode_label.startswith("Penalita") else "exclude"
    penalty_ms = st.sidebar.number_input(
        "Penalita' (ms)", value=400, step=50, disabled=error_mode != "penalty",
    )

    st.sidebar.subheader("Calcolo del D-score")
    sd_labels = {
        "SD di tutti i trial corretti (K&S 2006)": "all_correct",
        "SD pooled delle due condizioni": "pooled",
        "SD su tutti i trial, errori penalizzati inclusi": "all_with_penalty",
    }
    sd_choice = st.sidebar.selectbox("Denominatore (SD)", list(sd_labels.keys()))
    sd_method = sd_labels[sd_choice]

    sign_labels = {
        "(M incongruente - M congruente) / SD": "incong_minus_cong",
        "(M congruente - M incongruente) / SD": "cong_minus_incong",
    }
    sign_choice = st.sidebar.radio("Direzione del D", list(sign_labels.keys()))
    d_sign = sign_labels[sign_choice]

    st.sidebar.subheader("Esclusione partecipanti")
    st.sidebar.caption("I partecipanti esclusi vengono segnalati, non rimossi.")
    use_err = st.sidebar.checkbox("Esclusione per tasso di errore", value=True)
    err_pct = st.sidebar.number_input(
        "Soglia errori (%)", value=10.0, min_value=0.0, max_value=100.0, step=1.0,
        disabled=not use_err,
    )
    use_to = st.sidebar.checkbox("Esclusione per tasso di timeout", value=True)
    to_pct = st.sidebar.number_input(
        "Soglia timeout (%)", value=8.33, min_value=0.0, max_value=100.0, step=1.0,
        disabled=not use_to,
    )
    min_valid = st.sidebar.number_input("Minimo trial validi", value=29, step=1)
    drop_excluded = st.sidebar.checkbox(
        "Rimuovi davvero i partecipanti esclusi", value=False,
        help="Se disattivo, restano nell'output con il flag EXCLUDE (merge piu' semplice).",
    )

    return ScoringConfig(
        include_practice=include_practice,
        remove_fast=remove_fast, fast_threshold=float(fast_threshold),
        remove_slow=remove_slow, slow_threshold=float(slow_threshold),
        error_mode=error_mode, penalty_ms=float(penalty_ms),
        sd_method=sd_method, d_sign=d_sign,
        use_error_rate_exclusion=use_err, error_rate_threshold=err_pct / 100.0,
        use_timeout_exclusion=use_to, max_timeout_rate=to_pct / 100.0,
        min_valid_trials=int(min_valid), drop_excluded=drop_excluded,
    )


# --------------------------------------------------------------------------- #
#  Interfaccia principale                                                       #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="SC-IAT Parser", layout="wide")
st.title("SC-IAT Parser")
st.write(
    "Converte un export grezzo Qualtrics/MinnoJS in un file Excel con tre fogli: "
    "**Trial_Level**, **Quality_Report**, **Summary_Stats**. "
    "Compatibile con qualsiasi SC-IAT che abbia questa struttura."
)

cfg = scoring_sidebar()

uploaded = st.file_uploader("File grezzo (.csv)", type=["csv"])
if uploaded is None:
    st.info("Carica un file CSV per iniziare.")
    st.stop()

path = write_temp(uploaded)

st.divider()
st.subheader("1 - Impostazioni colonne")
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
        "Colonna con i dati SC-IAT", options=columns,
        index=columns.index(iat_guess) if iat_guess in columns else 0,
        help="La colonna che contiene il CSV annidato dell'esperimento (contiene 'block').",
    )
with c2:
    id_default = "ResponseId" if "ResponseId" in columns else columns[0]
    id_column = st.selectbox(
        "Colonna ID partecipante", options=columns, index=columns.index(id_default),
    )

if st.button("Carica e rileva i blocchi", type="primary"):
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            parser = ConfigurableSCIATParser(
                path, iat_column=iat_column, id_column=id_column, skip_rows=int(skip_rows),
            )
            parser.load_data().parse_all_participants()
        st.session_state["parser"] = parser
        st.session_state["blocks"] = distinct_block_texts(parser)
        st.session_state.pop("result", None)
    except Exception as e:
        st.error(f"Errore durante il caricamento: {e}")
    with st.expander("Log di caricamento"):
        st.code(log.getvalue() or "(nessun output)")

# ---- Mappatura condizioni ------------------------------------------------- #
if "parser" in st.session_state and "blocks" in st.session_state:
    parser = st.session_state["parser"]
    blocks = st.session_state["blocks"]

    st.divider()
    st.subheader("2 - Etichetta i blocchi")
    st.caption(
        "Per ogni configurazione di blocco rilevata indica se e' congruente o "
        "incongruente. Il suggerimento e' precompilato."
    )

    options = ["congruent", "incongruent", "ignora"]
    labels = {"congruent": "congruente", "incongruent": "incongruente", "ignora": "ignora"}
    condition_map: dict[str, str] = {}

    if not blocks:
        st.warning("Nessuna configurazione di blocco rilevata. Controlla le colonne scelte.")
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

    st.divider()
    st.subheader("3 - Genera l'output")
    st.caption("Le opzioni di scoring sono nella barra laterale a sinistra.")
    if st.button("Genera output", type="primary"):
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                parser.condition_map = condition_map
                parser.process_trials()
                trials, summary, quality = score(parser.trial_data, cfg)
                sheets = {
                    "Trial_Level": trials,
                    "Quality_Report": quality,
                    "Summary_Stats": summary,
                }
                combined = workbook_xlsx(sheets)
                per_sheet = {
                    name: {"xlsx": sheet_xlsx(df, name), "sav": sheet_sav(df)}
                    for name, df in sheets.items()
                }
            st.session_state["result"] = {
                "trials": trials, "summary": summary, "quality": quality,
                "combined": combined, "per_sheet": per_sheet,
                "stem": Path(uploaded.name).stem,
            }
            st.success("Output generato.")
        except Exception as e:
            st.error(f"Errore in elaborazione: {e}")
        with st.expander("Log di elaborazione"):
            st.code(log.getvalue() or "(nessun output)")

# ---- Anteprima + download ------------------------------------------------- #
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def sheet_download_buttons(res, sheet_name):
    """Due bottoni (xlsx, sav) per un singolo foglio."""
    files = res["per_sheet"][sheet_name]
    stem = res["stem"]
    b1, b2 = st.columns(2)
    b1.download_button(
        "Scarica foglio (.xlsx)", data=files["xlsx"],
        file_name=f"{stem}_{sheet_name}.xlsx", mime=XLSX_MIME,
        key=f"dl_xlsx_{sheet_name}",
    )
    if files["sav"] is not None:
        b2.download_button(
            "Scarica foglio (.sav)", data=files["sav"],
            file_name=f"{stem}_{sheet_name}.sav", mime="application/octet-stream",
            key=f"dl_sav_{sheet_name}",
        )
    else:
        b2.button("SPSS .sav non disponibile", disabled=True, key=f"nosav_{sheet_name}")


if "result" in st.session_state:
    res = st.session_state["result"]
    st.divider()
    st.subheader("4 - Risultati")

    d = res["summary"]["D_Score"].dropna()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Partecipanti", res["summary"]["ParticipantID"].nunique())
    m2.metric("D-score medio", f"{d.mean():.3f}" if len(d) else "n/d")
    m3.metric("Da escludere", int(res["quality"]["EXCLUDE"].sum()))
    m4.metric("Trial totali", len(res["trials"]))

    st.download_button(
        "Scarica tutto (.xlsx, 3 fogli)", data=res["combined"],
        file_name=f"{res['stem']}_parsed.xlsx", mime=XLSX_MIME, type="primary",
    )
    if not HAS_SAV:
        st.caption("Nota: l'export .sav richiede il pacchetto pyreadstat (assente).")

    t1, t2, t3 = st.tabs(["Trial_Level", "Quality_Report", "Summary_Stats"])
    with t1:
        st.caption(f"{len(res['trials'])} righe")
        sheet_download_buttons(res, "Trial_Level")
        st.dataframe(res["trials"].head(300), use_container_width=True)
    with t2:
        st.caption(f"{len(res['quality'])} partecipanti")
        sheet_download_buttons(res, "Quality_Report")
        st.dataframe(res["quality"], use_container_width=True)
    with t3:
        st.caption(f"{len(res['summary'])} partecipanti")
        sheet_download_buttons(res, "Summary_Stats")
        st.dataframe(res["summary"], use_container_width=True)
