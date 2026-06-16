"""
SC-IAT scoring configurabile
============================
Logica di scoring del D separata dal parsing, così ogni ricercatore può
scegliere la variante dell'algoritmo che preferisce (esistono piu' convenzioni).

Consuma il `trial_data` prodotto dal parser (colonne: ParticipantID, Block,
Trial, Condition, Category, TrialType, Error, RT, IsCritical, IsPractice) e
produce:
  - trial_data arricchito (RT_clean = NA per i trial filtrati; RT_scored = valore
    effettivamente usato, con penalita');
  - summary per partecipante (D-score);
  - quality report con i flag di esclusione.

Tutte le stime (D, error/timeout/anticipation rate, esclusioni) usano lo stesso
insieme di trial "di analisi": solo i blocchi critici, oppure critici + pratica
se include_practice e' True.

Convenzioni codifica MinnoJS:  Error 1 = corretto, 0 = errore, 2 = timeout.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ScoringConfig:
    """Tutte le scelte metodologiche, ognuna esposta come toggle nell'app."""

    # --- Quali blocchi entrano nelle analisi ---
    include_practice: bool = False    # False = solo critici; True = critici + pratica

    # --- Filtri sui tempi di risposta (a livello di trial -> NA) ---
    remove_fast: bool = True          # togli RT <= soglia (anticipazioni)
    fast_threshold: float = 350.0
    remove_slow: bool = True          # togli RT >= soglia (lente/timeout)
    slow_threshold: float = 1500.0

    # --- Gestione errori nelle MEDIE di condizione ---
    error_mode: str = "penalty"       # "penalty" | "exclude"
    penalty_ms: float = 400.0         # penalita' aggiunta alla media di blocco

    # --- Denominatore del D (la SD) ---
    sd_method: str = "all_correct"    # "all_correct" | "pooled" | "all_with_penalty"

    # --- Direzione del D ---
    d_sign: str = "incong_minus_cong"  # "incong_minus_cong" | "cong_minus_incong"

    # --- Esclusione partecipanti (solo flag, salvo drop_excluded) ---
    use_error_rate_exclusion: bool = True
    error_rate_threshold: float = 0.10
    use_timeout_exclusion: bool = True
    max_timeout_rate: float = 0.0833
    min_valid_trials: int = 29
    drop_excluded: bool = False        # se True rimuove davvero gli esclusi


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Block", "Trial", "RT", "Error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def analysis_mask(df: pd.DataFrame, cfg: ScoringConfig) -> pd.Series:
    """Trial inclusi nelle analisi: critici, piu' pratica se richiesto."""
    mask = df["IsCritical"] == True
    if cfg.include_practice:
        mask = mask | (df["IsPractice"] == True)
    return mask


def apply_trial_filters(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Aggiunge RT_clean (NA per i trial filtrati), motivo e flag di esclusione."""
    df = _ensure_numeric(df)

    rt = df["RT"]
    err = df["Error"]

    is_timeout = err == 2
    is_fast = cfg.remove_fast & (rt <= cfg.fast_threshold)
    is_slow = cfg.remove_slow & (rt >= cfg.slow_threshold)

    reason = pd.Series("", index=df.index, dtype=object)
    reason[is_slow] = "slow"
    reason[is_fast] = "fast"          # fast ha priorita' sul motivo se sovrapposto
    reason[is_timeout] = "timeout"

    excluded = is_timeout | is_fast | is_slow

    df["RT_clean"] = rt.where(~excluded, other=np.nan)
    df["Trial_Excluded"] = excluded
    df["Exclusion_Reason"] = reason
    return df


def _block_means(scope: pd.DataFrame) -> dict:
    """Media delle latenze CORRETTE (RT_clean) per partecipante x condizione."""
    correct = scope[scope["Error"] == 1]
    return correct.groupby(["ParticipantID", "Condition"])["RT_clean"].mean().to_dict()


def _add_rt_scored(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """RT_scored = valore usato nello scoring (corretti -> RT_clean; errori ->
    media blocco + penalita' se in modalita' penalty; altrimenti NA)."""
    in_scope = analysis_mask(df, cfg)
    scope = df[in_scope]
    means = _block_means(scope)

    rt_scored = df["RT_clean"].copy()

    if cfg.error_mode == "penalty":
        err_mask = in_scope & (df["Error"] == 0) & (~df["Trial_Excluded"])
        bm = df.loc[err_mask].apply(
            lambda r: means.get((r["ParticipantID"], r["Condition"]), np.nan), axis=1
        )
        if len(bm):
            rt_scored.loc[err_mask] = bm.values + cfg.penalty_ms
    else:  # exclude: gli errori non entrano nelle medie
        err_mask = in_scope & (df["Error"] == 0)
        rt_scored.loc[err_mask] = np.nan

    df = df.copy()
    df["RT_scored"] = rt_scored
    return df


def enrich_trials(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Pipeline trial-level completa: filtri + RT_scored."""
    df = apply_trial_filters(df, cfg)
    df = _add_rt_scored(df, cfg)
    return df


def _sd_denominator(scope: pd.DataFrame, cfg: ScoringConfig) -> float:
    if cfg.sd_method == "all_correct":
        vals = scope.loc[scope["Error"] == 1, "RT_clean"].dropna()
        return vals.std(ddof=1) if len(vals) > 1 else np.nan

    if cfg.sd_method == "all_with_penalty":
        vals = scope["RT_scored"].dropna()
        return vals.std(ddof=1) if len(vals) > 1 else np.nan

    # pooled: SD per condizione sui valori usati nelle medie (RT_scored), poi pool
    sd_c = scope.loc[scope["Condition"] == "congruent", "RT_scored"].dropna()
    sd_i = scope.loc[scope["Condition"] == "incongruent", "RT_scored"].dropna()
    v_c = sd_c.var(ddof=1) if len(sd_c) > 1 else np.nan
    v_i = sd_i.var(ddof=1) if len(sd_i) > 1 else np.nan
    if np.isnan(v_c) or np.isnan(v_i):
        return np.nan
    return np.sqrt((v_c + v_i) / 2)


def compute_summary(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Summary per partecipante con D-score secondo la configurazione."""
    rows = []
    for pid, p in df.groupby("ParticipantID"):
        scope = p[analysis_mask(p, cfg)]
        cong = scope[scope["Condition"] == "congruent"]
        incong = scope[scope["Condition"] == "incongruent"]

        m_cong = cong["RT_scored"].dropna().mean() if len(cong) else np.nan
        m_incong = incong["RT_scored"].dropna().mean() if len(incong) else np.nan
        sd = _sd_denominator(scope, cfg)

        if cfg.d_sign == "incong_minus_cong":
            d = (m_incong - m_cong) / sd if sd and not np.isnan(sd) else np.nan
        else:
            d = (m_cong - m_incong) / sd if sd and not np.isnan(sd) else np.nan

        first_cond = (
            scope.sort_values(["Block", "Trial"])["Condition"].iloc[0]
            if len(scope) else "unknown"
        )

        rows.append({
            "ParticipantID": pid,
            "N_congruent": int(cong["RT_scored"].notna().sum()),
            "N_incongruent": int(incong["RT_scored"].notna().sum()),
            "Mean_RT_congruent": round(m_cong, 2) if pd.notna(m_cong) else np.nan,
            "Mean_RT_incongruent": round(m_incong, 2) if pd.notna(m_incong) else np.nan,
            "SD_denominator": round(sd, 2) if pd.notna(sd) else np.nan,
            "D_Score": round(d, 4) if pd.notna(d) else np.nan,
            "First_Condition": first_cond,
        })

    return pd.DataFrame(rows)


def compute_quality(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Quality report con conteggi, tassi (in %) e flag di esclusione."""
    rows = []
    for pid, p in df.groupby("ParticipantID"):
        scope = p[analysis_mask(p, cfg)]

        n_total = len(scope)
        n_correct = int((scope["Error"] == 1).sum())
        n_errors = int((scope["Error"] == 0).sum())
        n_timeout = int((scope["Error"] == 2).sum())
        n_anticip = int((scope["RT"] <= cfg.fast_threshold).sum())
        n_resp = n_correct + n_errors

        error_rate = n_errors / n_resp if n_resp > 0 else np.nan       # proporzione
        timeout_rate = n_timeout / n_total if n_total > 0 else np.nan
        anticip_rate = n_anticip / n_total if n_total > 0 else np.nan
        n_valid = int(scope["RT_scored"].notna().sum())

        has_b2 = 2 in p["Block"].values
        has_b4 = 4 in p["Block"].values

        excl_error = (cfg.use_error_rate_exclusion and not np.isnan(error_rate)
                      and error_rate > cfg.error_rate_threshold)
        excl_timeout = (cfg.use_timeout_exclusion and not np.isnan(timeout_rate)
                        and timeout_rate >= cfg.max_timeout_rate)
        excl_few = n_valid < cfg.min_valid_trials
        excl_missing = not (has_b2 and has_b4)
        exclude = bool(excl_error or excl_timeout or excl_few or excl_missing)

        reasons = []
        if excl_error:
            reasons.append(f"error_rate>{cfg.error_rate_threshold:.0%}")
        if excl_timeout:
            reasons.append("timeout_rate")
        if excl_few:
            reasons.append("few_valid_trials")
        if excl_missing:
            reasons.append("missing_blocks")

        def pct(x):
            return round(x * 100, 2) if pd.notna(x) else np.nan

        rows.append({
            "ParticipantID": pid,
            "N_Analysis_Trials": n_total,
            "N_Correct": n_correct,
            "N_Errors": n_errors,
            "N_Timeouts": n_timeout,
            "N_Anticipations": n_anticip,
            "N_Valid": n_valid,
            "Error_Rate_pct": pct(error_rate),
            "Timeout_Rate_pct": pct(timeout_rate),
            "Anticipation_Rate_pct": pct(anticip_rate),
            "Has_Block_2": has_b2,
            "Has_Block_4": has_b4,
            "Exclude_ErrorRate": excl_error,
            "Exclude_TimeoutRate": excl_timeout,
            "Exclude_FewTrials": excl_few,
            "Exclude_MissingBlocks": excl_missing,
            "EXCLUDE": exclude,
            "Exclusion_Reasons": "; ".join(reasons),
        })

    return pd.DataFrame(rows)


# Colonne tenute nel Trial_Level (le altre, morte o ridondanti, vengono tolte).
TRIAL_KEEP = [
    "ParticipantID", "Block", "Trial", "Condition", "IsCritical", "IsPractice",
    "Category", "TrialType", "Error", "RT", "RT_clean", "RT_scored",
    "Trial_Excluded", "Exclusion_Reason",
]


def clean_trial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tiene solo le colonne utili del Trial_Level, scartando quelle morte."""
    cols = [c for c in TRIAL_KEEP if c in df.columns]
    return df[cols].copy()


def score(df: pd.DataFrame, cfg: ScoringConfig):
    """Esegue tutto e restituisce (trial_data_pulito, summary, quality).

    Se cfg.drop_excluded e' True, rimuove gli esclusi da summary, quality e trial.
    Altrimenti tutti i partecipanti restano (i missing sono NA) -> merge facile.
    """
    trials = enrich_trials(df, cfg)
    quality = compute_quality(trials, cfg)
    summary = compute_summary(trials, cfg)

    flag = quality.set_index("ParticipantID")[["EXCLUDE", "Exclusion_Reasons"]]
    summary = summary.merge(flag, on="ParticipantID", how="left")

    if cfg.drop_excluded:
        keep = quality.loc[~quality["EXCLUDE"], "ParticipantID"]
        trials = trials[trials["ParticipantID"].isin(keep)]
        summary = summary[summary["ParticipantID"].isin(keep)]
        quality = quality[quality["ParticipantID"].isin(keep)]

    trials = clean_trial_columns(trials)
    return trials, summary, quality
