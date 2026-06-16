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

Convenzioni codifica MinnoJS:  Error 1 = corretto, 0 = errore, 2 = timeout.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ScoringConfig:
    """Tutte le scelte metodologiche, ognuna esposta come toggle nell'app."""

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


CRITICAL = "IsCritical"


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Block", "Trial", "RT", "Error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def _block_means(crit: pd.DataFrame) -> dict:
    """Media delle latenze CORRETTE (RT_clean) per partecipante x condizione."""
    correct = crit[crit["Error"] == 1]
    means = (
        correct.groupby(["ParticipantID", "Condition"])["RT_clean"].mean().to_dict()
    )
    return means


def _add_rt_scored(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """RT_scored = valore usato nello scoring (corretti -> RT_clean; errori ->
    media blocco + penalita' se in modalita' penalty; altrimenti NA)."""
    crit = df[df[CRITICAL] == True]
    means = _block_means(crit)

    rt_scored = df["RT_clean"].copy()

    if cfg.error_mode == "penalty":
        err_mask = (df[CRITICAL] == True) & (df["Error"] == 0) & (~df["Trial_Excluded"])
        bm = df.loc[err_mask].apply(
            lambda r: means.get((r["ParticipantID"], r["Condition"]), np.nan), axis=1
        )
        rt_scored.loc[err_mask] = bm.values + cfg.penalty_ms
    else:  # exclude: gli errori non entrano nelle medie
        err_mask = (df[CRITICAL] == True) & (df["Error"] == 0)
        rt_scored.loc[err_mask] = np.nan

    df = df.copy()
    df["RT_scored"] = rt_scored
    return df


def enrich_trials(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Pipeline trial-level completa: filtri + RT_scored."""
    df = apply_trial_filters(df, cfg)
    df = _add_rt_scored(df, cfg)
    return df


def _sd_denominator(crit: pd.DataFrame, cfg: ScoringConfig,
                    m_cong: float, m_incong: float) -> float:
    if cfg.sd_method == "all_correct":
        vals = crit.loc[crit["Error"] == 1, "RT_clean"].dropna()
        return vals.std(ddof=1) if len(vals) > 1 else np.nan

    if cfg.sd_method == "all_with_penalty":
        vals = crit["RT_scored"].dropna()
        return vals.std(ddof=1) if len(vals) > 1 else np.nan

    # pooled: SD per condizione sui valori usati nelle medie (RT_scored), poi pool
    sd_c = crit.loc[crit["Condition"] == "congruent", "RT_scored"].dropna()
    sd_i = crit.loc[crit["Condition"] == "incongruent", "RT_scored"].dropna()
    v_c = sd_c.var(ddof=1) if len(sd_c) > 1 else np.nan
    v_i = sd_i.var(ddof=1) if len(sd_i) > 1 else np.nan
    if np.isnan(v_c) or np.isnan(v_i):
        return np.nan
    return np.sqrt((v_c + v_i) / 2)


def compute_summary(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Summary per partecipante con D-score secondo la configurazione."""
    rows = []
    for pid, p in df.groupby("ParticipantID"):
        crit = p[p[CRITICAL] == True]
        cong = crit[crit["Condition"] == "congruent"]
        incong = crit[crit["Condition"] == "incongruent"]

        m_cong = cong["RT_scored"].dropna().mean() if len(cong) else np.nan
        m_incong = incong["RT_scored"].dropna().mean() if len(incong) else np.nan
        sd = _sd_denominator(crit, cfg, m_cong, m_incong)

        if cfg.d_sign == "incong_minus_cong":
            d = (m_incong - m_cong) / sd if sd and not np.isnan(sd) else np.nan
        else:
            d = (m_cong - m_incong) / sd if sd and not np.isnan(sd) else np.nan

        first_cond = (
            crit.sort_values(["Block", "Trial"])["Condition"].iloc[0]
            if len(crit) else "unknown"
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
    """Quality report con flag di esclusione secondo la configurazione."""
    rows = []
    for pid, p in df.groupby("ParticipantID"):
        crit = p[p[CRITICAL] == True]
        non_timeout = crit[crit["Error"] != 2]

        n_correct = int((crit["Error"] == 1).sum())
        n_errors = int((crit["Error"] == 0).sum())
        n_timeout = int((crit["Error"] == 2).sum())
        n_total = len(crit)
        n_resp = n_correct + n_errors

        error_rate = n_errors / n_resp if n_resp > 0 else np.nan
        timeout_rate = n_timeout / n_total if n_total > 0 else np.nan
        n_valid = int(crit["RT_scored"].notna().sum())

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

        rows.append({
            "ParticipantID": pid,
            "N_Critical": n_total,
            "N_Correct": n_correct,
            "N_Errors": n_errors,
            "N_Timeouts": n_timeout,
            "N_Valid": n_valid,
            "Error_Rate": round(error_rate, 4) if pd.notna(error_rate) else np.nan,
            "Timeout_Rate": round(timeout_rate, 4) if pd.notna(timeout_rate) else np.nan,
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


def score(df: pd.DataFrame, cfg: ScoringConfig):
    """Esegue tutto e restituisce (trial_data_arricchito, summary, quality).

    Se cfg.drop_excluded e' True, rimuove gli esclusi da summary, quality e trial.
    Altrimenti tutti i partecipanti restano (i missing sono NA) -> merge facile.
    """
    trials = enrich_trials(df, cfg)
    quality = compute_quality(trials, cfg)
    summary = compute_summary(trials, cfg)

    # propaga il flag di esclusione nel summary, comodo per il merge
    flag = quality.set_index("ParticipantID")[["EXCLUDE", "Exclusion_Reasons"]]
    summary = summary.merge(flag, on="ParticipantID", how="left")

    if cfg.drop_excluded:
        keep = quality.loc[~quality["EXCLUDE"], "ParticipantID"]
        trials = trials[trials["ParticipantID"].isin(keep)]
        summary = summary[summary["ParticipantID"].isin(keep)]
        quality = quality[quality["ParticipantID"].isin(keep)]

    return trials, summary, quality
