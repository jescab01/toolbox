
import numpy as np
import pandas as pd


# %% Aux function for cleaning dataset
def progressive_missing_dropout(data, missing_percent_drop=None, verbose=True):
    '''

    :param data:  might have iso3 and Years in index
    :param missing_percent_drop:
    :param verbose:
    :return:
    '''

    if missing_percent_drop is None:
        missing_percent_drop = [95, 85]
    df_clean = data.copy()

    for pct_drop in missing_percent_drop:

        # a1. Calculate missingness per Variable
        var_na_pct = df_clean.isna().mean() * 100
        # a2. drop variables with pct_nan > pct_drop
        var_drop = df_clean.loc[:, var_na_pct > pct_drop].columns.tolist()
        df_clean = df_clean.drop(columns=var_drop)

        # a3. calculate missingness per observation
        obs_na_pct = df_clean.isna().mean(axis=1) * 100
        # a4. drop observations with pct_nan > pct_drop
        obs_drop = df_clean[obs_na_pct > pct_drop].index.tolist()
        df_clean = df_clean.drop(index=obs_drop)


    # b. What variables were dropped?
    vars_dropped = set(data.columns.tolist()).difference(set(df_clean.columns.tolist()))

    if verbose:
        print(f"Cleaning df ::"
              f" Vars dropped ({len(vars_dropped)}): {vars_dropped} |"
              f" Obs dropped: {len(data) - len(df_clean)}."
              f"    Final df shape {df_clean.shape} with {df_clean.dropna().shape[0]} complete obs.")

    return df_clean, vars_dropped, len(data) - len(df_clean), df_clean.shape[0], df_clean.dropna().shape[0]


# %% Select number of dimentions in the two hidden layers
def compute_hidden_dims(data: pd.DataFrame,k: float = 32.0, c: float = 25.0,
    h1_min: int = 32, h1_max: int = 512, h2_min: int = 16, use_missingness: bool = True):
    """
    Compute hidden layer sizes (H1, H2) for a 2-layer MLP encoder
    based on input dimensionality and effective number of observations.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with shape (N, D), rows = observations, columns = variables.
    k : float
        Scaling factor for proposed width: H1_prop = k * sqrt(D).
    c : float
        Parameters-per-observation budget (controls max capacity).
    h1_min, h1_max : int
        Lower and upper bounds for H1.
    h2_min : int
        Minimum width for H2.
    use_missingness : bool
        If True, use effective sample size N_eff = N * obs_rate.

    Returns
    -------
    hidden_dims : tuple[int, int]
        (H1, H2)
    info : dict
        Dictionary with intermediate quantities for logging / debugging.
    """

    # --- Dimensions ---
    N, D = data.shape

    # --- Effective number of observations ---
    if use_missingness:
        obs_rate = 1.0 - data.isna().mean().mean()
        N_eff = max(N * obs_rate, 1.0)
    else:
        obs_rate = 1.0
        N_eff = float(N)

    # --- Proposed width from input dimensionality ---
    H1_prop = int(round(k * np.sqrt(D)))

    # --- Max width from parameter budget ---
    # Solve: 0.5*H1^2 + D*H1 - c*N_eff <= 0
    disc = D**2 + 2.0 * c * N_eff
    H1_max_budget = int(np.floor(-D + np.sqrt(disc)))

    # --- Final H1 ---
    H1 = int(
        np.clip(
            min(H1_prop, H1_max_budget),
            h1_min,
            h1_max,
        )
    )

    # --- H2 ---
    H2 = max(int(round(H1 / 2)), h2_min)

    info = {
        "N": N,
        "D": D,
        "obs_rate": obs_rate,
        "N_eff": N_eff,
        "H1_prop": H1_prop,
        "H1_max_budget": H1_max_budget,
        "H1": H1,
        "H2": H2,
    }

    return (H1, H2), info


# %% Aux function to select best hyper params
def select_hp_1se(scores_inn: pd.DataFrame, metric: str = "mse", z_col: str = "zdims",
                  b_col: str = "beta", prefer: str = "max_beta_min_z",):

    # 1) Agrega por (z, beta): mean + SE
    g = scores_inn.groupby([z_col, b_col])[metric]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))

    # 2) Encuentra el mejor mean (mínimo)
    best_row = summary.loc[summary["mean"].idxmin()].copy()
    threshold = float(best_row["mean"] + best_row["se"])  # 1-SE rule

    # 3) Candidatos dentro de 1-SE
    cand = summary[summary["mean"] <= threshold].copy()

    # 4) Regla de desempate (parsimonia)
    if prefer == "max_beta_min_z":
        # Primero fuerza "latente más regularizado": beta alto
        # Luego fuerza "compacto": z pequeño
        cand = cand.sort_values([b_col, z_col], ascending=[False, True])
    elif prefer == "min_z_max_beta":
        cand = cand.sort_values([z_col, b_col], ascending=[True, False])
    else:
        raise ValueError("prefer must be 'max_beta_min_z' or 'min_z_max_beta'")

    chosen = cand.iloc[0]

    return {
        "chosen_z": int(chosen[z_col]),
        "chosen_beta": float(chosen[b_col]),
        "threshold": threshold,
        "best_mean": float(best_row["mean"]),
        "best_se": float(best_row["se"]),
        "summary": summary.sort_values([z_col, b_col]),
        "candidates": cand,
    }


def select_z_1se(scores_inn: pd.DataFrame, metric: str = "mse", z_col: str = "zdims"):

    # Agrega por z (colapsando beta)
    g = scores_inn.groupby(z_col)[metric]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))

    best_row = summary.loc[summary["mean"].idxmin()].copy()
    threshold = float(best_row["mean"] + best_row["se"])

    cand = summary[summary["mean"] <= threshold].copy()
    chosen_z = int(cand[z_col].min())  # parsimonia

    return {
        "chosen_z": chosen_z,
        "threshold": threshold,
        "best_mean": float(best_row["mean"]),
        "best_se": float(best_row["se"]),
        "summary": summary.sort_values(z_col),
        "candidates": cand.sort_values(z_col),
    }


def select_z_by_elbow_envars(
    scores: pd.DataFrame, block: str | None = None, okf: int | None = None, beta: float | None = None,
    z_col: str = "zdims", en_col: str = "EN_vars", fold_col: str = "ikf", beta_col: str = "beta",
    min_z: int | None = None, max_z: int | None = None,
    method: str = "delta", eps: float = 0.05, patience: int = 2, aggregate_beta: str = "mean"):
    """
        Selección de zdims por elbow en EN_vars (complejidad latente).

    Intuición:
      - EN_vars(z) suele crecer sublinealmente con z.
      - Elegimos el menor z a partir del cual la mejora marginal en EN_vars es "pequeña"
        durante 'patience' pasos.

    Parámetros clave:
      - method="delta": usa umbral absoluto sobre ΔEN = EN(z) - EN(z-1).
        eps=0.05 suele ser un punto de partida razonable (ajustable por bloque).
      - method="rel_delta": usa umbral relativo sobre ΔEN / EN(z-1).
        eps=0.05 significa <5% de mejora relativa.

      - foldwise: primero promedia dentro de cada fold (ikf) y luego calcula stats sobre folds.
      - beta:
          * si beta se especifica, filtra por ese beta.
          * si beta=None, colapsa sobre beta con aggregate_beta sobre la media foldwise por beta.

    Devuelve:
      dict con chosen_z, summary (por z), y diagnóstico.

    :param scores:
    :param block:
    :param okf:
    :param beta:
    :param z_col:
    :param en_col:
    :param fold_col:
    :param beta_col:
    :param min_z:
    :param max_z:
    :param method:  "delta" | "rel_delta"
    :param eps:  umbral de mejora marginal (ver arriba)
    :param patience:  nº de pasos consecutivos cumpliendo el criterio
    :param aggregate_beta:  si beta=None: "mean" | "median" | "min" | "max"
    :return:
    """

    df = scores.copy()

    # filtros
    if block is not None and "block" in df.columns:
        df = df[df["block"] == block]
    if okf is not None and "okf" in df.columns:
        df = df[df["okf"] == okf]
    if beta is not None:
        df = df[df[beta_col] == beta]

    if df.empty:
        raise ValueError("No hay filas tras aplicar los filtros (block/okf/beta).")

    # acotar rango de z
    if min_z is not None:
        df = df[df[z_col] >= min_z]
    if max_z is not None:
        df = df[df[z_col] <= max_z]
    if df.empty:
        raise ValueError("No hay filas tras aplicar min_z/max_z.")

    # 1) foldwise: media por (z, beta, fold)
    #    si no existe fold_col, caerá a agregación simple por z (menos recomendable)
    if fold_col in df.columns:
        keys = [z_col]
        if beta_col in df.columns:
            keys.append(beta_col)
        keys.append(fold_col)

        df_fold = (
            df.groupby(keys, as_index=False)[en_col]
              .mean()
        )
    else:
        df_fold = df[[z_col, en_col]].copy()
        df_fold[fold_col] = 0
        if beta_col in df.columns:
            df_fold[beta_col] = df[beta_col].values

    # 2) colapsar sobre beta si beta=None
    if beta is None and beta_col in df_fold.columns:
        # primero: media sobre folds para cada (z, beta)
        zb = (
            df_fold.groupby([z_col, beta_col], as_index=False)[en_col]
                  .mean()
        )

        # segundo: agregación sobre beta para cada z
        agg_map = {
            "mean": "mean",
            "median": "median",
            "min": "min",
            "max": "max",
        }
        if aggregate_beta not in agg_map:
            raise ValueError("aggregate_beta debe ser: 'mean', 'median', 'min' o 'max'.")

        z_en = (
            zb.groupby(z_col, as_index=False)[en_col]
              .agg(agg_map[aggregate_beta])
              .rename(columns={en_col: "en_mean"})
        )

        # no tenemos SE foldwise en este modo (porque colapsamos beta antes de folds)
        # (si lo quieres con SE, habría que fijar beta o definir una unidad bootstrap sobre beta)
        z_en["en_se"] = np.nan
        z_en["n_units"] = zb.groupby(z_col)[beta_col].nunique().values

    else:
        # stats sobre folds (unidad = fold_col)
        z_en = (
            df_fold.groupby(z_col)[en_col]
                  .agg(["mean", "std", "count"])
                  .reset_index()
                  .rename(columns={"mean": "en_mean", "std": "en_std", "count": "n_units"})
        )
        z_en["en_se"] = z_en["en_std"] / np.sqrt(z_en["n_units"].clip(lower=1))

    # ordenar por z
    z_en = z_en.sort_values(z_col).reset_index(drop=True)

    # 3) calcular mejoras marginales
    z_en["delta_en"] = z_en["en_mean"].diff()
    z_en["rel_delta_en"] = z_en["delta_en"] / z_en["en_mean"].shift(1)

    if method == "delta":
        crit = z_en["delta_en"] <= eps
    elif method == "rel_delta":
        crit = z_en["rel_delta_en"] <= eps
    else:
        raise ValueError("method debe ser 'delta' o 'rel_delta'.")

    # 4) encontrar primer z donde el criterio se cumple 'patience' veces consecutivas
    #    El candidato natural es el z actual cuando la racha alcanza patience.
    run = 0
    chosen_z = int(z_en[z_col].iloc[-1])  # fallback: z máximo si no hay elbow
    elbow_found = False

    for i in range(len(z_en)):
        if i == 0:
            continue  # no hay delta para el primer punto
        if bool(crit.iloc[i]):
            run += 1
        else:
            run = 0

        if run >= patience:
            chosen_z = int(z_en[z_col].iloc[i - patience + 1])
            elbow_found = True
            break

    # 5) devolver diagnóstico útil
    return {
        "chosen_z": chosen_z,
        "elbow_found": elbow_found,
        "method": method,
        "eps": float(eps),
        "patience": int(patience),
        "beta": None if beta is None else float(beta),
        "aggregate_beta": aggregate_beta if beta is None else None,
        "summary": z_en,   # dataframe con en_mean, deltas, etc.
    }


def select_z_klperdim(scores, scores_per_dim, kl_th=0.01, mse_col = "mse", beta=1):


    # 1) KL por dimensión
    kl = scores_per_dim[(scores_per_dim["var"] == "kl") & (scores_per_dim["beta"] == beta)].copy()

    # por fold y configuración: media KL en dims y % activas
    kl_g = (kl.groupby(["block", "beta", "zdims", "okf", "ikf"]).agg(
        kl_perdim_avg=("val", "mean"),
        act_dims=("val", lambda x: np.sum(np.asarray(x) > kl_th)),)
            .reset_index())

    kl_g["act_perc"] = kl_g["act_dims"] / kl_g["zdims"]

    # 2) MSE por fold y configuración (desde scores.csv)
    mse_g = (scores.loc[(scores["beta"] == beta)].groupby(["block", "beta", "zdims", "okf", "ikf"])[mse_col]
             .mean()
             .reset_index()
             .rename(columns={mse_col: "mse"}))

    # 3) merge
    diag = kl_g.merge(mse_g, on=["block", "beta", "zdims", "okf", "ikf"], how="left")

    # diag.groupby(["block", "beta", "zdims"], as_index=False).agg(
    #     mse=("mse", "median"),
    #     klavg=("kl_perdim_avg", "median"),
    #     actperc=("act_perc", "median"),).sort_values(["block", "beta", "zdims"])

    g = diag.groupby(["block", "beta", "zdims"], as_index=False).agg(
        klavg=("kl_perdim_avg", "median"),
        klstd=("kl_perdim_avg", "std"),
        n=("kl_perdim_avg", "count"),
        act_dims=("act_dims", "median"),
        act_perc=("act_perc", "median"),
    )

    g["se"] = g["klstd"] / np.sqrt(g["n"].clip(lower=1))
    g = g.sort_values("zdims")

    # 1-SE for MAX klavg
    best = g.loc[g["klavg"].idxmax()]
    thr = best["klavg"] - best["se"]
    cand = g[g["klavg"] >= thr].copy()

    # entre candidatos, preferimos el menor z que además tenga act_perc>=p_min si existe
    cand_ok = cand[cand["act_perc"] >= 0.75]
    if not cand_ok.empty:
        z0_row = cand_ok.sort_values("zdims").iloc[0]
    else:
        z0_row = cand.sort_values("zdims").iloc[0]

    z0 = int(z0_row["zdims"])

    # ajuste por dims activas
    act = float(z0_row["act_dims"])
    z1 = int(np.round(act))

    z1 = max(1, min(z1, z0))  # no aumentamos, solo reducimos

    return {
        "z_1se_klavg": z0,
        "chosen_z": z1,
        "best_klavg": float(best["klavg"]),
        "thr_1se": float(thr),
        "table": g,
        "candidates": cand,
    }


# %% Hyperparams selection functions: unified.

import numpy as np
import pandas as pd


# -------------------------
# Helpers: agregación + 1SE
# -------------------------

def _aggregate_mean_se(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    g = df.groupby(group_cols)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def _select_1se(
    summary: pd.DataFrame,
    group_cols: list[str],
    direction: str = "min",               # "min" o "max"
    prefer: str | None = None,            # reglas de desempate sobre columnas presentes
    prefer_cols: list[str] | None = None, # si quieres controlar explicitamente el sort
) -> dict:
    """
    summary debe contener: group_cols + ["mean", "se"].
    """
    if direction not in ("min", "max"):
        raise ValueError("direction debe ser 'min' o 'max'")

    if direction == "min":
        best_row = summary.loc[summary["mean"].idxmin()].copy()
        thr = float(best_row["mean"] + best_row["se"])   # 1-SE hacia arriba
        cand = summary[summary["mean"] <= thr].copy()
    else:
        best_row = summary.loc[summary["mean"].idxmax()].copy()
        thr = float(best_row["mean"] - best_row["se"])   # 1-SE hacia abajo
        cand = summary[summary["mean"] >= thr].copy()

    # Desempate / parsimonia
    if cand.empty:
        raise RuntimeError("No hay candidatos tras aplicar 1-SE (esto no debería ocurrir).")

    if prefer_cols is not None:
        cand = cand.sort_values(
            [c for c, _ in prefer_cols],
            ascending=[asc for _, asc in prefer_cols]
        )
    elif prefer is not None:
        # preferencias típicas para (beta, z)
        if prefer == "max_beta_min_z":
            # beta alto, z bajo
            if ("beta" in cand.columns) and ("zdims" in cand.columns):
                cand = cand.sort_values(["beta", "zdims"], ascending=[False, True])
            else:
                raise ValueError("prefer='max_beta_min_z' requiere columnas beta y zdims.")
        elif prefer == "min_z_max_beta":
            if ("beta" in cand.columns) and ("zdims" in cand.columns):
                cand = cand.sort_values(["zdims", "beta"], ascending=[True, False])
            else:
                raise ValueError("prefer='min_z_max_beta' requiere columnas beta y zdims.")
        elif prefer == "min_z":
            if "zdims" in cand.columns:
                cand = cand.sort_values(["zdims"], ascending=[True])
            else:
                raise ValueError("prefer='min_z' requiere columna zdims.")
        else:
            raise ValueError("prefer no reconocido.")
    else:
        # default razonable: si hay (beta, z) -> max beta, min z; si solo z -> min z
        if ("beta" in cand.columns) and ("zdims" in cand.columns):
            cand = cand.sort_values(["beta", "zdims"], ascending=[False, True])
        elif "zdims" in cand.columns:
            cand = cand.sort_values(["zdims"], ascending=[True])

    chosen = cand.iloc[0].copy()
    out = {
        "threshold": float(thr),
        "best_mean": float(best_row["mean"]),
        "best_se": float(best_row["se"]),
        "best_row": best_row,
        "candidates": cand,
        "chosen_row": chosen,
    }
    # añade claves comunes si existen
    for c in group_cols:
        if c in chosen.index:
            out[f"chosen_{c}"] = chosen[c]
    return out


# -------------------------
# Helper: diagnóstico KL active dims (opcional)
# -------------------------

def _kl_active_diag(
    scores_per_dim: pd.DataFrame,
    beta: float | None = None,
    kl_th: float = 0.01,
    kl_var: str = "kl",
    cols: dict | None = None,
) -> pd.DataFrame:
    """
    Devuelve, por (block, beta, zdims, okf, ikf), el KL medio por dim y nº/% dims activas.
    """
    cols = cols or {}
    c_block = cols.get("block", "block")
    c_beta  = cols.get("beta", "beta")
    c_z     = cols.get("z", "zdims")
    c_okf   = cols.get("okf", "okf")
    c_ikf   = cols.get("ikf", "ikf")
    c_var   = cols.get("var", "var")
    c_val   = cols.get("val", "val")

    kl = scores_per_dim[scores_per_dim[c_var] == kl_var].copy()
    if beta is not None:
        kl = kl[kl[c_beta] == beta]

    if kl.empty:
        raise ValueError("No hay filas de KL en scores_per_dim tras filtrar.")

    diag = (
        kl.groupby([c_block, c_beta, c_z, c_okf, c_ikf])
          .agg(
              kl_perdim_avg=(c_val, "mean"),
              act_dims=(c_val, lambda x: int(np.sum(np.asarray(x) > kl_th))),
          )
          .reset_index()
    )
    diag["act_perc"] = diag["act_dims"] / diag[c_z]
    return diag


def _apply_active_constraint_to_summary(
    summary: pd.DataFrame,
    active_summary: pd.DataFrame,
    on: list[str],
    act_perc_min: float | None = None,
    act_dims_min: int | None = None,
) -> pd.DataFrame:
    """
    Une métricas (mean/se) con métricas de actividad (act_perc/act_dims) al mismo nivel de agregación.
    """
    merged = summary.merge(active_summary, on=on, how="left")

    if act_perc_min is not None:
        merged = merged[merged["act_perc"] >= act_perc_min]
    if act_dims_min is not None:
        merged = merged[merged["act_dims"] >= act_dims_min]

    return merged


# -------------------------
# Helper: elbow (EN_vars u otra métrica de complejidad)
# -------------------------

def _select_elbow(
    df: pd.DataFrame,
    z_col: str,
    y_col: str,
    method: str = "delta",   # "delta" | "rel_delta"
    eps: float = 0.05,
    patience: int = 2,
) -> dict:
    """
    df debe estar agregado por z con una columna y_col (p.ej. en_mean).
    """
    if df.empty:
        raise ValueError("df vacío en _select_elbow.")

    d = df.sort_values(z_col).reset_index(drop=True).copy()
    d["delta"] = d[y_col].diff()
    d["rel_delta"] = d["delta"] / d[y_col].shift(1)

    if method == "delta":
        crit = d["delta"] <= eps
    elif method == "rel_delta":
        crit = d["rel_delta"] <= eps
    else:
        raise ValueError("method debe ser 'delta' o 'rel_delta'.")

    run = 0
    chosen_z = int(d[z_col].iloc[-1])
    elbow_found = False

    for i in range(len(d)):
        if i == 0:
            continue
        if bool(crit.iloc[i]):
            run += 1
        else:
            run = 0
        if run >= patience:
            chosen_z = int(d[z_col].iloc[i - patience + 1])
            elbow_found = True
            break

    return {
        "chosen_z": chosen_z,
        "elbow_found": elbow_found,
        "method": method,
        "eps": float(eps),
        "patience": int(patience),
        "summary": d,
    }


# =========================
# FUNCIÓN UNIFICADA
# =========================

def select_hp(
    scores: pd.DataFrame,
    rule: str = "1se",                    # "1se" | "elbow"
    metric: str = "mse",                  # para rule="1se"
    direction: str = "min",               # "min" (MSE) o "max" (KLavg, etc.)
    z_col: str = "zdims",
    b_col: str = "beta",
    group: str = "zb",                    # "zb" (elige z,beta) | "z" (colapsa beta)
    prefer: str | None = "max_beta_min_z",
    # filtros típicos (opcionales)
    block: str | None = None, okf: int | None = None, beta: float | None = None,
    # rango z
    min_z: int | None = None,
    max_z: int | None = None,
    # --- KL active dims (opcional; requiere scores_per_dim)
    scores_per_dim: pd.DataFrame | None = None,
    kl_th: float = 0.01,
    act_perc_min: float | None = None,   # hard constraint, p.ej. 0.75
    act_dims_min: int | None = None,     # hard constraint alternativa
    reduce_z_to_active: bool = False,    # soft postprocess: chosen_z := round(act_dims)
    # --- elbow options
    en_col: str = "EN_vars",
    fold_col: str = "ikf",
    method: str = "delta",
    eps: float = 0.05,
    patience: int = 2,
    aggregate_beta: str = "mean",        # si beta=None en elbow
) -> dict:
    """
    Selector unificado.

    Casos típicos:
      - 1SE sobre MSE para (z,beta): rule="1se", metric="mse", direction="min", group="zb"
      - 1SE sobre KLavg (maximizar): rule="1se", metric="kl_perdim_avg", direction="max"
        (si quieres construir metric desde scores_per_dim, ver nota abajo)
      - Elbow sobre EN_vars: rule="elbow" (elige z), opcionalmente con beta fijo o agregando sobre beta.

    Nota importante:
      - El módulo KL-active NO crea por sí mismo una métrica objetivo; solo añade columnas (act_dims/act_perc)
        para filtrar o para reducir z.
      - Si quieres que el objetivo sea “KLavg”, lo más limpio es que `scores` ya contenga una columna agregada
        tipo 'kl_perdim_avg' por fold/config, o que la construyas antes y se la pases como `metric`.
    """

    df = scores.copy()

    # -----------------
    # filtros generales
    # -----------------
    if block is not None and "block" in df.columns:
        df = df[df["block"] == block]
    if okf is not None and "okf" in df.columns:
        df = df[df["okf"] == okf]
    if beta is not None and b_col in df.columns:
        df = df[df[b_col] == beta]

    if min_z is not None:
        df = df[df[z_col] >= min_z]
    if max_z is not None:
        df = df[df[z_col] <= max_z]

    if df.empty:
        raise ValueError("No hay filas tras aplicar filtros (block/okf/beta/min_z/max_z).")

    # -----------------
    # modo ELBOW (elige z)
    # -----------------
    if rule == "elbow":
        # foldwise mean por (z, beta, fold) y luego colapsos como en tu función
        if fold_col in df.columns:
            keys = [z_col]
            if b_col in df.columns:
                keys.append(b_col)
            keys.append(fold_col)
            df_fold = df.groupby(keys, as_index=False)[en_col].mean()
        else:
            df_fold = df[[z_col, en_col]].copy()
            df_fold[fold_col] = 0
            if b_col in df.columns:
                df_fold[b_col] = df[b_col].values

        # beta fijo o agregado sobre beta
        if beta is None and b_col in df_fold.columns:
            zb = df_fold.groupby([z_col, b_col], as_index=False)[en_col].mean()
            agg_map = {"mean": "mean", "median": "median", "min": "min", "max": "max"}
            if aggregate_beta not in agg_map:
                raise ValueError("aggregate_beta debe ser: 'mean', 'median', 'min' o 'max'.")
            z_en = (
                zb.groupby(z_col, as_index=False)[en_col]
                  .agg(agg_map[aggregate_beta])
                  .rename(columns={en_col: "en_mean"})
            )
        else:
            z_en = (
                df_fold.groupby(z_col)[en_col]
                       .agg(["mean", "std", "count"])
                       .reset_index()
                       .rename(columns={"mean": "en_mean", "std": "en_std", "count": "n_units"})
            )

        elbow_res = _select_elbow(z_en, z_col=z_col, y_col="en_mean", method=method, eps=eps, patience=patience)
        elbow_res["chosen_beta"] = None if beta is None else float(beta)
        elbow_res["aggregate_beta"] = aggregate_beta if beta is None else None
        return elbow_res

    # -----------------
    # modo 1SE
    # -----------------
    if rule != "1se":
        raise ValueError("rule debe ser '1se' o 'elbow'.")

    if metric not in df.columns:
        raise ValueError(f"metric='{metric}' no está en scores. Columnas disponibles: {list(df.columns)}")

    if group == "zb":
        group_cols = [z_col, b_col]
    elif group == "z":
        group_cols = [z_col]
    else:
        raise ValueError("group debe ser 'zb' o 'z'.")

    summary = _aggregate_mean_se(df, group_cols=group_cols, metric=metric)

    # -----------------
    # (opcional) añadir info de dims activas a nivel (z,beta)
    # -----------------
    if (scores_per_dim is not None) or reduce_z_to_active or (act_perc_min is not None) or (act_dims_min is not None):
        if scores_per_dim is None:
            raise ValueError("Para restricciones/reducción por actividad necesitas scores_per_dim.")

        # Ojo: actividad se define por (block,beta,zdims,okf,ikf) en tu diseño.
        # Aquí agregamos a nivel (z,beta) o (z) consistente con summary.
        diag = _kl_active_diag(scores_per_dim, beta=beta, kl_th=kl_th)

        # si filtraste block/okf arriba, aplica también aquí
        if block is not None and "block" in diag.columns:
            diag = diag[diag["block"] == block]
        if okf is not None and "okf" in diag.columns:
            diag = diag[diag["okf"] == okf]
        if min_z is not None:
            diag = diag[diag[z_col] >= min_z]
        if max_z is not None:
            diag = diag[diag[z_col] <= max_z]

        if diag.empty:
            raise ValueError("diag KL-active quedó vacío tras filtros.")

        # agregación de actividad al mismo nivel que summary
        if group == "zb":
            act_sum = (
                diag.groupby([z_col, "beta"], as_index=False)
                    .agg(
                        act_dims=("act_dims", "median"),
                        act_perc=("act_perc", "median"),
                        kl_perdim_avg=("kl_perdim_avg", "median"),
                    )
            )
            # unify b_col name
            act_sum = act_sum.rename(columns={"beta": b_col})
            on = [z_col, b_col]
        else:
            act_sum = (
                diag.groupby([z_col], as_index=False)
                    .agg(
                        act_dims=("act_dims", "median"),
                        act_perc=("act_perc", "median"),
                        kl_perdim_avg=("kl_perdim_avg", "median"),
                    )
            )
            on = [z_col]

        # merge y aplica constraints
        summary2 = _apply_active_constraint_to_summary(
            summary, act_sum, on=on, act_perc_min=act_perc_min, act_dims_min=act_dims_min
        )
        if summary2.empty:
            # fallback: sin constraint (mejor devolver algo que petar en producción)
            summary2 = summary.merge(act_sum, on=on, how="left")
            constrained = False
        else:
            constrained = True
        summary = summary2
    else:
        constrained = False

    # -----------------
    # selección 1SE final
    # -----------------
    res = _select_1se(
        summary=summary,
        group_cols=group_cols,
        direction=direction,
        prefer=prefer if group == "zb" else "min_z",  # si solo z, parsimonia por z
    )

    chosen_z = int(res["chosen_row"][z_col])
    chosen_beta = float(res["chosen_row"][b_col]) if (group == "zb" and b_col in res["chosen_row"]) else None

    # -----------------
    # (opcional) reducción z -> round(act_dims)
    # -----------------
    if reduce_z_to_active:
        if "act_dims" not in res["chosen_row"].index or pd.isna(res["chosen_row"]["act_dims"]):
            # no disponible -> no reducimos
            z_reduced = chosen_z
        else:
            z_reduced = int(np.round(float(res["chosen_row"]["act_dims"])))
            z_reduced = max(1, min(z_reduced, chosen_z))  # solo reduce
    else:
        z_reduced = chosen_z

    out = {
        "rule": "1se",
        "metric": metric,
        "direction": direction,
        "group": group,
        "constrained_by_active": constrained,
        "chosen_z": z_reduced,
        "chosen_beta": chosen_beta,
        "chosen_z_raw": chosen_z,
        "threshold": res["threshold"],
        "best_mean": res["best_mean"],
        "best_se": res["best_se"],
        "summary": summary.sort_values(group_cols),
        "candidates": res["candidates"],
        "chosen_row": res["chosen_row"],
    }
    return out
