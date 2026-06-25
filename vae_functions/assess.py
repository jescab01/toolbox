import os
import time
import torch
import torch.nn.functional as F
from collections import defaultdict

import numpy as np
import pandas as pd

import umap.plot
import matplotlib.pyplot as plt

import sklearn

import shap
import statsmodels.api as sm
from pygam import LinearGAM, l, s

from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from models import VAE_encoder_wrapper, VAE_encoder_wrapper_masked


# %% Assessment functions

def assess_mVAE_performance(block, X, M, fnames, vae_model, zdims, beta, ikf, okf, vae_dir, v_tag,
                            avoid=None, kl_th=0.01, tw_n=15, seed=42, verbose=False):

    tic0 = time.time()

    if avoid is None:
        avoid = []

    res = {"model":"mVAE", "block": block, "zdims": zdims, "beta": beta,  "kl_th":kl_th, "ikf":ikf, "okf":okf}
    res_per_dim = []

    # %% 4.4) Deterministc evaluation of VAE on outer test
    X_test_sc_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)

    vae_model.eval()
    with torch.no_grad():
        mu, logvar = vae_model.encode(X_test_sc_t)  # (n_test, zdim)
        Xhat = vae_model.decode(mu)  # (n_test, xdim)

    mu_perdim, logvar_perdim = mu.mean(dim=0).detach().cpu().numpy(), logvar.mean(dim=0).detach().cpu().numpy()

    # Reconstruction errors
    se = (Xhat - X_test_sc_t).pow(2)
    ae = (Xhat - X_test_sc_t).abs()

    n_obs = M_t.sum()
    mse_obs = (se * M_t).sum() / n_obs
    mae_obs = (ae * M_t).sum() / n_obs

    # Per-feature (xdim) masked errors
    obs_per_xdim = M_t.sum(dim=0)    # (xdim,)
    se_per_xdim = (se * M_t).sum(dim=0) / obs_per_xdim
    ae_per_xdim = (ae * M_t).sum(dim=0) / obs_per_xdim

    # Per-sample masked errors
    obs_per_sample = M_t.sum(dim=1)  # (n_test,)
    se_per_sample = (se * M_t).sum(dim=1) / obs_per_sample
    ae_per_sample = (ae * M_t).sum(dim=1) / obs_per_sample

    # --- R² (masked) ---
    mean_ref_t = X_test_sc_t.mean(dim=0, keepdim=True)  # fallback si no guardas train_mean
    sst_obs = ((X_test_sc_t - mean_ref_t).pow(2) * M_t).sum() / n_obs
    r2_obs = 1.0 - (mse_obs / sst_obs)

    # --- KL diagnostics ---
    kl_per_dim_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (n_test, zdim)
    kl_per_sample = kl_per_dim_sample.sum(dim=1)  # (n_test,)
    kl = kl_per_sample.mean().item()
    kl_per_dim = kl_per_dim_sample.mean(dim=0).cpu().numpy()

    active_dims = int((kl_per_dim > kl_th).sum())

    # Scalars
    mse = mse_obs.item()
    mae = mae_obs.item()
    sst = sst_obs.item()
    r2 = r2_obs.item()

    # --- Loss / ELBO proxy (note: recon term should match training; if training uses masked MSE, do same) ---
    # loss_per_sample = se_per_sample + beta * kl_per_sample
    loss = mse + beta * kl

    res.update({"loss": loss, "mse": mse, "mae": mae, "kl": kl, "sst":sst, "r2":r2,
                "kl_perdim_avg":kl_per_dim.mean(), "kl_act_n": active_dims, "kl_act_perc":active_dims/zdims})

    if "per_dim" not in avoid:
        res_per_dim.extend([{"model":"mVAE", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z,
                             f"var": "kl", "val":kl_per_dim[z], "mu":mu_perdim[z], "logvar":logvar_perdim[z]} for z in range(zdims)])
        res_per_dim.extend([{"model":"mVAE", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "x": x, "x_fname":fnames[x],
                             f"var": "se_vae", "val":se_per_xdim[x].item()} for x in range(X.shape[1])])

    if verbose:
        print(f"\tDeterministic evaluation ({round((time.time() - tic0), 2)}sec) >> ", end="")



    if "latent_dist" not in avoid:

        tic_ass = time.time()

        sigma_perdim = np.exp(0.5 * logvar_perdim)

        # --- layout de subplots ---
        ncols = int(min(4, zdims))  # cámbialo si quieres
        nrows = int(np.ceil(zdims / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols, shared_yaxes=True,
            subplot_titles=[f"z{d} | KL={kl_per_dim[d]:.3f}" for d in range(zdims)],
            horizontal_spacing=0.06, vertical_spacing=0.10,
        )

        # --- eje x (mismo para todos; basado en rangos de mu±4σ, y también [-4,4]) ---
        left = np.min(mu_perdim - 4 * sigma_perdim)
        right = np.max(mu_perdim + 4 * sigma_perdim)
        x_min = min(-4, left)
        x_max = max(4, right)
        x = np.linspace(x_min, x_max, 400)

        # --- trazas ---
        for d in range(zdims):

            r = d // ncols + 1
            c = d % ncols + 1

            # q(z_d) ~ N(mu_d, sigma_d^2)
            y_q = norm.pdf(x, loc=mu_perdim[d], scale=max(sigma_perdim[d], 1e-8))
            # referencia N(0,1)
            y_ref = norm.pdf(x, loc=0.0, scale=1.0)

            # referencia (gris claro)
            fig.add_trace(go.Scatter(x=x, y=y_ref, mode="lines", name="N(0,1)",
                    line=dict(color="lightgray", width=1), showlegend=(d == 0),), row=r, col=c)

            # distribución del latente
            fig.add_trace(go.Scatter(x=x, y=y_q, mode="lines", name="q(z)",
                    line=dict(width=2, color="dimgrey"), showlegend=(d == 0),),row=r, col=c)

        # --- estética ---
        fig.update_layout(template="plotly_white", height=260 * nrows, width=340 * ncols,
            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0),)

        # opcional: quitar labels repetidos si molesta
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        pio.write_image(fig, os.path.join(vae_dir, v_tag, f"latent_dists.png"))

        if verbose:
            print(f"\tPlot Latent distributions ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    Z_mu = mu.detach().cpu().numpy() # prepare other assessments

    if "geometry" not in avoid:
        tic_ass = time.time()

        # Intrinsic dimension (TwoNN) en el latente
        id_twonn = intrinsic_dim_twonn(Z_mu, seed=seed)

        # Trustworthiness: X->Z (preserva vecinos del espacio original en el latente)
        tw = sklearn.manifold.trustworthiness(X, Z_mu, n_neighbors=tw_n)

        res.update({"intrinsic_dim_twonn": id_twonn, "tw_n":tw_n, "trustworthyness": tw})

        if verbose:
            print(f"\tGeometric evaluation ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    # %% 4.6) Latent correlations
    if "corrs" not in avoid:

        tic_ass = time.time()

        # 1,Latent Dimensions correlation
        r_z = np.ones((1, 1), dtype=float) if zdims == 1 else np.corrcoef(Z_mu, rowvar=False)
        # plt.imshow(r_z, aspect="auto"); plt.colorbar(); plt.title("VAE - r(latent dims)")
        # plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__rLatent.png")); plt.close()
        r_z_triu = r_z[np.triu_indices(zdims, k=1)] if zdims > 1 else 0
        res.update({"r_z": np.average(np.abs(r_z_triu))})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"mVAE", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z1, "z2": z2,
                                 f"var": "r_z", "val":r_z[z1, z2]} for z1 in range(zdims) for z2 in range(zdims) if z1>z2])

        # 2, Missingness - Latent correlations
        obs_row = M.mean(axis=1)
        r_zM_per_dim = [np.corrcoef(Z_mu[:, z], obs_row, rowvar=False)[0, 1] if np.std(obs_row)!=0 else np.nan for z in range(zdims) ]
        res.update({"r_zM":np.average(r_zM_per_dim)})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"mVAE", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z,
                                 f"var": "r_zM", "val":r_zM_per_dim[z]} for z in range(zdims)])

        # 3, Variables Loadings on Latent (thorugh correlations)
        r_zX = np.corrcoef(Z_mu, X, rowvar=False)[zdims:, :zdims] # (X in rows, z in cols)
        # np concatenates the arrays by shared dim, and corrs all columns.
        res_per_dim.extend([{"model":"mVAE", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z, "x": x, "x_fname":fnames[x],
                             f"var": "r_zX", "val":r_zX[x, z]} for z in range(zdims) for x in range(X.shape[1])])

        ## Calculate the Effective number of variables (derived from Herfindahl index)
        eps = 1e-12  # numerical stability
        p_per_dim = [r_zX[:, z] / r_zX[:, z].sum() + eps  for z in range(zdims)] # contribution proportion
        H_per_dim = [np.sum(p ** 2) for p in p_per_dim] # higher contributions p, increases the concentration (H index)
        EN_per_dim = [1.0 / max(h, eps) for h in H_per_dim]
        # EN ≈ 1–3 → excelente; EN ≈ 3–5 → aceptable; EN > 6–7 → factor poco interpretable
        res.update({"EN_vars": np.average(EN_per_dim)})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"mVAE","block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z,
                                 f"var": "EN_vars", "val":EN_per_dim[z]} for z in range(zdims)])

        if verbose:
            print(f"\tLatent correlations ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    # %% 4.7) SHAP per latent dimension
    if "shap" not in avoid:
        tic_ass = time.time()

        f = VAE_encoder_wrapper(vae_model)  # forward wrapper (from X to mu, only)

        # --- subsample background ---
        n_bg = min(100, X.shape[0])
        idx_bg = np.random.choice(X.shape[0], size=n_bg, replace=False)
        X_bg = torch.tensor(X[idx_bg], dtype=torch.float32)

        # --- subsample explained set ---
        n_exp = min(500, X_test_sc_t.shape[0])
        idx_exp = np.random.choice(X_test_sc_t.shape[0], size=n_exp, replace=False)
        X_exp = X_test_sc_t[idx_exp]  # ya es tensor en tu código

        explainer = shap.GradientExplainer(f, X_bg)
        shap_values = explainer(X_exp)

        # z = 0  # Iterate over dimensions
        for z in range(zdims):
            shap_exp = shap.Explanation(values=shap_values.values[:, :, z], data=X_exp, feature_names=fnames)

            shap.plots.beeswarm(shap_exp, max_display=shap_exp.shape[1], show=False)
            plt.gcf().set_size_inches(13, 18)  # más alto si tienes muchas features
            plt.gcf().subplots_adjust(left=0.45)  # más margen para labels largos
            plt.tight_layout();
            plt.title(f"Latent dim: {z}")
            plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__SHAP_beeswarm__z{z}.png"))
            plt.close()

            shap.plots.bar(shap_exp, max_display=shap_exp.shape[1], show=False)
            plt.gcf().set_size_inches(15, 18)  # más alto si tienes muchas features
            plt.gcf().subplots_adjust(left=0.45, top=0.95, bottom=0.05)  # más margen para labels largos
            plt.title(f"Latent dim: {z}");
            plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__SHAP_bars__z{z}.png"))
            plt.close()

            if "per_dim" not in avoid:
                res_per_dim.extend([{"model":"mVAE", "block": block, "zdims": zdims, "beta": beta, "ikf": ikf, "okf": okf, "z": z, "x": x,
                                     "x_fname": fnames[x],
                                     f"var": "shap", "val": np.average(shap_values.values[:, x, z])} for x in
                                    range(shap_exp.shape[1])])

        if verbose:
            print(f"\tSHAP ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    # %% 4.3) PLOT training RESULTS
    if "train_viz" not in avoid:
        tic_ass = time.time()

        df_metrics_w = pd.read_csv(os.path.join(vae_dir, v_tag, "metrics.csv"))
        df_metrics_w = df_metrics_w.groupby(["epoch", "step"], as_index=False).first()

        df_metrics_l = df_metrics_w.melt(  # convert wide to long for plotly.express
            id_vars=["epoch", "step"], var_name="metric", value_name="value",
            value_vars=["train_kl", "train_loss", "train_recon", "val_kl", "val_loss", "val_recon"], )
        df_metrics_l = df_metrics_l.dropna()

        df_metrics_l[["split", "metric_name"]] = (df_metrics_l["metric"].str.split("_", expand=True))
        fig = px.line(df_metrics_l, x="epoch", y="value", color="split", facet_row="metric_name", title=v_tag)
        df_fbeta = df_metrics_w.loc[:, ["epoch", "val_loss_fbeta"]].dropna()
        fig.add_trace(go.Scatter(x=df_fbeta["epoch"], y=df_fbeta["val_loss_fbeta"], mode="lines",
                                 line=dict(color="indianred", dash="dash"), name="val_loss (fixed beta)"))
        fig.update_yaxes(matches=None)  # permite escalas distintas por fila (clave para KL)
        fig.update_layout(template="plotly_white", width=700, height=600, legend=dict(orientation="h", y=1.1, x=0.2),
                          yaxis1=dict(title="Loss"), yaxis2=dict(title="Reconstruction error"),
                          yaxis3=dict(title="Kullback-Leibler<br>divergence"),
                          )
        pio.write_image(fig, os.path.join(vae_dir, v_tag, f"VAE__TrainingMetrics.png"), width=700, height=600, scale=3)
        del fig

        if verbose:
            print(f"\tPlot training results ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    # %% 4.5) Visualization of the latent; reduced with UMAP if zdim>=3.
    if "latent_viz" not in avoid:

        tic_ass = time.time()

        df_test = pd.DataFrame(X, columns=fnames)

        df_test["miss_pct"] = df_test.isna().mean(axis=1).values
        df_test["mse"] = se_per_sample.detach().numpy()
        df_test["kl"] = kl_per_sample.detach().numpy()

        vars_order = ["mse", "kl"] + fnames
        nrows = int(np.ceil((df_test.shape[1] - 1) / 2))
        y_tit = "z2" if zdims > 1 else "Jitter (viz 1d)"
        size, max_op = 4, 1

        if zdims > 2:
            ncols, width = 4, 1050
            sp_titles = [v for i in range(0, len(vars_order), 2) for v in vars_order[i:i + 2] * 2]
            # UMAP reduction
            umap_model = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2, metric="euclidean", )  # random_state=42,
            Z_umap = umap_model.fit_transform(Z_mu)
            # PCA reduction
            pca = sklearn.decomposition.PCA(n_components=zdims, random_state=seed)
            Z_pca = pca.fit_transform(Z_mu)
        else:
            ncols, width = 2, 700
            sp_titles = vars_order

        fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, subplot_titles=sp_titles,
                            x_title="z1", y_title=y_tit)
        for ii, var in enumerate(sp_titles):

            row, col = (ii // ncols) + 1, (ii % ncols) + 1
            hover = [(f"{var} = {row[var]:.2f}<br>"
                      f"missing = {row['miss_pct']:.2f}<br>"
                      f"Country x Year: {i}") for i, row in df_test.iterrows()]

            if zdims == 1:
                rng = np.random.default_rng(0)
                jitter = rng.normal(0, 0.03, size=len(Z_mu))  # σ controla el ancho visual
                fig.add_trace(go.Scatter(x=Z_mu[:, 0], y=jitter, mode="markers", showlegend=False, hoverinfo="x,y,text",
                                         hovertext=hover,
                                         marker=dict(size=size, color=df_test[var],
                                                     opacity=max_op - df_test["miss_pct"].values, colorscale="Viridis")),
                              row=row, col=col)
            elif zdims == 2:
                fig.add_trace(go.Scatter(x=Z_mu[:, 0], y=Z_mu[:, 1], mode="markers", showlegend=False, hoverinfo="x,y,text",
                                         hovertext=hover,
                                         marker=dict(size=size, color=df_test[var],
                                                     opacity=max_op - df_test["miss_pct"].values, colorscale="Viridis")),
                              row=row, col=col)
            else:
                Z_ = Z_pca if col <= 2 else Z_umap
                fig.add_trace(go.Scatter(x=Z_[:, 0], y=Z_[:, 1], mode="markers", showlegend=False, hoverinfo="x,y,text",
                                         hovertext=hover,
                                         marker=dict(size=size, color=df_test[var],
                                                     opacity=max_op - df_test["miss_pct"].values, colorscale="Viridis")),
                              row=row, col=col)
                fig.add_shape(type="line", xref="paper", yref="paper", x0=0.5, x1=0.5, y0=0.0, y1=1.0,
                              line=dict(color="black", width=1, dash="dot"), )
                fig.add_annotation(xref="paper", yref="paper", x=0.25, y=1.025, text="PCA", showarrow=False,
                                   font=dict(size=14, color="black"), )
                fig.add_annotation(xref="paper", yref="paper", x=0.75, y=1.025, text="UMAP", showarrow=False,
                                   font=dict(size=14, color="black"), )
        fig.update_layout(template="plotly_white", title=v_tag, width=width, height=nrows * 150)
        # fig.show("browser")
        pio.write_image(fig, os.path.join(vae_dir, v_tag, f"VAE__latent-viz.png"), width=width, height=nrows * 150,
                        scale=3)
        pio.write_html(fig, os.path.join(vae_dir, v_tag, f"VAE__latent-viz.html"), auto_open=False)
        del fig

        if verbose:
            print(f"\tLatent visualization ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    if verbose:
        print(f"  Total time ({round((time.time() - tic0), 2)}sec) #", end="\n")

    return  Z_mu, pd.DataFrame([res]), pd.DataFrame(res_per_dim)


def assess_latent_stability(block, scores_per_dim,  metric="corr", topk=3, pairs_mode="all",
                            sim_thr=0.7, jacc_thr=0.5,  aggfunc="mean"):

    """
    Computes latent stability based on |corr(X, z)| loadings across reps (okf, ikf).
    # sim_thr - threshold for matched similarity to count as "stable"
    # jacc_thr - threshold for topk jaccard to count as "stable"
    # aggfunc - safer than "first" if duplicates exist
    # fillna - treat NaN loadings as 0 correlation for stability scoring

    Returns:
      pair_df_all: rows for all (zdims,beta) and all rep-pairs
      summaries_df: one row per (zdims,beta)

    """

    stab = []
    stab_perdim = []

    for zdims, beta in scores_per_dim[["zdims", "beta"]].drop_duplicates().itertuples(index=False):

        df_L = scores_per_dim.loc[(scores_per_dim["block"] == block) & (scores_per_dim["var"] == "r_zX")
            & (scores_per_dim["zdims"] == zdims) & (scores_per_dim["beta"] == beta), ["z", "x", "okf", "ikf", "val"]].copy()

        # Use abs(L) for stability
        df_L["val"] = df_L["val"].abs()

        # Stable ordering
        z_order, x_order, okf_order, ikf_order = (
            sorted(df_L["z"].unique()), sorted(df_L["x"].unique()), sorted(df_L["okf"].unique()), sorted(df_L["ikf"].unique()))

        # Pivot into (z,x) × (okf,ikf)
        wide = (df_L.pivot_table(index=["z", "x"], columns=["okf", "ikf"],  values="val", aggfunc=aggfunc,)
        .reindex(index=pd.MultiIndex.from_product([z_order, x_order]), columns=pd.MultiIndex.from_product([okf_order, ikf_order]),))


        # (K,P,R): K=z dims, P=X vars, R=reps
        K = len(z_order)
        P = len(x_order)
        R = wide.shape[1]

        Ls = wide.to_numpy().reshape(K, P, R)

        # IMPORTANT: keep real (okf,ikf) index for within_okf
        rep_index = list(wide.columns)  # list of tuples (okf, ikf) length R

        # Choose rep pairs
        rep_pairs = []
        if pairs_mode == "all":
            rep_pairs = [(a, b) for a in range(R) for b in range(a + 1, R)]
        elif pairs_mode == "within_okf":
            by_okf = {}
            for r, (okf, ikf) in enumerate(rep_index):
                by_okf.setdefault(okf, []).append(r)
            for okf, rr in by_okf.items():
                rr = sorted(rr)
                rep_pairs.extend([(rr[i], rr[j]) for i in range(len(rr)) for j in range(i + 1, len(rr))])
        else:
            raise ValueError("pairs_mode must be 'all' or 'within_okf'")

        # Collect per-(zdims,beta) rows for summarization
        stab_pairs = []
        for a, b in rep_pairs:

            Ls_a = Ls[:, :, a]  # (K,P)
            Ls_b = Ls[:, :, b]  # (K,P)

            # similarity matrix between latents (K,K)
            S = simil_matrix_loadings(Ls_a, Ls_b, metric=metric)

            # maximize similarity; Hungarian algorithm to minimize cost (that's why -S).
            row_ind, col_ind = linear_sum_assignment(-S)
            perm = col_ind[np.argsort(row_ind)]  # sort by first rep (a)

            # For this pair, these are the correlations that maximize the similarity
            mS_per_dim = np.array([S[d, perm[d]] for d in range(K)], dtype=float)

            # Do the Ls of both matched reps share tops?
            Ls_b_aligned = Ls_b[perm, :]  # (K,P)
            jacc_per_dim = topk_jaccard_per_dim(Ls_a, Ls_b_aligned, topk=topk)

            # Fraction of stable dims (per rep-pair)
            frac_sim_stable = float(np.nanmean(mS_per_dim >= sim_thr))
            frac_jacc_stable = float(np.nanmean(jacc_per_dim >= jacc_thr))

            okf_a, ikf_a = rep_index[a]
            okf_b, ikf_b = rep_index[b]

            stab_pairs.extend([{
                "block": block, "zdims": zdims, "beta": beta, "z":z,
                "rep_a": a, "rep_b": b, "okf_a": okf_a, "ikf_a": ikf_a, "okf_b": okf_b, "ikf_b": ikf_b,
                "simil": mS_per_dim[z], "simil_frac": frac_sim_stable,
                "jacc": jacc_per_dim[z], "jacc_frac": frac_jacc_stable} for z in range(zdims)])

        df_temp = pd.DataFrame(stab_pairs)


        # Global stability
        stab.extend([
            {"block": block, "zdims": zdims, "beta": beta,
            "simil_mean": df_temp["simil"].mean() , "simil_min":df_temp["simil"].min(), "simil_frac":df_temp["simil_frac"].mean(),
            "jacc_mean": df_temp["jacc"].mean(), "jacc_min": df_temp["jacc"].min(), "jacc_frac":df_temp["jacc_frac"].mean(),
            "n_pairs": int(len(rep_pairs)), "metric": metric, "topk": int(topk), "pairs_mode": pairs_mode, "sim_thr": float(sim_thr),"jacc_thr": float(jacc_thr),}])

        # Stability per dimension
        for z in range(zdims):
            df_sub = df_temp[df_temp["z"]==z]
            stab_perdim.extend([{"block": block, "zdims": zdims, "beta": beta,
                        "simil_mean": df_sub["simil"].mean() , "simil_min":df_sub["simil"].min(), "simil_frac":df_sub["simil_frac"].mean(),
                        "jacc_mean": df_sub["jacc"].mean(), "jacc_min": df_sub["jacc"].min(), "jacc_frac":df_sub["jacc_frac"].mean()}])

    return pd.DataFrame(stab), pd.DataFrame(stab_perdim)


def assess_downstream(model, data, exp_vars, sim_vars, scaler, block, zdims, beta, ikf, okf,
                      covars=None, test_size=0.2, n_splits=1, seed=0, order_by="t", verbose=False):
    """
    Devuelve:
      - res: resultados globales (baseline/full) evaluados en TEST
      - res_per_dim: coeficientes del FULL (fit en TRAIN)
      - res_curve: curva incremental R2_test(m) y ΔR2_test(m)

    :param data: data from simulations
    :param exp_vars: Exposome variables in the block to encode.
    :param sim_vars: Simulation outcomes to target in GAM
        vars_down:  downstream associations options -
        # out: ['ent_E', 'ent_I',  'rate_E', 'rate_I',  'target', 'EI_ent', 'EI_rate']
        # net: ['DMN', 'DAN', 'VAN', 'SMN', 'VIS', 'LIM', 'FPN'];
        # roi: range(90) - AAL regions.

    :param vae_model: used to ENCODE data into latent
    :param scaler:
    :param block:
    :param zdims:
    :param beta:
    :param ikf:
    :param okf:
    :param gam_type: ["linear", "splines"]
    :param verbose:
    :return:
    """

    covars = covars or []

    tic = time.time()


    df = data[covars + exp_vars + sim_vars].copy() # Omit 1 subject with Age=None

    # 1) Exposome -> Latents -  Prepare the data
    X = df.loc[:, exp_vars].values
    X_sc = scaler.transform(X)
    X_sc_fill = np.nan_to_num(X_sc, nan=0.0)

    if model is None:
        model_name = "FEAT"
        Z = X_sc_fill

    elif model == "Z_vae":
        model_name = model
        Z = X_sc_fill

    elif hasattr(model, "encode"):
        model_name = "mVAE"
        # Use the model to encode the Exposome: latent
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(torch.tensor(X_sc_fill, dtype=torch.float32))
        Z = mu.detach().cpu().numpy()  # (n, zdim)

    elif hasattr(model, "transform"):
        model_name = "PCA"
        Z = model.transform(X_sc_fill)

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]

    # DataFrame de latentes
    zcols = [f"b{block[0]}_z{i}" for i in range(Z.shape[1])]
    dfZ = pd.DataFrame(Z, columns=zcols, index=df.index)


    # --- 2) Split 80/20 (repetible) ---
    splitter = sklearn.model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    res, res_per_dim, res_curve = [], [], []

    for split_id, (idx_tr, idx_te) in enumerate(splitter.split(df)):
        pass

        # Prepara matrices de covariables
        if covars:
            Xcov_tr = df.iloc[idx_tr][covars].copy()
            Xcov_te = df.iloc[idx_te][covars].copy()
            # Si hay categóricas (country), conviértelas a dummies aquí:
            Xcov_tr = pd.get_dummies(Xcov_tr, drop_first=True)
            Xcov_te = pd.get_dummies(Xcov_te, drop_first=True)
            # alinear columnas train/test
            Xcov_te = Xcov_te.reindex(columns=Xcov_tr.columns, fill_value=0.0)
        else:
            Xcov_tr = pd.DataFrame(index=df.iloc[idx_tr].index)
            Xcov_te = pd.DataFrame(index=df.iloc[idx_te].index)

        # latentes train/test
        Z_tr = dfZ.iloc[idx_tr].copy()
        Z_te = dfZ.iloc[idx_te].copy()

        # Design matrices (baseline y full)
        X_base_tr = sm.add_constant(Xcov_tr, has_constant="add").astype(float)
        X_base_te = sm.add_constant(Xcov_te, has_constant="add").astype(float)

        X_full_tr = sm.add_constant(pd.concat([Xcov_tr, Z_tr], axis=1), has_constant="add").astype(float)
        X_full_te = sm.add_constant(pd.concat([Xcov_te, Z_te], axis=1), has_constant="add").astype(float)

        for sv, sim_var in enumerate(sim_vars):
            pass

            # 2) target - Prepare simulated data
            y_tr = df.iloc[idx_tr][sim_var].values
            y_te = df.iloc[idx_te][sim_var].values


            # --- Baseline ---
            ols_base = sm.OLS(y_tr, X_base_tr).fit()
            yhat_base_te = ols_base.predict(X_base_te)
            r2_base_te = sklearn.metrics.r2_score(y_te, yhat_base_te)

            # --- Full ---
            ols_full = sm.OLS(y_tr, X_full_tr).fit()
            yhat_full_te = ols_full.predict(X_full_te)
            r2_full_te = sklearn.metrics.r2_score(y_te, yhat_full_te)

            f2_test = (r2_full_te - r2_base_te) / max(1 - r2_full_te, 1e-12)

            # --- modelo global ---
            res.append({
                "model":model_name, "block": block, "zdims": zdims, "beta": beta, "ikf": ikf, "okf": okf,
                "split": split_id, "sim_var": sim_var, "n_covars": len(covars), "n_pred": len(zcols),
                # ---- Test metrics (real performance) ----
                "r2_base_test": r2_base_te, "r2_full_test": r2_full_te, "delta_r2_test": (r2_full_te - r2_base_te),
                "f2_test": f2_test,
                # ---- Train diagnostics ----
                "r2_train": ols_full.rsquared, "r2_adj_train": ols_full.rsquared_adj, "aic_train": ols_full.aic,
                "bic_train": ols_full.bic, "f_pvalue_train": ols_full.f_pvalue,

                "n_train":len(idx_tr), "n_test":len(idx_te),
            })

            # Coefs FULL (en TRAIN): útiles para ranking/interpretación
            for term in ols_full.params.index:
                res_per_dim.append({
                    "model": model_name,
                    "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf,
                    "split": split_id,
                    "sim_var": sim_var, "n_covars": len(covars), "n_pred": len(zcols),
                    "term": term,
                    "coef": float(ols_full.params[term]),
                    "se": float(ols_full.bse[term]),
                    "t": float(ols_full.tvalues[term]),
                    "pval": float(ols_full.pvalues[term]),
                })


            # --- 3) Curva incremental (orden rápido sin LASSO) ---
            # Orden SOLO con TRAIN (para no mirar el test)
            if order_by == "t":
                # ordenar solo latentes (z*) por |t|
                tvals = ols_full.tvalues.reindex(zcols)
                order = tvals.abs().sort_values(ascending=False).index.tolist()
            elif order_by == "absbeta":
                betas = ols_full.params.reindex(zcols)
                order = betas.abs().sort_values(ascending=False).index.tolist()
            else:
                raise ValueError("order_by debe ser 't' o 'absbeta'.")


            # Construye curva evaluada en TEST
            for m in range(1, min(len(order) + 1, zdims + 1)):
                topm = order[:m]

                X_m_tr = sm.add_constant(pd.concat([Xcov_tr, Z_tr[topm]], axis=1), has_constant="add").astype(float)
                X_m_te = sm.add_constant(pd.concat([Xcov_te, Z_te[topm]], axis=1), has_constant="add").astype(float)

                ols_m = sm.OLS(y_tr, X_m_tr).fit()
                yhat_m_te = ols_m.predict(X_m_te)
                r2_m_te = sklearn.metrics.r2_score(y_te, yhat_m_te)

                res_curve.append({
                    "model": model_name,
                    "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf,
                    "split": split_id,
                    "sim_var": sim_var, "n_covars": len(covars), "n_pred": len(zcols),
                    "m": m,
                    "r2_test": r2_m_te,
                    "delta_r2_test": (r2_m_te - r2_base_te),
                    "order_by": order_by,
                })

            if verbose:
                print(f"[Downstream] split{split_id+1}/{n_splits} . {sim_var} ({sv}/{len(sim_vars)})", end="\r")

    if verbose:
        print(f"\tDownstream associations ({round((time.time() - tic), 2)}sec) . ")



    return Z, pd.DataFrame(res), pd.DataFrame(res_per_dim), pd.DataFrame(res_curve)


def assess_mVAE_fc(X, M, fnames, vae_model, zdims, beta, hdims,  ikf, okf, vae_dir, v_tag,
                            avoid=None, kl_th=0.01, tw_n=15, seed=42, verbose=False):

    tic0 = time.time()

    if avoid is None:
        avoid = []

    res = {"model":"mVAE", "zdims": zdims, "beta": beta, "hdims": hdims,  "kl_th":kl_th, "ikf":ikf, "okf":okf}
    res_per_dim = []

    # %% 4.4) Deterministc evaluation of VAE on outer test
    X_test_sc_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)

    vae_model.eval()
    with torch.no_grad():
        mu, logvar = vae_model.encode(X_test_sc_t)  # (n_test, zdim)
        Xhat = vae_model.decode(mu)  # (n_test, xdim)

    mu_perdim, logvar_perdim = mu.mean(dim=0).detach().cpu().numpy(), logvar.mean(dim=0).detach().cpu().numpy()

    # Reconstruction errors
    se = (Xhat - X_test_sc_t).pow(2)
    ae = (Xhat - X_test_sc_t).abs()

    n_obs = M_t.sum()
    mse_obs = (se * M_t).sum() / n_obs
    mae_obs = (ae * M_t).sum() / n_obs

    # Per-feature (xdim) masked errors
    obs_per_xdim = M_t.sum(dim=0)    # (xdim,)
    se_per_xdim = (se * M_t).sum(dim=0) / obs_per_xdim
    ae_per_xdim = (ae * M_t).sum(dim=0) / obs_per_xdim

    # Per-sample masked errors
    obs_per_sample = M_t.sum(dim=1)  # (n_test,)
    se_per_sample = (se * M_t).sum(dim=1) / obs_per_sample
    ae_per_sample = (ae * M_t).sum(dim=1) / obs_per_sample

    # --- R² (masked) ---
    mean_ref_t = X_test_sc_t.mean(dim=0, keepdim=True)  # fallback si no guardas train_mean
    sst_obs = ((X_test_sc_t - mean_ref_t).pow(2) * M_t).sum() / n_obs
    r2_obs = 1.0 - (mse_obs / sst_obs)

    # --- KL diagnostics ---
    kl_per_dim_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (n_test, zdim)
    kl_per_sample = kl_per_dim_sample.sum(dim=1)  # (n_test,)
    kl = kl_per_sample.mean().item()
    kl_per_dim = kl_per_dim_sample.mean(dim=0).cpu().numpy()

    active_dims = int((kl_per_dim > kl_th).sum())

    # Scalars
    mse = mse_obs.item()
    mae = mae_obs.item()
    sst = sst_obs.item()
    r2 = r2_obs.item()

    # --- Loss / ELBO proxy (note: recon term should match training; if training uses masked MSE, do same) ---
    # loss_per_sample = se_per_sample + beta * kl_per_sample
    loss = mse + beta * kl

    res.update({"loss": loss, "mse": mse, "mae": mae, "kl": kl, "sst":sst, "r2":r2,
                "kl_perdim_avg":kl_per_dim.mean(), "kl_act_n": active_dims, "kl_act_perc":active_dims/zdims})

    if "per_dim" not in avoid:
        res_per_dim.extend([{"model":"mVAE", "zdims": zdims, "beta": beta, "hdims":hdims, "ikf":ikf, "okf":okf, "z": z,
                             f"var": "kl", "val":kl_per_dim[z], "mu":mu_perdim[z], "logvar":logvar_perdim[z]} for z in range(zdims)])
        res_per_dim.extend([{"model":"mVAE", "zdims": zdims, "beta": beta, "hdims":hdims, "ikf":ikf, "okf":okf, "x": x, "x_fname":fnames[x],
                             f"var": "se_vae", "val":se_per_xdim[x].item()} for x in range(X.shape[1])])

    if verbose:
        print(f"\tDeterministic evaluation ({round((time.time() - tic0), 2)}sec) >> ", end="")



    if "latent_dist" not in avoid:

        tic_ass = time.time()

        sigma_perdim = np.exp(0.5 * logvar_perdim)

        # --- layout de subplots ---
        ncols = int(min(4, zdims))  # cámbialo si quieres
        nrows = int(np.ceil(zdims / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols, shared_yaxes=True,
            subplot_titles=[f"z{d} | KL={kl_per_dim[d]:.3f}" for d in range(zdims)],
            horizontal_spacing=0.06, vertical_spacing=0.10,
        )

        # --- eje x (mismo para todos; basado en rangos de mu±4σ, y también [-4,4]) ---
        left = np.min(mu_perdim - 4 * sigma_perdim)
        right = np.max(mu_perdim + 4 * sigma_perdim)
        x_min = min(-4, left)
        x_max = max(4, right)
        x = np.linspace(x_min, x_max, 400)

        # --- trazas ---
        for d in range(zdims):

            r = d // ncols + 1
            c = d % ncols + 1

            # q(z_d) ~ N(mu_d, sigma_d^2)
            y_q = norm.pdf(x, loc=mu_perdim[d], scale=max(sigma_perdim[d], 1e-8))
            # referencia N(0,1)
            y_ref = norm.pdf(x, loc=0.0, scale=1.0)

            # referencia (gris claro)
            fig.add_trace(go.Scatter(x=x, y=y_ref, mode="lines", name="N(0,1)",
                    line=dict(color="lightgray", width=1), showlegend=(d == 0),), row=r, col=c)

            # distribución del latente
            fig.add_trace(go.Scatter(x=x, y=y_q, mode="lines", name="q(z)",
                    line=dict(width=2, color="dimgrey"), showlegend=(d == 0),),row=r, col=c)

        # --- estética ---
        fig.update_layout(template="plotly_white", height=260 * nrows, width=340 * ncols,
            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0),)

        # opcional: quitar labels repetidos si molesta
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        pio.write_image(fig, os.path.join(vae_dir, v_tag, f"latent_dists.png"))

        if verbose:
            print(f"\tPlot Latent distributions ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    Z_mu = mu.detach().cpu().numpy() # prepare other assessments

    if "geometry" not in avoid:
        tic_ass = time.time()

        # Intrinsic dimension (TwoNN) en el latente
        id_twonn = intrinsic_dim_twonn(Z_mu, seed=seed)

        # Trustworthiness: X->Z (preserva vecinos del espacio original en el latente)
        tw = sklearn.manifold.trustworthiness(X, Z_mu, n_neighbors=tw_n)

        res.update({"intrinsic_dim_twonn": id_twonn, "tw_n":tw_n, "trustworthyness": tw})

        if verbose:
            print(f"\tGeometric evaluation ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    # %% 4.6) Latent correlations
    if "corrs" not in avoid:

        tic_ass = time.time()

        # 1,Latent Dimensions correlation
        r_z = np.ones((1, 1), dtype=float) if zdims == 1 else np.corrcoef(Z_mu, rowvar=False)
        # plt.imshow(r_z, aspect="auto"); plt.colorbar(); plt.title("VAE - r(latent dims)")
        # plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__rLatent.png")); plt.close()
        r_z_triu = r_z[np.triu_indices(zdims, k=1)] if zdims > 1 else 0
        res.update({"r_z": np.average(np.abs(r_z_triu))})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"mVAE",  "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z1, "z2": z2,
                                 f"var": "r_z", "val":r_z[z1, z2]} for z1 in range(zdims) for z2 in range(zdims) if z1>z2])

        # 2, Missingness - Latent correlations
        obs_row = M.mean(axis=1)
        r_zM_per_dim = [np.corrcoef(Z_mu[:, z], obs_row, rowvar=False)[0, 1] if np.std(obs_row)!=0 else np.nan for z in range(zdims) ]
        res.update({"r_zM":np.average(r_zM_per_dim)})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"mVAE",  "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z,
                                 f"var": "r_zM", "val":r_zM_per_dim[z]} for z in range(zdims)])

        # 3, Variables Loadings on Latent (thorugh correlations)
        r_zX = np.corrcoef(Z_mu, X, rowvar=False)[zdims:, :zdims] # (X in rows, z in cols)
        # np concatenates the arrays by shared dim, and corrs all columns.
        res_per_dim.extend([{"model":"mVAE", "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z, "x": x, "x_fname":fnames[x],
                             f"var": "r_zX", "val":r_zX[x, z]} for z in range(zdims) for x in range(X.shape[1])])

        ## Calculate the Effective number of variables (derived from Herfindahl index)
        eps = 1e-12  # numerical stability
        p_per_dim = [r_zX[:, z] / r_zX[:, z].sum() + eps  for z in range(zdims)] # contribution proportion
        H_per_dim = [np.sum(p ** 2) for p in p_per_dim] # higher contributions p, increases the concentration (H index)
        EN_per_dim = [1.0 / max(h, eps) for h in H_per_dim]
        # EN ≈ 1–3 → excelente; EN ≈ 3–5 → aceptable; EN > 6–7 → factor poco interpretable
        res.update({"EN_vars": np.average(EN_per_dim)})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"mVAE", "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z,
                                 f"var": "EN_vars", "val":EN_per_dim[z]} for z in range(zdims)])

        if verbose:
            print(f"\tLatent correlations ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    # %% 4.7) SHAP per latent dimension
    if "shap" not in avoid:
        tic_ass = time.time()

        f = VAE_encoder_wrapper(vae_model)  # forward wrapper (from X to mu, only)

        # --- subsample background ---
        n_bg = min(100, X.shape[0])
        idx_bg = np.random.choice(X.shape[0], size=n_bg, replace=False)
        X_bg = torch.tensor(X[idx_bg], dtype=torch.float32)

        # --- subsample explained set ---
        n_exp = min(500, X_test_sc_t.shape[0])
        idx_exp = np.random.choice(X_test_sc_t.shape[0], size=n_exp, replace=False)
        X_exp = X_test_sc_t[idx_exp]  # ya es tensor en tu código

        explainer = shap.GradientExplainer(f, X_bg)
        shap_values = explainer(X_exp)

        # z = 0  # Iterate over dimensions
        for z in range(zdims):
            shap_exp = shap.Explanation(values=shap_values.values[:, :, z], data=X_exp, feature_names=fnames)

            shap.plots.beeswarm(shap_exp, max_display=shap_exp.shape[1], show=False)
            plt.gcf().set_size_inches(13, 18)  # más alto si tienes muchas features
            plt.gcf().subplots_adjust(left=0.45)  # más margen para labels largos
            plt.tight_layout();
            plt.title(f"Latent dim: {z}")
            plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__SHAP_beeswarm__z{z}.png"))
            plt.close()

            shap.plots.bar(shap_exp, max_display=shap_exp.shape[1], show=False)
            plt.gcf().set_size_inches(15, 18)  # más alto si tienes muchas features
            plt.gcf().subplots_adjust(left=0.45, top=0.95, bottom=0.05)  # más margen para labels largos
            plt.title(f"Latent dim: {z}");
            plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__SHAP_bars__z{z}.png"))
            plt.close()

            if "per_dim" not in avoid:
                res_per_dim.extend([{"model":"mVAE", "zdims": zdims, "beta": beta, "ikf": ikf, "okf": okf, "z": z, "x": x,
                                     "x_fname": fnames[x],
                                     f"var": "shap", "val": np.average(shap_values.values[:, x, z])} for x in
                                    range(shap_exp.shape[1])])

        if verbose:
            print(f"\tSHAP ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    # %% 4.3) PLOT training RESULTS
    if "train_viz" not in avoid:
        tic_ass = time.time()

        df_metrics_w = pd.read_csv(os.path.join(vae_dir, v_tag, "metrics.csv"))
        df_metrics_w = df_metrics_w.groupby(["epoch", "step"], as_index=False).first()

        df_metrics_l = df_metrics_w.melt(  # convert wide to long for plotly.express
            id_vars=["epoch", "step"], var_name="metric", value_name="value",
            value_vars=["train_kl", "train_loss", "train_recon", "val_kl", "val_loss", "val_recon"], )
        df_metrics_l = df_metrics_l.dropna()

        df_metrics_l[["split", "metric_name"]] = (df_metrics_l["metric"].str.split("_", expand=True))
        fig = px.line(df_metrics_l, x="epoch", y="value", color="split", facet_row="metric_name", title=v_tag)
        df_fbeta = df_metrics_w.loc[:, ["epoch", "val_loss_fbeta"]].dropna()
        fig.add_trace(go.Scatter(x=df_fbeta["epoch"], y=df_fbeta["val_loss_fbeta"], mode="lines",
                                 line=dict(color="indianred", dash="dash"), name="val_loss (fixed beta)"))
        fig.update_yaxes(matches=None)  # permite escalas distintas por fila (clave para KL)
        fig.update_layout(template="plotly_white", width=700, height=600, legend=dict(orientation="h", y=1.1, x=0.2),
                          yaxis1=dict(title="Loss"), yaxis2=dict(title="Reconstruction error"),
                          yaxis3=dict(title="Kullback-Leibler<br>divergence"),
                          )
        pio.write_image(fig, os.path.join(vae_dir, v_tag, f"VAE__TrainingMetrics.png"), width=700, height=600, scale=3)
        del fig

        if verbose:
            print(f"\tPlot training results ({round((time.time() - tic_ass), 2)}sec) >> ", end="")


    # %% 4.5) Visualization of the latent; reduced with UMAP if zdim>=3.
    if "latent_viz" not in avoid:

        tic_ass = time.time()

        df_test = pd.DataFrame(X, columns=fnames)

        df_test["miss_pct"] = df_test.isna().mean(axis=1).values
        df_test["mse"] = se_per_sample.detach().numpy()
        df_test["kl"] = kl_per_sample.detach().numpy()

        vars_order = ["mse", "kl"] + fnames
        nrows = int(np.ceil((df_test.shape[1] - 1) / 2))
        y_tit = "z2" if zdims > 1 else "Jitter (viz 1d)"
        size, max_op = 4, 1

        if zdims > 2:
            ncols, width = 4, 1050
            sp_titles = [v for i in range(0, len(vars_order), 2) for v in vars_order[i:i + 2] * 2]
            # UMAP reduction
            umap_model = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2, metric="euclidean", )  # random_state=42,
            Z_umap = umap_model.fit_transform(Z_mu)
            # PCA reduction
            pca = sklearn.decomposition.PCA(n_components=zdims, random_state=seed)
            Z_pca = pca.fit_transform(Z_mu)
        else:
            ncols, width = 2, 700
            sp_titles = vars_order

        fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, subplot_titles=sp_titles,
                            x_title="z1", y_title=y_tit)
        for ii, var in enumerate(sp_titles):

            row, col = (ii // ncols) + 1, (ii % ncols) + 1
            hover = [(f"{var} = {row[var]:.2f}<br>"
                      f"missing = {row['miss_pct']:.2f}<br>"
                      f"Country x Year: {i}") for i, row in df_test.iterrows()]

            if zdims == 1:
                rng = np.random.default_rng(0)
                jitter = rng.normal(0, 0.03, size=len(Z_mu))  # σ controla el ancho visual
                fig.add_trace(go.Scatter(x=Z_mu[:, 0], y=jitter, mode="markers", showlegend=False, hoverinfo="x,y,text",
                                         hovertext=hover,
                                         marker=dict(size=size, color=df_test[var],
                                                     opacity=max_op - df_test["miss_pct"].values, colorscale="Viridis")),
                              row=row, col=col)
            elif zdims == 2:
                fig.add_trace(go.Scatter(x=Z_mu[:, 0], y=Z_mu[:, 1], mode="markers", showlegend=False, hoverinfo="x,y,text",
                                         hovertext=hover,
                                         marker=dict(size=size, color=df_test[var],
                                                     opacity=max_op - df_test["miss_pct"].values, colorscale="Viridis")),
                              row=row, col=col)
            else:
                Z_ = Z_pca if col <= 2 else Z_umap
                fig.add_trace(go.Scatter(x=Z_[:, 0], y=Z_[:, 1], mode="markers", showlegend=False, hoverinfo="x,y,text",
                                         hovertext=hover,
                                         marker=dict(size=size, color=df_test[var],
                                                     opacity=max_op - df_test["miss_pct"].values, colorscale="Viridis")),
                              row=row, col=col)
                fig.add_shape(type="line", xref="paper", yref="paper", x0=0.5, x1=0.5, y0=0.0, y1=1.0,
                              line=dict(color="black", width=1, dash="dot"), )
                fig.add_annotation(xref="paper", yref="paper", x=0.25, y=1.025, text="PCA", showarrow=False,
                                   font=dict(size=14, color="black"), )
                fig.add_annotation(xref="paper", yref="paper", x=0.75, y=1.025, text="UMAP", showarrow=False,
                                   font=dict(size=14, color="black"), )
        fig.update_layout(template="plotly_white", title=v_tag, width=width, height=nrows * 150)
        # fig.show("browser")
        pio.write_image(fig, os.path.join(vae_dir, v_tag, f"VAE__latent-viz.png"), width=width, height=nrows * 150,
                        scale=3)
        pio.write_html(fig, os.path.join(vae_dir, v_tag, f"VAE__latent-viz.html"), auto_open=False)
        del fig

        if verbose:
            print(f"\tLatent visualization ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    if verbose:
        print(f"  Total time ({round((time.time() - tic0), 2)}sec) #", end="\n")

    return  Z_mu, pd.DataFrame([res]), pd.DataFrame(res_per_dim)



# %% --    utilities


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def ordinal_probs_from_eta_tau_torch(eta, tau):
    """
    eta: (B,)
    tau: (K-1,)
    returns probs: (B, K)
    """
    cdf = torch.sigmoid(tau.unsqueeze(0) - eta.unsqueeze(1))

    p_first = cdf[:, :1]
    p_middle = cdf[:, 1:] - cdf[:, :-1]
    p_last = 1.0 - cdf[:, -1:]

    probs = torch.cat([p_first, p_middle, p_last], dim=1)
    probs = probs.clamp_min(1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs

def ordinal_probs_from_eta_tau_np(eta, tau):
    """
    eta: scalar o array 1D
    tau: array 1D de longitud K-1
    returns probs: (K,)
    """
    eta = np.asarray(eta, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()

    if eta.size != 1:
        raise ValueError("ordinal_probs_from_eta_tau_np espera un eta escalar o de longitud 1.")

    eta = eta[0]

    cdf = sigmoid_np(tau - eta)

    p0 = np.array([cdf[0]])
    pmid = cdf[1:] - cdf[:-1] if len(cdf) > 1 else np.array([])
    plast = np.array([1.0 - cdf[-1]])

    probs = np.concatenate([p0, pmid, plast], axis=0)
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / probs.sum()
    return probs

def expected_ordinal_value(probs):
    cats = np.arange(len(probs))
    return np.sum(cats * probs)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def nanmode_int(x):
    """
    Mode for categorical/integer data ignoring NaNs.
    Returns np.nan if there are no observed values.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan

    x = x.astype(int)
    vals, counts = np.unique(x, return_counts=True)
    return int(vals[np.argmax(counts)])


def nanmedian_int(x):
    """
    Median for ordinal/integer data ignoring NaNs.
    Rounded to nearest integer and clipped implicitly by observed support.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan

    return int(np.round(np.nanmedian(x)))



def build_baseline_constants(X_train_evalspace, M_train,
                             fnames, idx_num, idx_bin, idx_ord, vars_meta):
    """
    Estimate one constant prediction per variable using TRAIN only,
    in the same space where evaluation is performed.

    Parameters
    ----------
    X_train_evalspace : array-like, shape (N, D)
        Training data in evaluation space.
        In your pipeline this should be X_train_scmix_fill:
        - numeric vars already scaled
        - binary / ordinal unchanged
        - NaNs already filled

    M_train : array-like, shape (N, D)
        Observation mask from raw train data, 1 if observed, 0 if missing.

    Notes
    -----
    - Numeric baseline: mean of observed train values in eval space
    - Binary baseline: mode of observed train values
    - Ordinal baseline: mode of observed train values
    """
    X_train_evalspace = np.asarray(X_train_evalspace, dtype=float)
    M_train = np.asarray(M_train, dtype=float)

    var_rows = build_variable_table(fnames, idx_num, idx_bin, idx_ord, vars_meta)
    baseline = {}

    for info in var_rows:
        gidx = info["gidx"]
        vtype = info["type"]

        obs = M_train[:, gidx].astype(bool)
        x_obs = X_train_evalspace[:, gidx][obs]

        if len(x_obs) == 0:
            const = np.nan

        elif vtype == "numeric":
            const = float(np.mean(x_obs))

        elif vtype == "binary":
            const = nanmode_int(x_obs)

        elif vtype == "ordinal":
            const = nanmode_int(x_obs)

        else:
            raise ValueError(vtype)

        baseline[info["var"]] = {
            "const": const,
            "type": vtype,
            "strategy": "mean" if vtype == "numeric" else "mode"
        }

    return baseline



def confusion_2x2(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    cm = np.array([[tn, fp],
                   [fn, tp]])

    acc = (tp + tn) / max(cm.sum(), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    return cm, {"acc": acc, "sens": sens, "spec": spec, "n": int(cm.sum())}


def confusion_k(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    acc = np.trace(cm) / max(cm.sum(), 1)
    mae_cat = np.mean(np.abs(y_true - y_pred)) if len(y_true) > 0 else np.nan
    rank_r = safe_corr(y_true, y_pred)

    return cm, {"acc": acc, "mae_cat": mae_cat, "rank_r": rank_r, "n": int(cm.sum())}


def get_latent_reference(model, x_ref, m_ref):
    device = next(model.parameters()).device
    model.eval()

    x_ref_t = torch.tensor(x_ref, dtype=torch.float32, device=device)
    m_ref_t = torch.tensor(m_ref, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_ref, _ = model.encode(x_ref_t, m_ref_t)

    return mu_ref.mean(dim=0).cpu().numpy()


# --------------------------------------------------
# latent traversal (left column)
# --------------------------------------------------
def decode_latent_traversal_scalar(model, zdim, grid, z_ref=None):
    device = next(model.parameters()).device
    model.eval()

    latent_dim = model.hparams.latent_dim

    if z_ref is None:
        z_ref = np.zeros(latent_dim, dtype=np.float32)
    else:
        z_ref = np.asarray(z_ref, dtype=np.float32).copy()

    Z = np.tile(z_ref[None, :], (len(grid), 1))
    Z[:, zdim] = grid
    Z_t = torch.tensor(Z, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model.decode(Z_t)

    result = {"zgrid": np.asarray(grid), "num": None, "bin": None, "ord": None}

    if out["num_mu"] is not None:
        result["num"] = out["num_mu"].detach().cpu().numpy()

    if out["bin"] is not None:
        result["bin"] = sigmoid_np(out["bin"].detach().cpu().numpy())

    if out["ord"] is not None and len(out["ord"]) > 0:
        ord_exp = []
        for eta_t, tau_t in out["ord"]:
            eta = eta_t.detach().cpu().numpy()
            tau = tau_t.detach().cpu().numpy()
            exp_vals = []
            for e in eta:
                probs = ordinal_probs_from_eta_tau_np(e, tau)
                exp_vals.append(expected_ordinal_value(probs))
            ord_exp.append(exp_vals)
        result["ord"] = np.array(ord_exp).T


    return result


def build_variable_table(fnames, idx_num, idx_bin, idx_ord, vars_meta):
    meta = vars_meta.set_index("name")

    map_num = {g: j for j, g in enumerate(idx_num)}
    map_bin = {g: j for j, g in enumerate(idx_bin)}
    map_ord = {g: j for j, g in enumerate(idx_ord)}

    rows = []
    for gidx, var in enumerate(fnames):
        if gidx in map_num:
            vtype = "numeric"
            local_idx = map_num[gidx]
        elif gidx in map_bin:
            vtype = "binary"
            local_idx = map_bin[gidx]
        elif gidx in map_ord:
            vtype = "ordinal"
            local_idx = map_ord[gidx]
        else:
            continue

        long_name = meta.at[var, "long_name"] if var in meta.index else var

        rows.append({
            "gidx": gidx,
            "var": var,
            "type": vtype,
            "local_idx": local_idx,
            "long_name": str(long_name),
        })

    return rows


def extract_series_for_var(dec, var_info):
    vtype = var_info["type"]
    j = var_info["local_idx"]

    if vtype == "numeric":
        return dec["num"][:, j]
    elif vtype == "binary":
        return dec["bin"][:, j]
    elif vtype == "ordinal":
        return dec["ord"][:, j]
    else:
        raise ValueError(vtype)


# --------------------------------------------------
# reconstruction diagnostics (middle column + stats)
# --------------------------------------------------
def compute_reconstruction_payload(model, x_eval, m_eval):
    """
    Deterministic reconstruction using z = mu.
    """
    device = next(model.parameters()).device
    model.eval()

    x_t = torch.tensor(x_eval, dtype=torch.float32, device=device)
    m_t = torch.tensor(m_eval, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu, _ = model.encode(x_t, m_t)
        out = model.decode(mu)

    payload = {"num": None, "bin_prob": None, "bin_pred": None, "ord_probs": None, "ord_pred": None}

    if out["num_mu"] is not None:
        payload["num"] = out["num_mu"].detach().cpu().numpy()

    if out["bin"] is not None:
        bin_prob = sigmoid_np(out["bin"].detach().cpu().numpy())
        payload["bin_prob"] = bin_prob
        payload["bin_pred"] = (bin_prob >= 0.5).astype(int)

    if out["ord"] is not None and len(out["ord"]) > 0:
        probs_list = []
        pred_list = []

        for eta_t, tau_t in out["ord"]:
            probs_t = ordinal_probs_from_eta_tau_torch(eta_t, tau_t)  # (N, K)
            probs = probs_t.detach().cpu().numpy()
            pred = np.argmax(probs, axis=1)

            probs_list.append(probs)
            pred_list.append(pred)

        payload["ord_probs"] = probs_list
        payload["ord_pred"] = pred_list


    return payload


def build_reconstruction_info(model, x_eval, m_eval, fnames, idx_num, idx_bin, idx_ord, vars_meta):
    """
    Per-variable true/pred/mask/stats for plotting and annotation.
    """
    rec = compute_reconstruction_payload(model, x_eval, m_eval)
    var_rows = build_variable_table(fnames, idx_num, idx_bin, idx_ord, vars_meta)

    x_eval = np.asarray(x_eval)
    m_eval = np.asarray(m_eval)

    recon_info = []

    for info in var_rows:
        gidx = info["gidx"]
        j = info["local_idx"]
        vtype = info["type"]

        obs = m_eval[:, gidx].astype(bool)
        y_true = x_eval[:, gidx][obs]

        item = dict(info)
        item["n_obs"] = int(obs.sum())

        if vtype == "numeric":
            y_pred = rec["num"][:, j][obs]

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) if len(y_true) > 0 else np.nan
            mae = np.mean(np.abs(y_true - y_pred)) if len(y_true) > 0 else np.nan
            r = safe_corr(y_true, y_pred)

            item["plot_kind"] = "numeric"
            item["y_true"] = y_true
            item["y_pred"] = y_pred
            item["stats"] = {"r": r, "rmse": rmse, "mae": mae, "n": len(y_true)}

        elif vtype == "binary":
            prob = rec["bin_prob"][:, j][obs]
            pred = rec["bin_pred"][:, j][obs].astype(int)
            y_true_bin = y_true.astype(int)

            cm, stats = confusion_2x2(y_true_bin, pred)

            item["plot_kind"] = "binary"
            item["y_true"] = y_true_bin
            item["y_prob"] = prob
            item["y_pred"] = pred
            item["cm"] = cm
            item["stats"] = stats

        elif vtype == "ordinal":
            pred = rec["ord_pred"][j][obs].astype(int)
            y_true_ord = y_true.astype(int)

            n_classes = rec["ord_probs"][j].shape[1]
            cm, stats = confusion_k(y_true_ord, pred, n_classes)

            item["plot_kind"] = "ordinal"
            item["y_true"] = y_true_ord
            item["y_pred"] = pred
            item["cm"] = cm
            item["n_classes"] = n_classes
            item["stats"] = stats

        recon_info.append(item)

    return recon_info


def build_baseline_reconstruction_info(baseline_constants, x_eval, m_eval,
                                       fnames, idx_num, idx_bin, idx_ord, vars_meta):
    """
    Per-variable true/pred/mask/stats for a constant baseline estimated on train.
    Mirrors build_reconstruction_info().
    """
    var_rows = build_variable_table(fnames, idx_num, idx_bin, idx_ord, vars_meta)

    x_eval = np.asarray(x_eval)
    m_eval = np.asarray(m_eval)

    recon_info = []

    for info in var_rows:
        gidx = info["gidx"]
        vtype = info["type"]
        var = info["var"]

        obs = m_eval[:, gidx].astype(bool)
        y_true = x_eval[:, gidx][obs]

        item = dict(info)
        item["n_obs"] = int(obs.sum())
        item["baseline_const"] = baseline_constants[var]["const"]
        item["baseline_strategy"] = baseline_constants[var]["strategy"]

        const = baseline_constants[var]["const"]

        if vtype == "numeric":
            y_pred = np.full(len(y_true), const, dtype=float)

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) if len(y_true) > 0 else np.nan
            mae = np.mean(np.abs(y_true - y_pred)) if len(y_true) > 0 else np.nan
            r = safe_corr(y_true, y_pred)

            item["plot_kind"] = "numeric"
            item["y_true"] = y_true
            item["y_pred"] = y_pred
            item["stats"] = {"r": r, "rmse": rmse, "mae": mae, "n": len(y_true)}

        elif vtype == "binary":
            y_true_bin = y_true.astype(int)

            if baseline_constants[var]["strategy"] == "mean_prob":
                prob = np.full(len(y_true_bin), const, dtype=float)
                pred = (prob >= 0.5).astype(int)
            else:
                pred = np.full(len(y_true_bin), int(const), dtype=int)
                prob = pred.astype(float)

            cm, stats = confusion_2x2(y_true_bin, pred)

            item["plot_kind"] = "binary"
            item["y_true"] = y_true_bin
            item["y_prob"] = prob
            item["y_pred"] = pred
            item["cm"] = cm
            item["stats"] = stats

        elif vtype == "ordinal":
            y_true_ord = y_true.astype(int)
            pred = np.full(len(y_true_ord), int(const), dtype=int)

            # infer number of classes from observed eval support + pred
            vals = np.concatenate([y_true_ord, pred]) if len(y_true_ord) > 0 else pred
            n_classes = int(np.nanmax(vals)) + 1 if len(vals) > 0 else 1

            cm, stats = confusion_k(y_true_ord, pred, n_classes)

            item["plot_kind"] = "ordinal"
            item["y_true"] = y_true_ord
            item["y_pred"] = pred
            item["cm"] = cm
            item["n_classes"] = n_classes
            item["stats"] = stats

        recon_info.append(item)

    return recon_info


def stats_to_html(item):
    s = item["stats"]
    if item["type"] == "numeric":
        txt = (
            f"<b>{item['var']}</b> <i>{item['type']}</i><br>"
            f"{item['long_name']}<br>"
            f"n={s['n']} | r={s['r']:.3f} | RMSE={s['rmse']:.3f} | MAE={s['mae']:.3f}"
        )
    elif item["type"] == "binary":
        txt = (
            f"<b>{item['var']}</b> <i>{item['type']}</i><br>"
            f"{item['long_name']}<br>"
            f"n={s['n']} | acc={s['acc']:.3f} | sens={s['sens']:.3f} | spec={s['spec']:.3f}"
        )
    else:
        txt = (
            f"<b>{item['var']}</b> <i>{item['type']}</i><br>"
            f"{item['long_name']}<br>"
            f"n={s['n']} | acc={s['acc']:.3f} | MAE(cat)={s['mae_cat']:.3f} | rank-r={s['rank_r']:.3f}"
        )
    return txt


# --------------------------------------------------
# main plotting function
# --------------------------------------------------
def make_decoder_grid(model, block, fnames, idx_num, idx_bin, idx_ord, vars_meta, x_eval, m_eval,
                      zdim=0, kl=None, z_ref=None, grid=None, max_vars=None):
    """
    3 columns:
      col1 -> latent traversal line + moving marker
      col2 -> reconstruction diagnostic
      col3 -> annotation + reconstruction stats
    one row per variable
    """
    if grid is None:
        grid = np.linspace(-3, 3, 41)

    dec = decode_latent_traversal_scalar(model=model, zdim=zdim, grid=grid, z_ref=z_ref)
    zgrid = dec["zgrid"]
    init_idx = int(np.argmin(np.abs(grid)))

    recon_info = build_reconstruction_info(
        model=model,
        x_eval=x_eval,
        m_eval=m_eval,
        fnames=fnames,
        idx_num=idx_num,
        idx_bin=idx_bin,
        idx_ord=idx_ord,
        vars_meta=vars_meta,
    )

    if max_vars is not None:
        recon_info = recon_info[:max_vars]

    n_rows = len(recon_info)

    specs = []
    for item in recon_info:
        mid_type = "xy" if item["plot_kind"] == "numeric" else "heatmap"
        specs.append([{"type": "xy"}, {"type": mid_type}, {"type": "xy"}])

    fig = make_subplots(
        rows=n_rows,
        cols=3,
        specs=specs,
        column_widths=[0.35, 0.25, 0.4],
        horizontal_spacing=0.1,
        vertical_spacing=0.03,
        y_title="Decoded Values"
    )

    # left-column traversal ranges
    y_series_list = []
    y_ranges = []

    for item in recon_info:
        y = np.asarray(extract_series_for_var(dec, item), dtype=float)
        y_series_list.append(y)

        ymin = np.nanmin(y)
        ymax = np.nanmax(y)

        if item["type"] == "binary":
            yr = [-0.02, 1.02]
        elif item["type"] == "ordinal":
            upper = max(1.0, np.ceil(np.nanmax(y)))
            yr = [-0.1, upper + 0.1]
        else:
            if np.isclose(ymin, ymax):
                pad = 0.5 if np.isclose(ymin, 0) else abs(ymin) * 0.1 + 0.1
            else:
                pad = 0.1 * (ymax - ymin)
            yr = [ymin - pad, ymax + pad]

        y_ranges.append(yr)

    moving_trace_ids = []
    trace_counter = 0

    for r, (item, y, yr) in enumerate(zip(recon_info, y_series_list, y_ranges), start=1):
        var = item["var"]

        # -------------------------
        # col 1: traversal
        # -------------------------
        fig.add_trace(
            go.Scatter(
                x=zgrid,
                y=y,
                mode="lines",
                line_color="lightgrey",
                showlegend=False,
                hovertemplate=f"{var}<br>z=%{{x:.2f}}<br>decoded=%{{y:.3f}}<extra></extra>",
            ),
            row=r, col=1
        )
        trace_counter += 1

        fig.add_trace(
            go.Scatter(
                x=[zgrid[init_idx]],
                y=[y[init_idx]],
                mode="markers",
                marker=dict(size=5, symbol="circle", color="dimgrey"),
                showlegend=False,
                hovertemplate=f"{var}<br>z=%{{x:.2f}}<br>decoded=%{{y:.3f}}<extra></extra>",
            ),
            row=r, col=1
        )
        moving_trace_ids.append(trace_counter)
        trace_counter += 1

        fig.update_xaxes(range=[zgrid.min(), zgrid.max()], row=r, col=1)
        fig.update_yaxes(range=yr, row=r, col=1)

        # -------------------------
        # col 2: reconstruction behavior
        # -------------------------
        if item["plot_kind"] == "numeric":
            y_true = item["y_true"]
            y_pred = item["y_pred"]

            if len(y_true) > 0:
                lo = np.nanmin(np.r_[y_true, y_pred])
                hi = np.nanmax(np.r_[y_true, y_pred])
            else:
                lo, hi = 0, 1

            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    marker=dict(size=4, color="dimgrey", opacity=0.6),
                    showlegend=False,
                    hovertemplate="true=%{x:.3f}<br>pred=%{y:.3f}<extra></extra>",
                ),
                row=r, col=2
            )
            trace_counter += 1

            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    line=dict(color="lightgrey", dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r, col=2
            )
            trace_counter += 1

            fig.update_xaxes(title_text="true" if r == n_rows else None, row=r, col=2)
            fig.update_yaxes(title_text="pred", row=r, col=2)

        elif item["plot_kind"] == "binary":
            cm = item["cm"].transpose()

            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    y=["pred 0", "pred 1"],
                    x=["true 0", "true 1"],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    showscale=False,
                    hovertemplate="count=%{z}<extra></extra>",
                    colorscale="Blues",
                ),
                row=r, col=2
            )
            trace_counter += 1

        elif item["plot_kind"] == "ordinal":
            cm = item["cm"].transpose()
            K = item["n_classes"]
            labels = [str(k) for k in range(K)]

            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    y=[f"pred {k}" for k in labels],
                    x=[f"true {k}" for k in labels],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showscale=False,
                    hovertemplate="count=%{z}<extra></extra>",
                    colorscale="Purples",
                ),
                row=r, col=2
            )
            trace_counter += 1

        # -------------------------
        # col 3: annotation
        # -------------------------
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r, col=3
        )
        trace_counter += 1

        fig.add_annotation(
            x=-.2,
            y=0.5,
            xref="x domain",
            yref="y domain",
            text=stats_to_html(item),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=11),
            row=r,
            col=3,
        )

        fig.update_xaxes(visible=False, row=r, col=3, range=[0, 1])
        fig.update_yaxes(visible=False, row=r, col=3, range=[0, 1])

    # slider updates only the moving markers in col 1
    steps = []
    for s, zval in enumerate(zgrid):
        x_updates = []
        y_updates = []

        for y in y_series_list:
            x_updates.append([zval])
            y_updates.append([y[s]])

        steps.append(
            dict(
                method="restyle",
                args=[{"x": x_updates, "y": y_updates}, moving_trace_ids],
                label=f"{zval:.2f}",
            )
        )

    # hide most x tick labels except bottom row
    for r in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
        fig.update_xaxes(showticklabels=False, row=r, col=2)

    fig.update_xaxes(title_text=f"z[{zdim}]", row=n_rows, col=1)

    fig.update_layout(
        template="plotly_white",
        width=900,
        height=max(75 * n_rows, 500),
        title=f"{block} | Decoding from latent dim z[{zdim}] | KL[{kl[zdim]}]",
        margin=dict(t=80, l=70, r=20, b=70),
        sliders=[dict(
            active=init_idx,
            currentvalue={"prefix": f"z[{zdim}] = ", "font": {"size": 12}},
            pad={"t": 8},
            x=0.01,
            len=0.36,
            xanchor="left",
            y=-0.06,
            yanchor="top",
            steps=steps,
        )],
    )

    return fig, dec, recon_info


def summarize_reconstruction(recon_info, show=True):
    grouped = defaultdict(list)

    for item in recon_info:
        grouped[item["type"]].append(item)

    summary = {}

    for vtype, items in grouped.items():
        stats = {}

        if vtype == "numeric":
            stats["r"] = np.nanmean([i["stats"]["r"] for i in items])
            stats["rmse"] = np.nanmean([i["stats"]["rmse"] for i in items])
            stats["mae"] = np.nanmean([i["stats"]["mae"] for i in items])

        elif vtype == "binary":
            stats["acc"] = np.nanmean([i["stats"]["acc"] for i in items])
            stats["sens"] = np.nanmean([i["stats"]["sens"] for i in items])
            stats["spec"] = np.nanmean([i["stats"]["spec"] for i in items])

        elif vtype == "ordinal":
            stats["acc"] = np.nanmean([i["stats"]["acc"] for i in items])
            stats["mae_cat"] = np.nanmean([i["stats"]["mae_cat"] for i in items])
            stats["rank_r"] = np.nanmean([i["stats"]["rank_r"] for i in items])

        summary[vtype] = stats

    if show:
        print("\n=== Reconstruction Summary ===\n")

        for vtype, stats in summary.items():
            print(f"[{vtype.upper()}]")

            for k, v in stats.items():
                print(f"  {k:10s}: {v:.4f}")

            print()

    return summary


def compare_reconstructions_to_dataframe(recon_info_vae, recon_info_base):
    """
    Returns one row per variable with VAE metrics, baseline metrics, and deltas.
    """
    rows = []

    base_map = {item["var"]: item for item in recon_info_base}

    for item in recon_info_vae:
        var = item["var"]
        b = base_map[var]

        row = {
            "var": var,
            "type": item["type"],
            "long_name": item["long_name"],
            "n_obs": item["n_obs"],
        }

        if item["type"] == "numeric":
            row.update({
                "vae_r": item["stats"]["r"],
                "vae_rmse": item["stats"]["rmse"],
                "vae_mae": item["stats"]["mae"],
                "base_r": b["stats"]["r"],
                "base_rmse": b["stats"]["rmse"],
                "base_mae": b["stats"]["mae"],
                "delta_rmse": b["stats"]["rmse"] - item["stats"]["rmse"],  # >0 means VAE better
                "delta_mae": b["stats"]["mae"] - item["stats"]["mae"],     # >0 means VAE better
            })

        elif item["type"] == "binary":
            row.update({
                "vae_acc": item["stats"]["acc"],
                "vae_sens": item["stats"]["sens"],
                "vae_spec": item["stats"]["spec"],
                "base_acc": b["stats"]["acc"],
                "base_sens": b["stats"]["sens"],
                "base_spec": b["stats"]["spec"],
                "delta_acc": item["stats"]["acc"] - b["stats"]["acc"],     # >0 means VAE better
            })

        elif item["type"] == "ordinal":
            row.update({
                "vae_acc": item["stats"]["acc"],
                "vae_mae_cat": item["stats"]["mae_cat"],
                "vae_rank_r": item["stats"]["rank_r"],
                "base_acc": b["stats"]["acc"],
                "base_mae_cat": b["stats"]["mae_cat"],
                "base_rank_r": b["stats"]["rank_r"],
                "delta_acc": item["stats"]["acc"] - b["stats"]["acc"],         # >0 means VAE better
                "delta_mae_cat": b["stats"]["mae_cat"] - item["stats"]["mae_cat"],  # >0 means VAE better
            })

        rows.append(row)

    return pd.DataFrame(rows)


def run_encoder_shap(vae_model, X, M, fnames, zdims, block, beta, ikf, okf, vae_dir, v_tag,
                     res_per_dim, use_mask=False,n_bg=100,n_exp=500, seed=42, long_name_map=None,):
    """
    Explica mu = encoder(.) con SHAP GradientExplainer.

    Parámetros
    ----------
    X : np.ndarray, shape (N, D)
        Datos ya en el mismo espacio que ve el modelo.
    M : np.ndarray, shape (N, D)
        Máscara 1 observado / 0 missing.
    use_mask : bool
        True para modelos con encode(x, mask), False para encode(x).
    """

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(vae_dir, v_tag), exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)

    if long_name_map is None:
        long_name_map = {}

    pretty_fnames = [long_name_map.get(f, f) for f in fnames]

    n = X.shape[0]
    if n < 2:
        return res_per_dim

    n_bg = min(n_bg, n)
    n_exp = min(n_exp, n)

    idx_bg = rng.choice(n, size=n_bg, replace=False)
    idx_exp = rng.choice(n, size=n_exp, replace=False)

    if use_mask:
        # entrada explicada = [x, mask]
        XM_bg = np.concatenate([X[idx_bg], M[idx_bg]], axis=1).astype(np.float32)
        XM_exp = np.concatenate([X[idx_exp], M[idx_exp]], axis=1).astype(np.float32)

        f = VAE_encoder_wrapper_masked(vae_model, xdim=X.shape[1])
        bg_tensor = torch.tensor(XM_bg, dtype=torch.float32)
        exp_tensor = torch.tensor(XM_exp, dtype=torch.float32)


    else:
        f = VAE_encoder_wrapper(vae_model)
        bg_tensor = torch.tensor(X[idx_bg], dtype=torch.float32)
        exp_tensor = torch.tensor(X[idx_exp], dtype=torch.float32)

    explainer = shap.GradientExplainer(f, bg_tensor)
    shap_values = explainer(exp_tensor)

    # Normalizar salida a numpy
    shap_arr = shap_values.values if hasattr(shap_values, "values") else shap_values
    shap_arr = np.asarray(shap_arr)

    # Esperamos shape: [n_samples, n_features, zdims]
    if shap_arr.ndim != 3:
        raise ValueError(f"SHAP inesperado: shape={shap_arr.shape}, esperaba 3 dimensiones")

    data_for_plot = exp_tensor.detach().cpu().numpy()

    if use_mask:
        n_x = len(fnames)
        shap_arr_plot = shap_arr[:, :n_x, :]  # solo X
        data_for_plot = data_for_plot[:, :n_x]  # solo X
        feature_names_plot = pretty_fnames
    else:
        shap_arr_plot = shap_arr
        feature_names_plot = pretty_fnames

    for z in range(zdims):
        shap_exp = shap.Explanation(
            values=shap_arr_plot[:, :, z],
            data=data_for_plot,
            feature_names=feature_names_plot
        )

        shap.plots.beeswarm(shap_exp, max_display=min(40, shap_exp.shape[1]), show=False)
        plt.gcf().set_size_inches(13, 18)
        plt.gcf().subplots_adjust(left=0.45)
        plt.tight_layout()
        plt.title(f"Latent dim: {z}")
        plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__SHAP_beeswarm__z{z}.png"), dpi=200, bbox_inches="tight")
        plt.close()

        shap.plots.bar(shap_exp, max_display=min(40, shap_exp.shape[1]), show=False)
        plt.gcf().set_size_inches(15, 18)
        plt.gcf().subplots_adjust(left=0.45, top=0.95, bottom=0.05)
        plt.title(f"Latent dim: {z}")
        plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__SHAP_bars__z{z}.png"), dpi=200, bbox_inches="tight")
        plt.close()

        for x in range(len(fnames)):
            res_per_dim.append({
                "model": "mVAE",
                "block": block,
                "zdims": zdims,
                "beta": beta,
                "ikf": ikf,
                "okf": okf,
                "z": z,
                "x": x,
                "x_fname": fnames[x],
                "long_name": pretty_fnames[x],
                "var": "shap_abs",
                "val": float(np.mean(np.abs(shap_arr_plot[:, x, z])))
            })
            res_per_dim.append({
                "model": "mVAE",
                "block": block,
                "zdims": zdims,
                "beta": beta,
                "ikf": ikf,
                "okf": okf,
                "z": z,
                "x": x,
                "x_fname": fnames[x],
                "long_name": pretty_fnames[x],
                "var": "shap_signed",
                "val": float(np.mean(shap_arr_plot[:, x, z]))
            })
    return res_per_dim

# %% Assessment functions SDH

def assess_mVAE_sdh_performance(block, X, M, fnames, vae_model, zdims, beta, ikf, okf, vae_dir, v_tag,
                                X_train=None, M_train=None, vars_meta=None,
                                avoid=None, kl_th=0.01, tw_n=15, seed=42, verbose=False):

    tic0 = time.time()

    if avoid is None:
        avoid = []

    res = {"model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
           "kl_th": kl_th, "ikf": ikf, "okf": okf}
    res_per_dim = []

    X_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)

    vae_model.eval()
    with torch.no_grad():
        mu, logvar = vae_model.encode(X_t, M_t)
        out = vae_model.decode(mu)

    mu_perdim = mu.mean(dim=0).detach().cpu().numpy()
    logvar_perdim = logvar.mean(dim=0).detach().cpu().numpy()

    eps = 1e-8

    idx_num = list(getattr(vae_model, "numeric_idx", []))
    idx_bin = list(getattr(vae_model, "binary_idx", []))
    idx_ord = list(getattr(vae_model, "ordinal_idx", []))

    # --------------------------------------------------
    # reconstruction metrics in the same style as VAE_sdh_v4
    # --------------------------------------------------

    df_cmp = None

    if vars_meta is not None:
        recon_info = build_reconstruction_info(
            model=vae_model,
            x_eval=X,
            m_eval=M,
            fnames=fnames,
            idx_num=idx_num,
            idx_bin=idx_bin,
            idx_ord=idx_ord,
            vars_meta=vars_meta,
        )

        # aggregate summaries
        num_items = [it for it in recon_info if it["type"] == "numeric"]
        bin_items = [it for it in recon_info if it["type"] == "binary"]
        ord_items = [it for it in recon_info if it["type"] == "ordinal"]

        if len(num_items) > 0:
            res.update({
                "num_r": float(np.nanmean([it["stats"]["r"] for it in num_items])),
                "num_rmse": float(np.nanmean([it["stats"]["rmse"] for it in num_items])),
                "num_mae": float(np.nanmean([it["stats"]["mae"] for it in num_items])),
            })

        if len(bin_items) > 0:
            res.update({
                "bin_acc": float(np.nanmean([it["stats"]["acc"] for it in bin_items])),
                "bin_sens": float(np.nanmean([it["stats"]["sens"] for it in bin_items])),
                "bin_spec": float(np.nanmean([it["stats"]["spec"] for it in bin_items])),
            })

        if len(ord_items) > 0:
            res.update({
                "ord_acc": float(np.nanmean([it["stats"]["acc"] for it in ord_items])),
                "ord_mae_cat": float(np.nanmean([it["stats"]["mae_cat"] for it in ord_items])),
                "ord_rank_r": float(np.nanmean([it["stats"]["rank_r"] for it in ord_items])),
            })

        # optional baseline comparison and csv
        if X_train is not None and M_train is not None:
            baseline_constants = build_baseline_constants(
                X_train_evalspace=X_train,
                M_train=M_train,
                fnames=fnames,
                idx_num=idx_num,
                idx_bin=idx_bin,
                idx_ord=idx_ord,
                vars_meta=vars_meta,
            )

            base_info = build_baseline_reconstruction_info(
                baseline_constants=baseline_constants,
                x_eval=X,
                m_eval=M,
                fnames=fnames,
                idx_num=idx_num,
                idx_bin=idx_bin,
                idx_ord=idx_ord,
                vars_meta=vars_meta,
            )

            df_cmp = compare_reconstructions_to_dataframe(recon_info, base_info)

            os.makedirs(os.path.join(vae_dir, v_tag), exist_ok=True)
            df_cmp.to_csv(os.path.join(vae_dir, v_tag, f"compare_vae_baseline__{block}.csv"),index=False)

            # aggregate deltas
            for vtype, df_sub in df_cmp.groupby("type"):
                if vtype == "numeric":
                    res.update({
                        "delta_num_rmse": float(df_sub["delta_rmse"].mean()),
                        "delta_num_mae": float(df_sub["delta_mae"].mean()),
                    })
                elif vtype == "binary":
                    res.update({
                        "delta_bin_acc": float(df_sub["delta_acc"].mean()),
                    })
                elif vtype == "ordinal":
                    res.update({
                        "delta_ord_acc": float(df_sub["delta_acc"].mean()),
                        "delta_ord_mae_cat": float(df_sub["delta_mae_cat"].mean()),
                    })

        # per-variable results into res_per_dim
        for item in recon_info:
            row = {
                "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                "ikf": ikf, "okf": okf,
                "x": item["gidx"], "x_fname": item["var"],
                "type": item["type"], "long_name": item["long_name"], "n_obs": item["n_obs"],
            }

            if item["type"] == "numeric":
                res_per_dim.extend([
                    {**row, "var": "r", "val": float(item["stats"]["r"])},
                    {**row, "var": "rmse", "val": float(item["stats"]["rmse"])},
                    {**row, "var": "mae", "val": float(item["stats"]["mae"])},
                ])

            elif item["type"] == "binary":
                res_per_dim.extend([
                    {**row, "var": "acc", "val": float(item["stats"]["acc"])},
                    {**row, "var": "sens", "val": float(item["stats"]["sens"])},
                    {**row, "var": "spec", "val": float(item["stats"]["spec"])},
                ])

            elif item["type"] == "ordinal":
                res_per_dim.extend([
                    {**row, "var": "acc", "val": float(item["stats"]["acc"])},
                    {**row, "var": "mae_cat", "val": float(item["stats"]["mae_cat"])},
                    {**row, "var": "rank_r", "val": float(item["stats"]["rank_r"])},
                ])

        if df_cmp is not None:
            for _, rr in df_cmp.iterrows():
                base_row = {
                    "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf,
                    "x_fname": rr["var"], "type": rr["type"], "long_name": rr["long_name"], "n_obs": rr["n_obs"]
                }

                for col in rr.index:
                    if col in ["var", "type", "long_name", "n_obs"]:
                        continue
                    if pd.notna(rr[col]):
                        res_per_dim.append({**base_row, "var": col, "val": float(rr[col])})


    losses_by_type = {}
    obs_by_type = {}
    per_xdim_mse = np.full(X.shape[1], np.nan, dtype=float)
    per_xdim_mae = np.full(X.shape[1], np.nan, dtype=float)

    # ---------- numeric ----------
    if len(idx_num) > 0 and out["num_mu"] is not None:
        x_num = X_t[:, idx_num]
        m_num = M_t[:, idx_num]
        mu_num = out["num_mu"]

        se_num = (mu_num - x_num).pow(2)
        ae_num = (mu_num - x_num).abs()

        den = m_num.sum().clamp_min(eps)
        mse_num = (se_num * m_num).sum() / den
        mae_num = (ae_num * m_num).sum() / den

        losses_by_type["mse_num"] = float(mse_num.detach().cpu())
        losses_by_type["mae_num"] = float(mae_num.detach().cpu())
        obs_by_type["nobs_num"] = float(m_num.sum().detach().cpu())

        obs_per_x = m_num.sum(dim=0).clamp_min(1.0)
        mse_per_x = ((se_num * m_num).sum(dim=0) / obs_per_x).detach().cpu().numpy()
        mae_per_x = ((ae_num * m_num).sum(dim=0) / obs_per_x).detach().cpu().numpy()

        for j, x_idx in enumerate(idx_num):
            per_xdim_mse[x_idx] = mse_per_x[j]
            per_xdim_mae[x_idx] = mae_per_x[j]

    # ---------- binary ----------
    if len(idx_bin) > 0 and out["bin"] is not None:
        x_bin = X_t[:, idx_bin]
        m_bin = M_t[:, idx_bin]
        logits_bin = out["bin"]
        prob_bin = torch.sigmoid(logits_bin)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_bin, x_bin, reduction="none"
        )
        ae_bin = (prob_bin - x_bin).abs()

        den = m_bin.sum().clamp_min(eps)
        bce_bin = (bce * m_bin).sum() / den
        mae_bin = (ae_bin * m_bin).sum() / den

        losses_by_type["bce_bin"] = float(bce_bin.detach().cpu())
        losses_by_type["mae_bin"] = float(mae_bin.detach().cpu())
        obs_by_type["nobs_bin"] = float(m_bin.sum().detach().cpu())

        obs_per_x = m_bin.sum(dim=0).clamp_min(1.0)
        bce_per_x = ((bce * m_bin).sum(dim=0) / obs_per_x).detach().cpu().numpy()
        mae_per_x = ((ae_bin * m_bin).sum(dim=0) / obs_per_x).detach().cpu().numpy()

        for j, x_idx in enumerate(idx_bin):
            per_xdim_mse[x_idx] = bce_per_x[j]   # aquí guardas BCE, no MSE
            per_xdim_mae[x_idx] = mae_per_x[j]

    # ---------- ordinal ----------
    if len(idx_ord) > 0 and len(out["ord"]) > 0:
        x_ord = X_t[:, idx_ord].long()
        m_ord = M_t[:, idx_ord]

        ord_num = torch.tensor(0.0, dtype=torch.float32, device=X_t.device)
        ord_den = torch.tensor(0.0, dtype=torch.float32, device=X_t.device)

        ord_nll_per_x = []
        ord_mae_per_x = []

        for k, x_idx in enumerate(idx_ord):
            eta_k, tau_k = out["ord"][k]  # eta: (B,), tau: (K-1,)
            target_k = x_ord[:, k]
            mask_k = m_ord[:, k]

            # same family as training loss
            nll_k = ordinal_logistic_nll_eval(eta_k, tau_k, target_k)

            # convert ordinal head output to class probabilities for prediction
            probs_k = ordinal_probs_from_eta_tau_torch(eta_k, tau_k)  # (B, K)
            pred_k = torch.argmax(probs_k, dim=1)

            ae_k = (pred_k.float() - target_k.float()).abs()

            ord_num += (nll_k * mask_k).sum()
            ord_den += mask_k.sum()

            den_k = mask_k.sum().clamp_min(1.0)
            ord_nll_per_x.append(float(((nll_k * mask_k).sum() / den_k).detach().cpu()))
            ord_mae_per_x.append(float(((ae_k * mask_k).sum() / den_k).detach().cpu()))

        ord_nll = ord_num / ord_den.clamp_min(eps)

        losses_by_type["nll_ord"] = float(ord_nll.detach().cpu())
        losses_by_type["mae_ord"] = float(np.nanmean(ord_mae_per_x))
        obs_by_type["nobs_ord"] = float(m_ord.sum().detach().cpu())

        for j, x_idx in enumerate(idx_ord):
            per_xdim_mse[x_idx] = ord_nll_per_x[j]  # keep slot name if you want backward compatibility
            per_xdim_mae[x_idx] = ord_mae_per_x[j]


    # ---------- global observed loss ----------
    total_obs = float(M_t.sum().detach().cpu())
    res.update(obs_by_type)
    res.update(losses_by_type)
    res["nobs_total"] = total_obs

    # un agregado simple, explícitamente heterogéneo
    available_main_losses = [v for k, v in losses_by_type.items() if k in ["gnll_num", "mse_num", "bce_bin", "nll_ord"]]
    res["recon_main_mean"] = float(np.mean(available_main_losses)) if len(available_main_losses) > 0 else np.nan

    # ---------- KL ----------
    kl_per_dim_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl_per_sample = kl_per_dim_sample.sum(dim=1)
    kl = kl_per_sample.mean()
    kl_per_dim = kl_per_dim_sample.mean(dim=0).detach().cpu().numpy()

    active_dims = int(np.sum(kl_per_dim > kl_th))

    res.update({
        "kl": float(kl.detach().cpu()),
        "active_dims": active_dims,
        "kl_mean_per_dim": float(np.mean(kl_per_dim)),
        "mu_abs_mean": float(np.mean(np.abs(mu_perdim))),
        "logvar_mean": float(np.mean(logvar_perdim)),
    })

    if "per_dim" not in avoid:
        for z in range(zdims):
            res_per_dim.append({
                "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                "ikf": ikf, "okf": okf, "z": z, "var": "kl", "val": float(kl_per_dim[z])
            })

    # ---------- Latent arrays ----------
    Z_mu = mu.detach().cpu().numpy()

    if "decoder_grid" not in avoid:

        if vars_meta is None:
            raise ValueError("vars_meta is required to build decoder grids.")
        if X_train is None or M_train is None:
            raise ValueError("X_train_ref and M_train_ref are required to build decoder grids.")

        tic_ass = time.time()

        z_ref = get_latent_reference(
            model=vae_model,
            x_ref=X_train,
            m_ref=M_train
        )

        out_dir = os.path.join(vae_dir, v_tag)
        os.makedirs(out_dir, exist_ok=True)

        idx_num = list(getattr(vae_model, "numeric_idx", []))
        idx_bin = list(getattr(vae_model, "binary_idx", []))
        idx_ord = list(getattr(vae_model, "ordinal_idx", []))

        for zid in range(zdims):
            fig_dec, dec, recon_info = make_decoder_grid(
                model=vae_model,
                block=block,
                fnames=fnames,
                idx_num=idx_num,
                idx_bin=idx_bin,
                idx_ord=idx_ord,
                vars_meta=vars_meta,
                x_eval=X,
                m_eval=M,
                zdim=zid,
                kl=kl_per_dim,
                z_ref=z_ref,
                grid=np.linspace(-3, 3, 41),
                max_vars=40,
            )

            pio.write_html(
                fig_dec,
                os.path.join(out_dir, f"decoded_{block}_z{zid}.html"),
                auto_open=False
            )
            pio.write_image(
                fig_dec,
                os.path.join(out_dir, f"decoded_{block}_z{zid}.png")
            )

        if verbose:
            print(f"\tDecoder grid ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    if "geometry" not in avoid:
        tic_ass = time.time()

        id_twonn = intrinsic_dim_twonn(Z_mu, seed=seed)
        tw = sklearn.manifold.trustworthiness(X, Z_mu, n_neighbors=tw_n)

        res.update({"intrinsic_dim_twonn": id_twonn, "tw_n": tw_n, "trustworthyness": tw})

        if verbose:
            print(f"\tGeometric evaluation ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    if "corrs" not in avoid:
        tic_ass = time.time()

        r_z = np.ones((1, 1), dtype=float) if zdims == 1 else np.corrcoef(Z_mu, rowvar=False)
        r_z_triu = r_z[np.triu_indices(zdims, k=1)] if zdims > 1 else np.array([0.0])
        res.update({"r_z": float(np.nanmean(np.abs(r_z_triu)))})

        if "per_dim" not in avoid:
            for z1 in range(zdims):
                for z2 in range(z1):
                    res_per_dim.append({
                        "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                        "ikf": ikf, "okf": okf, "z": z1, "z2": z2, "var": "r_z", "val": float(r_z[z1, z2])
                    })

        obs_row = M.mean(axis=1)
        r_zM_per_dim = [
            np.corrcoef(Z_mu[:, z], obs_row, rowvar=False)[0, 1] if np.std(obs_row) != 0 else np.nan
            for z in range(zdims)
        ]
        res.update({"r_zM": float(np.nanmean(r_zM_per_dim))})

        if "per_dim" not in avoid:
            for z in range(zdims):
                res_per_dim.append({
                    "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf, "z": z, "var": "r_zM", "val": float(r_zM_per_dim[z])
                })

        # correlaciones z-X: aquí puedes mantenerlo como estaba
        r_zX = np.corrcoef(Z_mu, X, rowvar=False)[zdims:, :zdims]

        if "per_dim" not in avoid:
            for z in range(zdims):
                for x in range(X.shape[1]):
                    res_per_dim.append({
                        "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                        "ikf": ikf, "okf": okf, "z": z, "x": x, "x_fname": fnames[x],
                        "var": "r_zX", "val": float(r_zX[x, z])
                    })

        eps_h = 1e-12
        p_per_dim = []
        for z in range(zdims):
            a = np.abs(r_zX[:, z])
            s = a.sum()
            if s <= eps_h:
                p = np.full_like(a, np.nan, dtype=float)
            else:
                p = a / s
            p_per_dim.append(p)

        en_per_dim = [
            1.0 / np.nansum(p ** 2) if np.all(np.isfinite(p)) else np.nan
            for p in p_per_dim
        ]
        res.update({"EN": float(np.nanmean(en_per_dim))})

        if "per_dim" not in avoid:
            for z in range(zdims):
                res_per_dim.append({
                    "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf, "z": z, "var": "EN", "val": float(en_per_dim[z])
                })

        if verbose:
            print(f"\tLatent correlations ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    # ---------- per feature metrics ----------
    if "per_dim" not in avoid:
        for x_idx, x_name in enumerate(fnames):
            if np.isfinite(per_xdim_mse[x_idx]):
                res_per_dim.append({
                    "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf, "x": x_idx, "x_fname": x_name,
                    "var": "recon_x_main", "val": float(per_xdim_mse[x_idx])
                })
            if np.isfinite(per_xdim_mae[x_idx]):
                res_per_dim.append({
                    "model": "mVAE", "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf, "x": x_idx, "x_fname": x_name,
                    "var": "recon_x_mae", "val": float(per_xdim_mae[x_idx])
                })


    # %% 4.7) SHAP per latent dimension
    if "shap" not in avoid:
        tic_ass = time.time()

        long_name_map = None
        if vars_meta is not None:
            # ajusta estos nombres de columnas si en tu df se llaman distinto
            long_name_map = (
                vars_meta.drop_duplicates("name")
                .set_index("name")["long_name"]
                .to_dict()
            )

        res_per_dim = run_encoder_shap(
            vae_model=vae_model, X=X, M=M, fnames=fnames, zdims=zdims, block=block, beta=beta,
            ikf=ikf, okf=okf, vae_dir=vae_dir, v_tag=v_tag, res_per_dim=res_per_dim,
            use_mask=True, n_bg=100, n_exp=300, seed=seed, long_name_map=long_name_map
        )

        if verbose:
            print(f"\tSHAP ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

    if verbose:
        print(f"  Total time ({round((time.time() - tic0), 2)}sec) #", end="\n")

    return Z_mu, pd.DataFrame([res]), pd.DataFrame(res_per_dim)


def assess_downstream_sdh(model, data, exp_vars, sim_vars, scaler, block, vars_meta, zdims, beta, ikf, okf,
                      covars=None, test_size=0.2, n_splits=1, seed=0, order_by="t", verbose=False):
    """
    Devuelve:
      - res: resultados globales (baseline/full) evaluados en TEST
      - res_per_dim: coeficientes del FULL (fit en TRAIN)
      - res_curve: curva incremental R2_test(m) y ΔR2_test(m)

    :param data: data from simulations
    :param exp_vars: Exposome variables in the block to encode.
    :param sim_vars: Simulation outcomes to target in GAM
        vars_down:  downstream associations options -
        # out: ['ent_E', 'ent_I',  'rate_E', 'rate_I',  'target', 'EI_ent', 'EI_rate']
        # net: ['DMN', 'DAN', 'VAN', 'SMN', 'VIS', 'LIM', 'FPN'];
        # roi: range(90) - AAL regions.

    :param vae_model: used to ENCODE data into latent
    :param scaler:
    :param block:
    :param zdims:
    :param beta:
    :param ikf:
    :param okf:
    :param gam_type: ["linear", "splines"]
    :param verbose:
    :return:
    """

    covars = covars or []

    tic = time.time()

    # Define var types
    meta = vars_meta.set_index("name")

    idx_num = [i for i, var in enumerate(exp_vars) if meta.at[var, "type"] == "numeric"]
    idx_ord = [i for i, var in enumerate(exp_vars) if meta.at[var, "type"] == "ordinal"]
    idx_bin = [i for i, var in enumerate(exp_vars) if meta.at[var, "type"] == "binary"]
    ord_ncat = [int(meta.at[var, "ord_ncat"]) for var in exp_vars if meta.at[var, "type"] == "ordinal"]

    df = data[covars + exp_vars + sim_vars].copy() # Omit 1 subject with Age=None

    # 1) Exposome -> Latents -  Prepare the data
    X = df.loc[:, exp_vars].values

    X_sc_fill = X.copy()
    if len(idx_num) > 0:
        X_sc_fill[:, idx_num] = scaler.transform(X[:, idx_num])
    X_sc_fill = np.nan_to_num(X_sc_fill, nan=0.0)

    M = (~np.isnan(X)).astype(np.float32)

    if model is None:
        model_name = "FEAT"
        Z = X_sc_fill

    elif model == "Z_vae":
        model_name = model
        Z = X_sc_fill

    elif hasattr(model, "encode"):
        model_name = "mVAE"
        # Use the model to encode the Exposome: latent
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(torch.tensor(X_sc_fill, dtype=torch.float32),
                                      torch.tensor(M, dtype=torch.float32))
        Z = mu.detach().cpu().numpy()  # (n, zdim)

    elif hasattr(model, "transform"):
        model_name = "PCA"
        Z = model.transform(X_sc_fill)

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]

    # DataFrame de latentes
    zcols = [f"b{block[0]}_z{i}" for i in range(Z.shape[1])]
    dfZ = pd.DataFrame(Z, columns=zcols, index=df.index)


    # --- 2) Split 80/20 (repetible) ---
    splitter = sklearn.model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    res, res_per_dim, res_curve = [], [], []

    for split_id, (idx_tr, idx_te) in enumerate(splitter.split(df)):
        pass

        # Prepara matrices de covariables
        if covars:
            Xcov_tr = df.iloc[idx_tr][covars].copy()
            Xcov_te = df.iloc[idx_te][covars].copy()
            # Si hay categóricas (country), conviértelas a dummies aquí:
            Xcov_tr = pd.get_dummies(Xcov_tr, drop_first=True)
            Xcov_te = pd.get_dummies(Xcov_te, drop_first=True)
            # alinear columnas train/test
            Xcov_te = Xcov_te.reindex(columns=Xcov_tr.columns, fill_value=0.0)
        else:
            Xcov_tr = pd.DataFrame(index=df.iloc[idx_tr].index)
            Xcov_te = pd.DataFrame(index=df.iloc[idx_te].index)

        # latentes train/test
        Z_tr = dfZ.iloc[idx_tr].copy()
        Z_te = dfZ.iloc[idx_te].copy()

        # Design matrices (baseline y full)
        X_base_tr = sm.add_constant(Xcov_tr, has_constant="add").astype(float)
        X_base_te = sm.add_constant(Xcov_te, has_constant="add").astype(float)

        X_full_tr = sm.add_constant(pd.concat([Xcov_tr, Z_tr], axis=1), has_constant="add").astype(float)
        X_full_te = sm.add_constant(pd.concat([Xcov_te, Z_te], axis=1), has_constant="add").astype(float)

        for sv, sim_var in enumerate(sim_vars):
            pass

            # 2) target - Prepare simulated data
            y_tr = df.iloc[idx_tr][sim_var].values
            y_te = df.iloc[idx_te][sim_var].values


            # --- Baseline ---
            ols_base = sm.OLS(y_tr, X_base_tr).fit()
            yhat_base_te = ols_base.predict(X_base_te)
            r2_base_te = sklearn.metrics.r2_score(y_te, yhat_base_te)

            # --- Full ---
            ols_full = sm.OLS(y_tr, X_full_tr).fit()
            yhat_full_te = ols_full.predict(X_full_te)
            r2_full_te = sklearn.metrics.r2_score(y_te, yhat_full_te)

            f2_test = (r2_full_te - r2_base_te) / max(1 - r2_full_te, 1e-12)

            # --- modelo global ---
            res.append({
                "model":model_name, "block": block, "zdims": zdims, "beta": beta, "ikf": ikf, "okf": okf,
                "split": split_id, "sim_var": sim_var, "n_covars": len(covars), "n_pred": len(zcols),
                # ---- Test metrics (real performance) ----
                "r2_base_test": r2_base_te, "r2_full_test": r2_full_te, "delta_r2_test": (r2_full_te - r2_base_te),
                "f2_test": f2_test,
                # ---- Train diagnostics ----
                "r2_train": ols_full.rsquared, "r2_adj_train": ols_full.rsquared_adj, "aic_train": ols_full.aic,
                "bic_train": ols_full.bic, "f_pvalue_train": ols_full.f_pvalue,

                "n_train":len(idx_tr), "n_test":len(idx_te),
            })

            # Coefs FULL (en TRAIN): útiles para ranking/interpretación
            for term in ols_full.params.index:
                res_per_dim.append({
                    "model": model_name,
                    "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf,
                    "split": split_id,
                    "sim_var": sim_var, "n_covars": len(covars), "n_pred": len(zcols),
                    "term": term,
                    "coef": float(ols_full.params[term]),
                    "se": float(ols_full.bse[term]),
                    "t": float(ols_full.tvalues[term]),
                    "pval": float(ols_full.pvalues[term]),
                })


            # --- 3) Curva incremental (orden rápido sin LASSO) ---
            # Orden SOLO con TRAIN (para no mirar el test)
            if order_by == "t":
                # ordenar solo latentes (z*) por |t|
                tvals = ols_full.tvalues.reindex(zcols)
                order = tvals.abs().sort_values(ascending=False).index.tolist()
            elif order_by == "absbeta":
                betas = ols_full.params.reindex(zcols)
                order = betas.abs().sort_values(ascending=False).index.tolist()
            else:
                raise ValueError("order_by debe ser 't' o 'absbeta'.")


            # Construye curva evaluada en TEST
            for m in range(1, min(len(order) + 1, zdims + 1)):
                topm = order[:m]

                X_m_tr = sm.add_constant(pd.concat([Xcov_tr, Z_tr[topm]], axis=1), has_constant="add").astype(float)
                X_m_te = sm.add_constant(pd.concat([Xcov_te, Z_te[topm]], axis=1), has_constant="add").astype(float)

                ols_m = sm.OLS(y_tr, X_m_tr).fit()
                yhat_m_te = ols_m.predict(X_m_te)
                r2_m_te = sklearn.metrics.r2_score(y_te, yhat_m_te)

                res_curve.append({
                    "model": model_name,
                    "block": block, "zdims": zdims, "beta": beta,
                    "ikf": ikf, "okf": okf,
                    "split": split_id,
                    "sim_var": sim_var, "n_covars": len(covars), "n_pred": len(zcols),
                    "m": m,
                    "r2_test": r2_m_te,
                    "delta_r2_test": (r2_m_te - r2_base_te),
                    "order_by": order_by,
                })

            if verbose:
                print(f"[Downstream] split{split_id+1}/{n_splits} . {sim_var} ({sv}/{len(sim_vars)})", end="\r")

    if verbose:
        print(f"\tDownstream associations ({round((time.time() - tic), 2)}sec) . ")



    return Z, pd.DataFrame(res), pd.DataFrame(res_per_dim), pd.DataFrame(res_curve)



def ordinal_logistic_nll_eval(eta, tau, target):
    """
    Same likelihood used by the model, but local to assess.py
    target must be integer labels 0..K-1
    """
    probs = ordinal_probs_from_eta_tau_torch(eta, tau)
    logp = torch.log(probs)
    return -logp.gather(1, target.unsqueeze(1)).squeeze(1)



# %% PCA as a linear baseline model

def assess_PCA_benchmarck(block, X_train, X_val, fnames,  zdims, beta, ikf, okf, vae_dir, v_tag,
                            avoid=None, seed=42, verbose=False, plot=False):

        tic_ass = time.time()

        res = {"model":"PCA", "block": block, "zdims": zdims, "beta": beta, "ikf": ikf, "okf": okf}
        res_per_dim = []

        pca = sklearn.decomposition.PCA(n_components=zdims, random_state=seed)
        pca.fit(X_train)

        Z_pca = pca.transform(X_val)
        Xhat_pca = pca.inverse_transform(Z_pca)

        # Reconstruction error for PCA
        se = (Xhat_pca - X_val) ** 2
        se_per_xdim = se.mean(axis=0)
        mse = se.mean()

        sst = ((X_val - X_val.mean(axis=0, keepdims=True)) ** 2).mean()  # o sum/sum
        r2 = 1.0 - mse / sst

        res.update({"mse": mse, "sst": sst, "r2": r2})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"PCA", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "x": x, "x_fname":fnames[x],
                                 f"var": "se_pca", "val":se_per_xdim[x]} for x in range(X_val.shape[1])])

        # 1,Latent Dimensions correlation
        r_z = np.ones((1, 1), dtype=float) if zdims == 1 else np.corrcoef(Z_pca, rowvar=False)
        # plt.imshow(r_z, aspect="auto"); plt.colorbar(); plt.title("VAE - r(latent dims)")
        # plt.savefig(os.path.join(vae_dir, v_tag, f"VAE__rLatent.png")); plt.close()
        r_z_triu = r_z[np.triu_indices(zdims, k=1)] if zdims > 1 else 0
        res.update({"r_z": np.average(np.abs(r_z_triu))})
        if "per_dim" not in avoid:
            res_per_dim.extend([{"model":"PCA", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z1, "z2": z2,
                                 f"var": "r_z", "val":r_z[z1, z2]} for z1 in range(zdims) for z2 in range(zdims) if z1>z2])

        # 3, Variables Loadings on Latent (thorugh correlations)
        r_zX = np.corrcoef(Z_pca, X_val, rowvar=False)[zdims:, :zdims] # (X in rows, z in cols)
        # np concatenates the arrays by shared dim, and corrs all columns.
        res_per_dim.extend([{"model":"PCA", "block": block, "zdims": zdims, "beta": beta, "ikf":ikf, "okf":okf, "z": z, "x": x, "x_fname":fnames[x],
                             f"var": "r_zX", "val":r_zX[x, z]} for z in range(zdims) for x in range(X_val.shape[1])])



        if plot:
            # Plot PCA VARIANCE EXPLAINED
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
                     pca.explained_variance_ratio_, marker='o')
            plt.xlabel("Principal Component"); plt.ylabel("Explained variance ratio")
            plt.title("PCA variance explained plot"); plt.tight_layout();  # plt.show()
            plt.savefig(os.path.join(vae_dir, v_tag, f"PCA__variance_explained.png"))
            plt.close()

            # Plot PCA LOADINGS
            loadings = pd.DataFrame(pca.components_.T, index=fnames, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
            plt.figure(figsize=(10, 14))
            sns.heatmap(loadings, cmap="coolwarm", center=0, cbar_kws={"aspect": 50, "shrink": 0.85})
            plt.title("PCA loadings (variables × PCs)"); plt.tight_layout();  # plt.show()
            plt.savefig(os.path.join(vae_dir, v_tag, f"PCA__loadings.png"))
            plt.close()

        if verbose:
            print(f"\tPCA ({round((time.time() - tic_ass), 2)}sec) >> ", end="")

        return Z_pca, pd.DataFrame([res]), pd.DataFrame(res_per_dim), pca



# %% Aux functions for assessment

def intrinsic_dim_twonn(Z: np.ndarray, seed: int = 0) -> float:
    """
    TwoNN intrinsic dimension estimator (Facco et al.).
    Z: (n_samples, n_features)
    Devuelve un escalar (ID global).
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=float)

    n = Z.shape[0]
    if n < 5:
        return np.nan

    # Distancias euclídeas completas (n pequeño en tu caso, suele ser OK).
    # Para n grande usarías vecinos aproximados.
    # Calculamos para cada punto: r1 (1er vecino), r2 (2º vecino), ratio = r2/r1
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)  # (n,n)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)

    # ordena distancias por fila y toma dos primeras
    d_sorted = np.sort(d, axis=1)
    r1 = d_sorted[:, 0]
    r2 = d_sorted[:, 1]

    # evita divisiones raras
    eps = 1e-12
    mu_ratio = (r2 + eps) / (r1 + eps)

    # Ajuste lineal: F(log mu) ~ 1 - exp(-d * log mu)
    # Forma práctica TwoNN: y = -log(1 - F), x = log(mu), slope = d
    x = np.log(mu_ratio)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 5:
        return np.nan

    x_sorted = np.sort(x)
    F = (np.arange(1, x_sorted.size + 1) - 0.5) / x_sorted.size
    y = -np.log(1.0 - F)

    # slope through origin? normalmente con intercept.
    slope, intercept = np.polyfit(x_sorted, y, 1)
    return float(slope)


def simil_matrix_loadings(LA, LB, metric="corr", eps=1e-12):
    """
    LA, LB: (Z, X) loadings for two reps (latent dims Z x X vars), already abs.
    Returns S: (Z, Z) similarity between dims based on loading vectors over X.
    """
    # Compare vectors over X: treat each dim as a vector of length P
    # So build matrices of shape (X, Z)
    A = LA.T.astype(float)  # (X, Z)
    B = LB.T.astype(float)

    if metric == "corr":

        # fill NaNs with column means (or 0)
        A = np.where(np.isnan(A), np.nanmean(A, axis=0, keepdims=True), A)
        B = np.where(np.isnan(B), np.nanmean(B, axis=0, keepdims=True), B)

        A = np.where(np.isnan(A), 0.0, A)
        B = np.where(np.isnan(B), 0.0, B)

        # center + scale columns
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)

        A = A / (A.std(axis=0, keepdims=True) + eps)
        B = B / (B.std(axis=0, keepdims=True) + eps)

        # corr between columns -> (K,K)
        S = (A.T @ B) / max(A.shape[0], 1) # If A and B are standardized,
        # the dot matrix/N is exactly the matrix of corrs between columns

        return np.abs(S)  # abs for sign invariance (you already used abs(L) anyway)

    if metric == "cosine":
        # cosine similarity between columns
        A2 = np.where(np.isnan(A), 0.0, A)
        B2 = np.where(np.isnan(B), 0.0, B)
        An = A2 / (np.linalg.norm(A2, axis=0, keepdims=True) + 1e-12)
        Bn = B2 / (np.linalg.norm(B2, axis=0, keepdims=True) + 1e-12)
        S = np.abs(An.T @ Bn)
        return S

    raise ValueError(f"Unknown metric: {metric}")


def topk_jaccard_per_dim(LA, LB_aligned, topk=3):
    """
    LA: (K,P) rep A
    LB_aligned: (K,P) rep B, dims already matched to A
    Returns: (K,) Jaccard per dim based on topk variables (over P)
    """
    K, P = LA.shape
    j = np.zeros(K, dtype=float)
    for d in range(K):
        a = LA[d, :]
        b = LB_aligned[d, :]
        # indices topk
        k = min(int(topk), P)
        ia = np.argpartition(a, -k)[-k:]
        ib = np.argpartition(b, -k)[-k:]
        inter = len(set(ia).intersection(set(ib)))
        union = len(set(ia).union(set(ib)))
        j[d] = inter / union if union else np.nan
    return j