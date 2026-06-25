
import os
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import seaborn as sns

import nibabel as nib
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import mni152_to_fslr
from surfplot import Plot
from nilearn import surface

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# %% Visualizations
def plot_inner_folding_effects(block, scores, vae_dir, var="mse"):

    df = scores.loc[scores["block"]==block]
    df["ikf"] = df["ikf"].astype(str)

    # 1. Check CV folding effect - inner CV symm on var
    fig = px.strip(df, x="zdims", y=var, facet_col="beta", facet_row="okf", color="ikf",
                   title=f"Inner CV (folding asymmetry) on{var.upper()}", labels={"color": "Inner k-fold"})
    fig.update_layout(template="plotly_white")
    pio.write_image(fig, os.path.join(vae_dir, f"kf_asymm_on{var.upper()}.png"), width=800, height=500, scale=3)


def plot_assessment(blocks, df_scores, df_stab=None, df_down=None, select_vars=None, select_down=None, dir="", title="", params=None):
    """

    :param df_scores:
    :param df_down:
    :param df_stab:
    :param block:
    :param dir:
    :return:
    """

    # Determine number of plot columns based on dfs
    sc_vars = [var for var in select_vars if var in df_scores.columns]
    stab_vars = [var for var in select_vars if var in df_stab.columns] if df_stab is not None else []
    down_vars = [var for var in select_vars if var in df_down.columns] if df_down is not None else []

    down_outs = len(select_down) if select_down is not None else 0

    nrows = len(sc_vars + stab_vars + down_vars * down_outs)
    ncols = len(blocks)

    titles_map = {"mse": "MSE", "kl": "KL", "kl_act_dims": "Active<br>dims",
                  "r_z": "r(Lats)<br>redund.", "r_zM": "r(Lat-Miss)", "EN_vars": "Eff. Num.<br>vars",
                  "simil_mean": "Stability<br>r(Lats)",
                  "pseudo_r2": "pR\u00B2", "loss": "Loss"}

    cmap = px.colors.qualitative.Plotly
    jitter, size = (0.1, 4) if params is None else (params["jitter"], params["size"]) # adjust scale

    # Figure
    fig = make_subplots(rows=nrows, cols=ncols, column_titles=blocks,
                        shared_xaxes=True, shared_yaxes=True, x_title="zdims")

    for b, block in enumerate(blocks):
        col = b + 1
        # Plot scores
        for sc, sc_var in enumerate(sc_vars):
            row = sc + 1
            if col == 1:
                tit = titles_map[sc_var] if sc_var in titles_map.keys() else sc_var
                fig.update_yaxes(title_text=tit, row=row, col=1)
            sl = True if sc + b == 0 else False
            for c, beta in enumerate(sorted(set(df_scores["beta"]))):
                temp = df_scores.loc[(df_scores["block"]==block) & (df_scores["beta"]==beta)]
                zdims_j = temp["zdims"] + np.random.uniform(-jitter, jitter, size=len(temp["zdims"]))
                fig.add_trace(go.Scatter(x=zdims_j, y=temp[sc_var],
                                         name=f"\u03B2 {beta}", legendgroup=f"\u03B2 {beta}", showlegend=sl,
                                         mode="markers", marker=dict(color=cmap[c], size=size)), row=row, col=col)

            if sc_var == "mse" and "mse_pca" in temp.columns:
                fig.add_trace(go.Scatter(x=zdims_j, y=temp["mse_pca"],
                                         name=f"PCA ref", legendgroup=f"PCA ref", showlegend=sl,
                                         mode="markers", marker=dict(symbol="triangle-up", color="dimgray", size=size+2)),
                              row=row, col=col)


        # Plot stab
        for st, st_var in enumerate(stab_vars):
            row = len(sc_vars) + 1
            if col == 1:
                tit = titles_map[st_var] if st_var in titles_map.keys() else st_var
                fig.update_yaxes(title_text=tit, row=row, col=1)
            for c, beta in enumerate(sorted(set(df_stab["beta"]))):
                temp = df_stab.loc[(df_stab["block"] == block) & (df_stab["beta"] == beta)]
                fig.add_trace(go.Scatter(x=temp["zdims"], y=temp[st_var],
                                         name=f"\u03B2 {beta}", legendgroup=f"\u03B2 {beta}", showlegend=False,
                                         mode="markers", marker=dict(color=cmap[c], size=size)), row=row, col=col)

        # Plot down
        gam_type = "linear" # if "linear" in df_down["gam_type"].unique()[0] else "splines"
        for d, down_var in enumerate(down_vars):
            for o, out in enumerate(select_down):
                row = len(sc_vars) + len(stab_vars) + 1 + o
                if col == 1:
                    tit = titles_map[down_var]+"<br>"+out if down_var in titles_map.keys() else down_var+"<br>"+out
                    fig.update_yaxes(title_text=tit, row=row, col=1)
                for c, beta in enumerate(sorted(set(df_down["beta"]))):
                    temp = df_down.loc[(df_down["block"] == block) & (df_down["beta"] == beta) &
                                       (df_down["sim_var"] == out) & (df_down["gam_type"]==gam_type),]
                    zdims_j = temp["zdims"] + np.random.uniform(-jitter, jitter, size=len(temp["zdims"]))
                    fig.add_trace(go.Scatter(x=zdims_j, y=temp[down_var],
                                             name=f"\u03B2 {beta}", legendgroup=f"\u03B2 {beta}", showlegend=False,
                                             mode="markers", marker=dict(color=cmap[c], size=size)), row=row, col=col)

        fig.update_layout(template="plotly_white", height=200+nrows*125, width=100+ncols*300,
                          legend=dict(orientation="h", y=1.1, x=0))

        pio.write_html(fig, os.path.join(dir, f"assessment_summary_{title}.html"))
        pio.write_image(fig, os.path.join(dir, f"assessment_summary_{title}.png"), height=200+nrows*125, width=100+ncols*300, scale=3)
        # fig.show("browser")


def plot_assessment_perdim(df_scores_perdim):

    # Create a new variable: kl_perdim_avg
    # 1. Filter only KL rows
    kl_df = df_scores_perdim[df_scores_perdim["var"] == "kl"]

    group_cols = ["block", "zdims", "beta", "ikf", "okf"]

    # Plot KL per dim
    # kl_df ya está filtrado a var=="kl"
    # Asegúrate de tener la columna de dimensión latente, por ejemplo "z"
    # (si se llama distinto, cambia "z" por tu nombre real)
    kl_sorted = (
        kl_df
        .sort_values(group_cols + ["val"], ascending=[True] * len(group_cols) + [False])
        .assign(kl_rank=lambda d: d.groupby(group_cols).cumcount())
    )

    fig = px.strip(kl_sorted, x="kl_rank", y="val", color="zdims", facet_col="beta", facet_row="block")
    pio.write_html(fig, os.path.join(dir, f"kl_perdim.html"))
    # fig.show("browser")


def plot_brains(data, metric, title, atlas_path, save_dir, save_title, colorscale="viridis", crange=None, show=False):
    """
    Prepared to be used in the project with AAL atlas.

    :param data: dataframe with at least two columns (roi, metric)
    :param metric: the name of the column of values to use
    :param atlas_path: path to the atlas image
    :return:
    """

    # 1) Atlas AAL + labels
    atlas_img = nib.load(os.path.join(atlas_path, "ROI_MNI_V4.nii"))
    atlas_data = atlas_img.get_fdata().astype(int)

    atlas_labels = pd.read_csv(os.path.join(atlas_path, "ROI_MNI_V4.txt"), sep="\t", header=None)
    atlas_labels.columns = ["abbr", "roi", "id"]


    # 2) Vector de valores por label_id
    atlas_labels_vals = atlas_labels.copy()
    atlas_labels_vals = atlas_labels_vals.set_index("roi")
    atlas_labels_vals = atlas_labels_vals.join(data.set_index("roi")[metric], on="roi", how="left")

    # 3) Construir volumen con esos valores
    stat_data = np.zeros_like(atlas_data, dtype=float)

    # En AAL, los voxels tienen IDs 1..N. labels está en orden 1..N
    for i, row in atlas_labels_vals.iterrows():
        stat_data[atlas_data == row["id"]] = row[metric]

    stat_img = nib.Nifti1Image(stat_data, atlas_img.affine, atlas_img.header)

    surfaces = fetch_fslr()
    lh_surf, rh_surf = surfaces['inflated']

    # 2) MNI (volumen) -> fsLR (superficie). Devuelve GiftiImages (normalmente)
    gii_lh, gii_rh = mni152_to_fslr(stat_img, method="linear")

    # 3) Extrae arrays 1D por vértice (lo que surfplot necesita)
    data_lh = np.asarray(gii_lh.darrays[0].data)
    data_rh = np.asarray(gii_rh.darrays[0].data)

    # 4) Plot
    p = Plot(surf_lh=lh_surf, surf_rh=rh_surf)
    p.add_layer({"left": data_lh, "right": data_rh}, cmap=colorscale, cbar=True, color_range=crange)

    fig = p.build()
    cbar_ax = fig.axes[1]
    cbar_ax.set_xlabel(f"{metric}", fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.98)

    fig.savefig(os.path.join(save_dir, f"regional_{metric}_{save_title}.png"), dpi=300, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    plt.close()

    return os.path.join(save_dir, f"regional_{metric}_{save_title}.png")


def plot_mosaic_linear(paths, title, save_dir, save_title, show=False):

    fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 4))

    if len(paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, f"mosaic_{save_title}.png"),
        dpi=300, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    plt.close()

# %%
def plot_mosaic_pairplot(paths_info, block_names, title, save_dir, save_title, show=False,
                         figsize_scale=3.5, row_labels=True, col_labels=True):
    """
    Plot a pairwise mosaic of precomputed PNG images.

    Parameters
    ----------
    paths_info : list of tuples/lists
        Each element must be (i, j, path), where:
        - i = row index
        - j = column index
        - path = image file path
    block_names : list of str
        Names of the blocks, used as row/column labels.
    title : str
        Figure title.
    save_dir : str
        Directory where the final mosaic will be saved.
    save_title : str
        Output filename suffix.
    show : bool, optional
        Whether to display the figure.
    figsize_scale : float, optional
        Size multiplier per cell.
    row_labels : bool, optional
        Whether to display row labels.
    col_labels : bool, optional
        Whether to display column labels.
    """

    n = len(block_names)

    fig, axes = plt.subplots(
        n, n,
        figsize=(figsize_scale * n, figsize_scale * n),
        squeeze=False
    )

    # Turn all axes off by default
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Fill available images
    for i, j, path in paths_info:
        if path is None or not os.path.exists(path):
            continue

        img = mpimg.imread(path)
        ax = axes[i, j]
        ax.imshow(img)
        ax.axis("off")

    # Add labels
    if col_labels:
        for j, name in enumerate(block_names):
            # axes[0, j].set_title(name, fontsize=12, pad=12)
            axes[-1, j].text(
                0.5, -0.12,
                name,
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                transform=axes[-1, j].transAxes
            )

    if row_labels:
        for i, name in enumerate(block_names):
            # axes[i, 0].set_ylabel(name, fontsize=12, rotation=90, labelpad=18)
            axes[i, 0].text(
                -0.35, 0.5,
                name,
                ha="center",
                va="center",
                rotation=0,
                fontsize=13,
                fontweight="bold",
                transform=axes[i, 0].transAxes
            )

    fig.suptitle(title, fontsize=18, y=0.97)
    fig.tight_layout()

    plt.subplots_adjust(
        left=0.125,
        right=0.98,
        top=0.93,
        bottom=0.05,
        wspace=0.02,
        hspace=0.02
    )

    out_path = os.path.join(save_dir, f"mosaic_pairplot_{save_title}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    plt.close(fig)
    return out_path

