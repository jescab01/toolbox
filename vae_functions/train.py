import os
import gc
import time
import numpy as np
import pandas as pd
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from models import mVAE, mVAE_mixed, compute_ordinal_class_weights, compute_binary_class_weights
from model_selection import compute_hidden_dims, select_hp
from assess import  assess_mVAE_performance, assess_PCA_benchmarck, assess_downstream, assess_latent_stability
from assess import assess_mVAE_sdh_performance, assess_downstream_sdh
from assess import assess_mVAE_fc
from plotting import plot_inner_folding_effects, plot_assessment


# %% Training functions for Exposome
def nCV_mVAE_block(data, block, hparams, gkf, gkf_vars=None, max_epochs=50, batch_size=30, lr=1e-3, patience=10, delta=1e-3,
                   beta_warmup=0.3, dir=None, cfg_ass=None, cfg_down=None, cfg_plot=None):
    '''

    :param data:
    :param block:
    :param hparams:
    :param kf:
    :param max_epochs:
    :param batch_size:
    :param lr:
    :param patience:
    :param delta:
    :param beta_warmup:
    :param dir:
    :param cfg_ass:
    :param cfg_down:  data_down, gam_type & vars_down = downstream associations options -
        # out: ['ent_E', 'ent_I',  'rate_E', 'rate_I',  'target', 'EI_ent', 'EI_rate']
        # net: ['DMN', 'DAN', 'VAN', 'SMN', 'VIS', 'LIM', 'FPN']; roi: range(90) - AAL regions.
    :return:
    '''


    # Data
    fnames = data.columns.values.tolist()

    # Prepare country-year groups for k-folding
    data_cy = data.copy().reset_index()
    data_cy["gkf_var"] = data_cy[gkf_vars].astype(str).agg("_".join, axis=1)

    scores, scores_perdim, down, down_perdim, down_curve = (
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


    # Create aux dirs
    vae_inn_dir = os.path.join(dir, block, f"VAE_inner")
    vae_out_dir = os.path.join(dir, block, f"VAE_outer")
    vae_inn_vs = os.path.join(vae_inn_dir, f"versions")

    _ = [os.makedirs(path, exist_ok=True) for path in [vae_out_dir, vae_inn_vs]]


    pr_end = "\n" if cfg_ass["verbose"] or cfg_down["verbose"] else "\r"

    # %% Hyper parameters (those depending on data structure): hiddendims, warm-up steps.
    hiddendims, hd_info = compute_hidden_dims(data)
    warmup_steps = int((data.shape[0] / batch_size) * max_epochs * beta_warmup)

    tic0 = time.time()

    for idx_outer_kf, (idx_outer_train, idx_outer_test) in enumerate(gkf.split(data_cy, groups=data_cy["gkf_var"])):

        X_outer_train, X_outer_test = data.iloc[idx_outer_train].values, data.iloc[idx_outer_test].values
        pass

        # # Define groups for downstream splitting
        # cygroups_outer_train = set(data_cy["g_cy"].iloc[idx_outer_train].values)
        # cygroups_outer_test = set(data_cy["g_cy"].iloc[idx_outer_test].values)

        tic_out = time.time()

        #  ---- INNER LOOP (model selection) ----
        data_inn = data.iloc[idx_outer_train].copy()
        data_cy_inn = data_cy.iloc[idx_outer_train].copy()

        for idx_inner_kf, (idx_inner_train, idx_inner_val) in enumerate(gkf.split(data_cy_inn, groups=data_cy_inn["gkf_var"])):

            X_inner_train, X_inner_val = data_inn.iloc[idx_inner_train].values, data_inn.iloc[idx_inner_val].values
            pass


            tic_inn = time.time()

            # 1) Fit preprocessing ONLY on inner_train
            scaler = StandardScaler()
            X_inner_train_sc = scaler.fit_transform(X_inner_train)
            X_inner_val_sc = scaler.transform(X_inner_val)

            # Fill NaNs with zero
            X_inner_train_sc_fill = np.nan_to_num(X_inner_train_sc, nan=0.0)
            X_inner_val_sc_fill = np.nan_to_num(X_inner_val_sc, nan=0.0)

            # Get the MASKS for NaNs
            M_inner_train = (~np.isnan(X_inner_train)).astype(np.float32)
            M_inner_val = (~np.isnan(X_inner_val)).astype(np.float32)

            # DataLoader for efficient network feeding (training)
            X_train_dl = DataLoader(TensorDataset(
                torch.tensor(X_inner_train_sc_fill, dtype=torch.float32),
                torch.tensor(M_inner_train, dtype=torch.float32)),
                batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
            X_val_dl = DataLoader(TensorDataset(
                torch.tensor(X_inner_val_sc_fill, dtype=torch.float32),
                torch.tensor(M_inner_val, dtype=torch.float32)),
                batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)


            print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf} :: data ready. Running first training ...", end="\r")

            for v_id, (zdims, beta) in enumerate(hparams):
                pass

                tic_hp = time.time()

                # %% 2) Create VAE MODEL and TRAIN it
                vae_model = mVAE(input_dim=X_inner_train_sc.shape[1], latent_dim=zdims, hidden_dims=hiddendims, lr=lr,
                                 beta_end=beta, beta_warmup_steps=warmup_steps, missing_dropout_p=0.1, use_mask_in_encoder=False)

                v_tag = f"z{zdims}-b{beta}_o{idx_outer_kf}i{idx_inner_kf}"
                logger = CSVLogger(vae_inn_vs, name=None, version=v_tag)
                n_batch, wu_epochs = np.floor(len(X_inner_train) / batch_size), int(warmup_steps / len(X_train_dl))
                # Training Callbacks :: stopping criteria. Include them in trainer.
                early = EarlyStopping(monitor="val_loss_fbeta", mode="min", patience=patience, min_delta=delta)  # or "val_recon"
                trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                                     accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                                     enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,)
                trainer.fit(vae_model, X_train_dl, X_val_dl)


                print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf}  ({v_id+1}/{len(hparams)}) z{zdims} - b{beta}  :: "
                      f"time (hp){round((time.time()-tic_hp)/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() - tic_out)/60, 2)} minutes", end=pr_end)


                # %% Training Assessment
                _, sc_temp, sc_temp_per_dim = (
                    assess_mVAE_performance(block, X_inner_val_sc_fill, M_inner_val, fnames, vae_model,
                                            zdims, beta, idx_inner_kf, idx_outer_kf, vae_inn_vs, v_tag,
                                            avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                            seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

                scores = pd.concat([scores, sc_temp])
                scores_perdim = pd.concat([scores_perdim, sc_temp_per_dim])

                _, down_temp, down_temp_perdim, down_temp_curve = (
                    assess_downstream(vae_model, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                                      block, zdims, beta, idx_inner_kf, idx_outer_kf,
                                      covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"],
                                      verbose=cfg_down["verbose"]))

                down = pd.concat([down, down_temp])
                down_perdim = pd.concat([down_perdim, down_temp_perdim])
                down_curve = pd.concat([down_curve, down_temp_curve])


                # Evaluate features prediction
                if "feat" not in cfg_down["avoid"]:
                    _, down_temp, down_temp_perdim, down_temp_curve = (
                        assess_downstream(None, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                                          block, zdims, beta, idx_inner_kf, idx_outer_kf,
                                          covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"],
                                          verbose=cfg_down["verbose"]))

                    down = pd.concat([down, down_temp])
                    down_perdim = pd.concat([down_perdim, down_temp_perdim])
                    down_curve = pd.concat([down_curve, down_temp_curve])


                if "pca" not in cfg_ass["avoid"]:

                    _, sc_temp, sc_temp_perdim, pca_model = (
                        assess_PCA_benchmarck(block, X_inner_train_sc_fill, X_inner_val_sc_fill, fnames, zdims, beta,
                                          idx_inner_kf, idx_outer_kf, vae_inn_vs, v_tag,
                                          avoid=cfg_ass["avoid"], seed=42, verbose=False))

                    scores = pd.concat([scores, sc_temp])
                    scores_perdim = pd.concat([scores_perdim, sc_temp_perdim])

                    if "pca" not in cfg_down["avoid"]:
                        _, down_temp, down_temp_perdim, down_temp_curve = (
                            assess_downstream(pca_model, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                                              block, zdims, beta, idx_inner_kf, idx_outer_kf,
                                              covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"], verbose=cfg_down["verbose"]))

                        down = pd.concat([down, down_temp])
                        down_perdim = pd.concat([down_perdim, down_temp_perdim])
                        down_curve = pd.concat([down_curve, down_temp_curve])


                # si tienes referencias a outputs grandes, bórralas aquí
                del sc_temp, sc_temp_per_dim, down_temp, down_temp_perdim
                gc.collect()

                # si usas torch, vacía caché (aunque estés en CPU no molesta)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf}  _completed :: "
                  f"time (hp){round(( time.time()-tic_hp )/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() -tic_out) / 60, 2)} minutes", end=pr_end)

        print(f"[out{idx_outer_kf}] INNER loop __completed :: "
              f"time (hp){round(( time.time()-tic_hp )/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() -tic_out) / 60, 2)} minutes")



        # 3.2 Average inner performance per hparams, and select best to process further
        # scores_innAvg = scores.loc[scores["ikf"] == idx_inner_kf].groupby(["zdims", "beta"]).mean("numeric").reset_index()
        # scores_innBest = scores_innAvg.iloc[scores_innAvg["mse"].idxmin()]
        # zdims_innBest, beta_innBest = scores_innBest[["zdims", "beta"]].values
        # zdims_innBest = int(zdims_innBest)
        hp_sel = select_hp(scores, scores_per_dim=scores_perdim,
                           rule="1se", metric="mse", direction="min", group="zb", kl_th=0.01, reduce_z_to_active=False)
        zdims_innBest, beta_innBest = hp_sel["chosen_z"], hp_sel["chosen_beta"]
        # %% 4. OUTER loop - complete the outer iteration by training the model with the selected hyper-params
        tic_out2 = time.time()
        print(f"[OUTER {idx_outer_kf}] Best model z{zdims_innBest} - b{beta_innBest} :: ", end="")

        # 4.1) Fit preprocessing on outer_train
        scaler = StandardScaler()
        X_outer_train_sc = scaler.fit_transform(X_outer_train)
        X_outer_test_sc = scaler.transform(X_outer_test)

        # Fill NaNs with zero
        X_outer_train_sc_fill = np.nan_to_num(X_outer_train_sc, nan=0.0)
        X_outer_test_sc_fill = np.nan_to_num(X_outer_test_sc, nan=0.0)

        # Get the MASKS for NaNs
        M_outer_train = (~np.isnan(X_outer_train)).astype(np.float32)
        M_outer_test = (~np.isnan(X_outer_test)).astype(np.float32)

        # DataLoader for efficient network feeding (training)
        X_train_dl = DataLoader(TensorDataset(
            torch.tensor(X_outer_train_sc_fill, dtype=torch.float32),
            torch.tensor(M_outer_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
        X_test_dl = DataLoader(TensorDataset(
            torch.tensor(X_outer_test_sc_fill, dtype=torch.float32),
            torch.tensor(M_outer_test, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)


        # 4.2) Create VAE MODEL and TRAIN it
        vae_model = mVAE(input_dim=X_outer_train_sc.shape[1], latent_dim=zdims_innBest, hidden_dims=hiddendims, lr=lr,
                         beta_end=beta_innBest, beta_warmup_steps=warmup_steps, missing_dropout_p=0.1, use_mask_in_encoder=False)

        v_tag = f"o{idx_outer_kf}-innBest_z{zdims_innBest}-b{beta_innBest}"
        logger = CSVLogger(vae_out_dir, name=None, version=v_tag)
        n_batch, wu_epochs = np.floor(len(X_inner_train) / batch_size), int(warmup_steps / len(X_train_dl))
        trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                             accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                             enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False)
        trainer.fit(vae_model, X_train_dl, X_test_dl)

        print(f"\tModel training ({round((time.time() - tic_out2) / 60, 2)}min)")

        # %% Training Assessment
        _, scores_out, scores_perdim_out = (
            assess_mVAE_performance(block, X_outer_test_sc_fill, M_outer_test, fnames, vae_model,
                                    zdims_innBest, beta_innBest, None, idx_outer_kf, vae_out_dir, v_tag,
                                    avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                    seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

        _, down_out, down_perdim_out, down_curve_out = (
            assess_downstream(vae_model, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                              block, zdims_innBest, beta_innBest, None, idx_outer_kf,
                              covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"],
                              verbose=cfg_down["verbose"]))


        # Evaluate features prediction
        if "feat" not in cfg_down["avoid"]:
            _, down_temp, down_temp_perdim, down_temp_curve = (
                assess_downstream(None, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                                  block, zdims_innBest, beta_innBest, None, idx_outer_kf,
                                  covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"],
                                  verbose=cfg_down["verbose"]))

            down_out = pd.concat([down_out, down_temp])
            down_perdim_out = pd.concat([down_perdim_out, down_temp_perdim])
            down_curve_out = pd.concat([down_curve_out, down_temp_curve])

        if "pca" not in cfg_ass["avoid"]:
            _, sc_temp, sc_temp_perdim, pca_model = (
                assess_PCA_benchmarck(block, X_inner_train_sc_fill, X_inner_val_sc_fill, fnames,
                                      zdims_innBest, beta_innBest, None, idx_outer_kf, vae_inn_vs, v_tag,
                                      avoid=cfg_ass["avoid"], seed=42, verbose=False))

            scores_out = pd.concat([scores_out, sc_temp])
            scores_perdim_out = pd.concat([scores_perdim_out, sc_temp_perdim])

            if "pca" not in cfg_down["avoid"]:
                _, down_temp, down_temp_perdim, down_temp_curve = (
                    assess_downstream(pca_model, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                                      block, zdims_innBest, beta_innBest, None, idx_outer_kf,
                                      covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"],
                                      verbose=cfg_down["verbose"]))

                down_out = pd.concat([down_out, down_temp])
                down_perdim_out = pd.concat([down_perdim_out, down_temp_perdim])
                down_curve_out = pd.concat([down_curve_out, down_temp_curve])


        scores_out.to_csv(os.path.join(vae_out_dir, v_tag, f"scores_out.csv"), index=False)
        scores_perdim_out.to_csv(os.path.join(vae_out_dir, v_tag, f"scores_perdim_out.csv"), index=False)

        down_out.to_csv(os.path.join(vae_out_dir, v_tag, f"down_out.csv"), index=False)
        down_perdim_out.to_csv(os.path.join(vae_out_dir, v_tag, f"down_perdim_out.csv"), index=False)
        down_curve_out.to_csv(os.path.join(vae_out_dir, v_tag, f"down_curve_out.csv"), index=False)

        # Clean up
        del trainer, vae_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # CV Folding effects
    plot_inner_folding_effects(block, scores.loc[scores["model"]=="mVAE"], vae_inn_dir, var="mse")
    plot_inner_folding_effects(block, scores.loc[scores["model"]=="mVAE"], vae_inn_dir, var="kl")

    # Latent Stability across repetitions TODO adapt to multiple models: PCA mVAE
    stability, stab_perdim = (  # uses r_zX for stability
        assess_latent_stability(block, scores_perdim, metric="corr", topk=3, pairs_mode="all",
                                sim_thr=0.7, jacc_thr=0.5,  aggfunc="mean"))

    # Plot assessment summary
    plot_assessment([block], df_scores=scores,  df_stab=stability, df_down=down,
                    select_vars=cfg_plot["select_vars"], select_down=cfg_plot["select_down"],
                    dir=vae_inn_dir, params=cfg_plot["params"],)

    # Save results
    scores.to_csv(os.path.join(vae_inn_dir, "scores.csv"), index=False)
    scores_perdim.to_csv(os.path.join(vae_inn_dir, "scores_perdim.csv"), index=False)

    down.to_csv(os.path.join(vae_inn_dir, "down.csv"), index=False)
    down_perdim.to_csv(os.path.join(vae_inn_dir, "down_perdim.csv"), index=False)
    down_curve.to_csv(os.path.join(vae_inn_dir, "down_curve.csv"), index=False)

    stability.to_csv(os.path.join(vae_inn_dir, "stability.csv"), index=False)
    stab_perdim.to_csv(os.path.join(vae_inn_dir, "stability_perdim.csv"), index=False)

    plt.close("all")  # por si algún fig quedó abierto
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n\t\tnested-CV ended :: Block {block.upper()} . Time {round((time.time() - tic0) / 60, 2)}min\n")

    return scores, scores_perdim, down, down_perdim, down_curve, stability, stab_perdim




def fTrain_mVAE_block(data, block,  zdims, beta, max_epochs=50, batch_size=30,
                      lr=1e-3, patience=10, delta=1e-3, beta_warmup=0.3, dir=None,
                      cfg_ass=None, cfg_down=None):

    # Data
    X = data.values
    fnames = data.columns.tolist()

    # Create aux dirs
    vae_fin_dir = os.path.join(dir, block, f"VAE_final")
    os.makedirs(vae_fin_dir, exist_ok=True)


    # %% Hyper parameters (those depending on data structure): hiddendims, warm-up steps.
    hiddendims, hd_info = compute_hidden_dims(data)
    warmup_steps = int((X.shape[0] / batch_size) * max_epochs * beta_warmup)


    tic0 = time.time()

    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42, shuffle=True)


    # 1) Fit preprocessing ONLY on train
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Fill NaNs with zero
    X_train_sc_fill = np.nan_to_num(X_train_sc, nan=0.0)
    X_val_sc_fill = np.nan_to_num(X_val_sc, nan=0.0)

    # Get the MASKS for NaNs
    M_inner_train = (~np.isnan(X_train)).astype(np.float32)
    M_inner_val = (~np.isnan(X_val)).astype(np.float32)

    # DataLoader for efficient network feeding (training)
    X_train_dl = DataLoader(TensorDataset(
        torch.tensor(X_train_sc_fill, dtype=torch.float32),
        torch.tensor(M_inner_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
    X_val_dl = DataLoader(TensorDataset(
        torch.tensor(X_val_sc_fill, dtype=torch.float32),
        torch.tensor(M_inner_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)

    print(f"[FINAL] Final model z{zdims} - b{beta}  ::  Data ready. Training ...", end="\r")
    # %% 2) Create VAE MODEL and TRAIN it
    vae_model = mVAE(input_dim=X_train_sc.shape[1], latent_dim=zdims, hidden_dims=hiddendims, lr=lr,
                     beta_end=beta, beta_warmup_steps=warmup_steps, missing_dropout_p=0.1, use_mask_in_encoder=False)

    v_tag = f"z{zdims}-b{beta}"
    logger = CSVLogger(vae_fin_dir, name=None, version=v_tag)
    n_batch, wu_epochs = np.floor(len(X_train) / batch_size), int(warmup_steps / len(X_train_dl))
    # Training Callbacks :: stopping criteria. Include them in trainer.
    early = EarlyStopping(monitor="val_loss_fbeta", mode="min", patience=patience, min_delta=delta)  # or "val_recon"
    trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                         accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                         enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=True,)
    trainer.fit(vae_model, X_train_dl, X_val_dl)

    print(f"[FINAL] {block.upper()} training :: done - time {round((time.time() - tic0) / 60, 2)} minutes")


    # %% Training Assessment of VAE on full data
    # 1) Fit preprocessing ONLY on train
    X_sc = scaler.fit_transform(X)
    # Fill NaNs with zero
    X_sc_fill = np.nan_to_num(X_sc, nan=0.0)
    # Get the MASKS for NaNs
    M = (~np.isnan(X)).astype(np.float32)

    Z_mu, sc_temp, sc_temp_per_dim = (
        assess_mVAE_performance(block, X_sc_fill, M, fnames, vae_model,
                                zdims, beta, np.nan, np.nan, vae_fin_dir, v_tag,
                                avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

    Z_ni, down_temp, down_temp_perdim, down_temp_curve = (
        assess_downstream(vae_model, cfg_down["data_indiv"], fnames, cfg_down["vars_sim"], scaler,
                          block, zdims, beta, np.nan, np.nan,
                          covars=cfg_down["covars"], order_by=cfg_down["r2inc_curve_order"],
                          verbose=cfg_down["verbose"]))

    # # Plot assessment summary
    # plot_assessment([block], df_scores=sc_temp,  df_stab=None, df_down=down_temp,
    #                 select_vars=cfg_plot["select_vars"], select_down=cfg_plot["select_down"], dir=vae_fin_dir)

    # Save results
    sc_temp.to_csv(os.path.join(vae_fin_dir, "scores.csv"), index=False)
    sc_temp_per_dim.to_csv(os.path.join(vae_fin_dir, "scores_per_dim.csv"), index=False)

    down_temp.to_csv(os.path.join(vae_fin_dir, "down.csv"), index=False)
    down_temp_perdim.to_csv(os.path.join(vae_fin_dir, "down_per_dim.csv"), index=False)
    down_temp_curve.to_csv(os.path.join(vae_fin_dir, "down_curve.csv"), index=False)

    # clean up
    plt.close("all")  # por si algún fig quedó abierto
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[FINAL] {block.upper()} evaluation :: ended.  {round((time.time() - tic0) / 60, 2)} additional mins. ")

    return Z_mu, Z_ni, sc_temp, sc_temp_per_dim, down_temp, down_temp_perdim, down_temp_curve


# %% Training functions for Social Determinants
def nCV_mmVAE_block(data, block, vars_meta, hparams, kf, max_epochs=50, batch_size=30, lr=1e-3, patience=10, delta=1e-3,
                   beta_warmup=0.3, logs_dir=None, cfg_ass=None, cfg_plot=None,
                       alpha_num=1, alpha_ord=1, alpha_bin=1):
    '''

    :param data:
    :param block:
    :param vars_meta:
    :param hparams:
    :param kf:
    :param max_epochs:
    :param batch_size:
    :param lr:
    :param patience:
    :param delta:
    :param beta_warmup:
    :param logs_dir:
    :param cfg_ass:
    :param cfg_plot:
    :param alpha_num:
    :param alpha_ord:
    :param alpha_bin:
    :return:
    '''

    # Data
    fnames = data.columns.values.tolist()

    # Define var types
    meta = vars_meta.set_index("name")

    idx_num = [i for i, var in enumerate(fnames) if meta.at[var, "type"] == "numeric"]
    idx_ord = [i for i, var in enumerate(fnames) if meta.at[var, "type"] == "ordinal"]
    idx_bin = [i for i, var in enumerate(fnames) if meta.at[var, "type"] == "binary"]
    ord_ncat = [int(meta.at[var, "ord_ncat"]) for var in fnames if meta.at[var, "type"] == "ordinal"]

    scores, scores_perdim, = pd.DataFrame(), pd.DataFrame()
    scores_out, scores_perdim_out = pd.DataFrame(), pd.DataFrame()

    # Create aux dirs
    vae_inn_dir = os.path.join(logs_dir, block, f"VAE_inner")
    vae_out_dir = os.path.join(logs_dir, block, f"VAE_outer")
    vae_inn_vs = os.path.join(vae_inn_dir, f"versions")

    _ = [os.makedirs(path, exist_ok=True) for path in [vae_out_dir, vae_inn_vs]]


    pr_end = "\n" if cfg_ass["verbose"] else "\r"

    # %% Hyper parameters (those depending on data structure): hiddendims, warm-up steps.
    hiddendims, hd_info = compute_hidden_dims(data)
    warmup_steps = int((data.shape[0] / batch_size) * max_epochs * beta_warmup)

    cfg_ass_inn = copy.deepcopy(cfg_ass)
    cfg_ass_inn["avoid"].append("decoder_grid")

    tic0 = time.time()


    for idx_outer_kf, (idx_outer_train, idx_outer_test) in enumerate(kf.split(data)):

        X_outer_train, X_outer_test = data.iloc[idx_outer_train].values, data.iloc[idx_outer_test].values
        pass


        tic_out = time.time()

        #  ---- INNER LOOP (model selection) ----
        data_inn = data.iloc[idx_outer_train].copy()

        for idx_inner_kf, (idx_inner_train, idx_inner_val) in enumerate(kf.split(data_inn)):

            X_inner_train, X_inner_val = data_inn.iloc[idx_inner_train].values, data_inn.iloc[idx_inner_val].values
            pass


            tic_inn = time.time()

            # 1) Fit preprocessing ONLY on inner_train
            scaler = StandardScaler()

            X_inner_train_scmix_fill = X_inner_train.copy().astype(float)
            X_inner_val_scmix_fill = X_inner_val.copy().astype(float)

            if len(idx_num) > 0:
                X_inner_train_scmix_fill[:, idx_num] = scaler.fit_transform(X_inner_train[:, idx_num])
                X_inner_val_scmix_fill[:, idx_num] = scaler.transform(X_inner_val[:, idx_num])

            # Fill NaNs with zero; also categoricals. Add use mask in encoder.
            X_inner_train_scmix_fill = np.nan_to_num(X_inner_train_scmix_fill, nan=0.0)
            X_inner_val_scmix_fill = np.nan_to_num(X_inner_val_scmix_fill, nan=0.0)

            # Get the MASKS for NaNs
            M_inner_train = (~np.isnan(X_inner_train)).astype(np.float32)
            M_inner_val = (~np.isnan(X_inner_val)).astype(np.float32)

            # ORD data for ordinal loss. Fill SOLO para evitar NaNs en tensores
            X_inner_train_ord = X_inner_train[:, idx_ord].copy()
            M_inner_train_ord = M_inner_train[:, idx_ord].copy()
            X_inner_train_ord_fill = np.nan_to_num(X_inner_train_ord, nan=0).astype(np.int64)
            X_inner_val_ord = X_inner_val[:, idx_ord].copy()
            X_inner_val_ord_fill = np.nan_to_num(X_inner_val_ord, nan=0).astype(np.int64)

            # BIN data for binary loss.
            X_inner_train_bin = X_inner_train[:, idx_bin].copy()
            M_inner_train_bin = M_inner_train[:, idx_bin].copy()


            # DataLoader for efficient network feeding (training)
            X_train_dl = DataLoader(TensorDataset(
                torch.tensor(X_inner_train_scmix_fill, dtype=torch.float32),
                torch.tensor(M_inner_train, dtype=torch.float32),
                torch.tensor(X_inner_train_ord_fill, dtype=torch.long),),
                batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)

            X_val_dl = DataLoader(TensorDataset(
                torch.tensor(X_inner_val_scmix_fill, dtype=torch.float32),
                torch.tensor(M_inner_val, dtype=torch.float32),
                torch.tensor(X_inner_val_ord_fill, dtype=torch.long)),
                batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)

            # Loss weights per variable (+info, -info)
            var_weights = torch.ones(X_inner_train.shape[1], dtype=torch.float32)  # TODO ¿variable entropy?

            # Loss weights per response class (+freq, rare)
            bin_class_w = compute_binary_class_weights(
                X_inner_train_bin, M_inner_train_bin, max_weight=2, eps=1.0) if len(idx_bin) > 0 else None

            ord_class_w = compute_ordinal_class_weights(
                X_inner_train_ord, M_inner_train_ord, max_weight=2, eps=1.0, ordinal_n_classes=ord_ncat, ) if len(
                idx_ord) > 0 else None

            print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf} :: data ready. Running first training ...", end="\r")

            for v_id, (zdims, beta) in enumerate(hparams):
                pass

                tic_hp = time.time()

                # %% 2) Create VAE MODEL and TRAIN it; mask in encoder.
                vae_model = mVAE_mixed(
                    input_dim=X_inner_train.shape[1], latent_dim=zdims, hidden_dims=hiddendims, lr=lr,
                    beta_end=beta, beta_warmup_steps=warmup_steps, missing_dropout_p=0, use_mask_in_encoder=True,
                    numeric_idx=idx_num, ordinal_idx=idx_ord, binary_idx=idx_bin, ordinal_n_classes=ord_ncat,
                    ordinal_class_weights=ord_class_w, binary_class_weights=bin_class_w,
                    alpha_num=alpha_num, alpha_ord=alpha_ord, alpha_bin=alpha_bin)

                v_tag = f"z{zdims}-b{beta}_o{idx_outer_kf}i{idx_inner_kf}"
                logger = CSVLogger(vae_inn_vs, name=None, version=v_tag)
                n_batch, wu_epochs = np.floor(len(X_inner_train) / batch_size), int(warmup_steps / len(X_train_dl))
                # Training Callbacks :: stopping criteria. Include them in trainer.
                early = EarlyStopping(monitor="val_loss_fbeta", mode="min", patience=patience, min_delta=delta)  # or "val_recon"
                trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                                     accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                                     enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,)
                trainer.fit(vae_model, X_train_dl, X_val_dl)


                print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf}  ({v_id+1}/{len(hparams)}) z{zdims} - b{beta}  :: "
                      f"time (hp){round((time.time()-tic_hp)/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() - tic_out)/60, 2)} minutes", end=pr_end)


                # %% Training Assessment
                _, sc_temp, sc_temp_per_dim = (
                    assess_mVAE_sdh_performance(block, X_inner_val_scmix_fill, M_inner_val, fnames, vae_model,
                                            zdims, beta, idx_inner_kf, idx_outer_kf, vae_inn_vs, v_tag,
                                                X_inner_train_scmix_fill, M_inner_train, vars_meta=cfg_ass["vars_meta"],
                                            avoid=cfg_ass_inn["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                            seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

                scores = pd.concat([scores, sc_temp])
                scores_perdim = pd.concat([scores_perdim, sc_temp_per_dim])

                if "pca" not in cfg_ass["avoid"]:

                    _, sc_temp, sc_temp_perdim, pca_model = (
                        assess_PCA_benchmarck(block, X_inner_train_scmix_fill, X_inner_val_scmix_fill, fnames, zdims, beta,
                                          idx_inner_kf, idx_outer_kf, vae_inn_vs, v_tag,
                                          avoid=cfg_ass["avoid"], seed=42, verbose=False))

                    scores = pd.concat([scores, sc_temp])
                    scores_perdim = pd.concat([scores_perdim, sc_temp_perdim])


            # si usas torch, vacía caché (aunque estés en CPU no molesta)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf}  _completed :: "
                  f"time (hp){round(( time.time()-tic_hp )/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() -tic_out) / 60, 2)} minutes", end=pr_end)

        print(f"[out{idx_outer_kf}] INNER loop __completed :: "
              f"time (hp){round(( time.time()-tic_hp )/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() -tic_out) / 60, 2)} minutes")



        # 3.2 Average inner performance per hparams, and select best to process further
        # scores_innAvg = scores.loc[scores["ikf"] == idx_inner_kf].groupby(["zdims", "beta"]).mean("numeric").reset_index()
        # scores_innBest = scores_innAvg.iloc[scores_innAvg["mse"].idxmin()]
        # zdims_innBest, beta_innBest = scores_innBest[["zdims", "beta"]].values
        # zdims_innBest = int(zdims_innBest)
        hp_sel = select_hp(scores.loc[scores["okf"]==idx_outer_kf], scores_per_dim=scores_perdim.loc[scores_perdim["okf"]==idx_outer_kf],
                           rule="1se", metric="recon_main_mean", direction="min", group="zb", kl_th=0.01, reduce_z_to_active=False)
        zdims_innBest, beta_innBest = hp_sel["chosen_z"], hp_sel["chosen_beta"]

        # %% 4. OUTER loop - complete the outer iteration by training the model with the selected hyper-params
        tic_out2 = time.time()
        print(f"[OUTER {idx_outer_kf}] Best model z{zdims_innBest} - b{beta_innBest} :: ", end="")

        # 4.1) Fit preprocessing on outer_train
        scaler = StandardScaler()

        X_outer_train_scmix_fill = X_outer_train.copy().astype(float)
        X_outer_test_scmix_fill = X_outer_test.copy().astype(float)

        if len(idx_num) > 0:
            X_outer_train_scmix_fill[:, idx_num] = scaler.fit_transform(X_outer_train[:, idx_num])
            X_outer_test_scmix_fill[:, idx_num] = scaler.transform(X_outer_test[:, idx_num])

        # Fill NaNs with zero
        X_outer_train_scmix_fill = np.nan_to_num(X_outer_train_scmix_fill, nan=0.0)
        X_outer_test_scmix_fill = np.nan_to_num(X_outer_test_scmix_fill, nan=0.0)

        # Get the MASKS for NaNs
        M_outer_train = (~np.isnan(X_outer_train)).astype(np.float32)
        M_outer_test = (~np.isnan(X_outer_test)).astype(np.float32)

        # ORD data for ordinal loss. Fill SOLO para evitar NaNs en tensores
        X_outer_train_ord = X_outer_train[:, idx_ord].copy()
        M_outer_train_ord = M_outer_train[:, idx_ord].copy()
        X_outer_train_ord_fill = np.nan_to_num(X_outer_train_ord, nan=0).astype(np.int64)
        X_outer_test_ord = X_outer_test[:, idx_ord].copy()
        X_outer_test_ord_fill = np.nan_to_num(X_outer_test_ord, nan=0).astype(np.int64)

        # BIN data for binary loss.
        X_outer_train_bin = X_outer_train[:, idx_bin].copy()
        M_outer_train_bin = M_outer_train[:, idx_bin].copy()

        # DataLoader for efficient network feeding (training)
        X_train_dl = DataLoader(TensorDataset(
            torch.tensor(X_outer_train_scmix_fill, dtype=torch.float32),
            torch.tensor(M_outer_train, dtype=torch.float32),
            torch.tensor(X_outer_train_ord_fill, dtype=torch.long),),
            batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
        X_test_dl = DataLoader(TensorDataset(
            torch.tensor(X_outer_test_scmix_fill, dtype=torch.float32),
            torch.tensor(M_outer_test, dtype=torch.float32),
            torch.tensor(X_outer_test_ord_fill, dtype=torch.long)),
            batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)

        # Loss weights per variable (+info, -info)
        var_weights = torch.ones(X_outer_train.shape[1], dtype=torch.float32)  # TODO ¿variable entropy?

        # Loss weights per response class (+freq, rare)
        bin_class_w = compute_binary_class_weights(
            X_outer_train_bin, M_outer_train_bin, max_weight=2, eps=1.0) if len(idx_bin) > 0 else None

        ord_class_w = compute_ordinal_class_weights(
            X_outer_train_ord, M_outer_train_ord, max_weight=2, eps=1.0, ordinal_n_classes=ord_ncat, ) if len(idx_ord) > 0 else None

        # 4.2) Create VAE MODEL and TRAIN it
        vae_model = mVAE_mixed(
            input_dim=X_outer_train.shape[1], latent_dim=zdims_innBest, hidden_dims=hiddendims, lr=lr,
            beta_end=beta_innBest, beta_warmup_steps=warmup_steps, missing_dropout_p=0, use_mask_in_encoder=True,
            numeric_idx=idx_num, ordinal_idx=idx_ord, binary_idx=idx_bin, ordinal_n_classes=ord_ncat,
            ordinal_class_weights=ord_class_w, binary_class_weights=bin_class_w,
            alpha_num=alpha_num, alpha_ord=alpha_ord, alpha_bin=alpha_bin
        )

        v_tag = f"o{idx_outer_kf}-innBest_z{zdims_innBest}-b{beta_innBest}"
        logger = CSVLogger(vae_out_dir, name=None, version=v_tag)
        n_batch, wu_epochs = np.floor(len(X_inner_train) / batch_size), int(warmup_steps / len(X_train_dl))
        trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,  callbacks=[early],
                             accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                             enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=True)
        trainer.fit(vae_model, X_train_dl, X_test_dl)

        print(f"\tModel training ({round((time.time() - tic_out2) / 60, 2)}min)")

        # %% Training Assessment
        _, sc_temp_out, sc_perdim_temp_out = (
            assess_mVAE_sdh_performance(block, X_outer_test_scmix_fill, M_outer_test, fnames, vae_model,
                                    zdims_innBest, beta_innBest, None, idx_outer_kf, vae_out_dir, v_tag,
                                        X_outer_train_scmix_fill, M_outer_train, cfg_ass["vars_meta"],
                                    avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                    seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

        scores_out = pd.concat([scores_out, sc_temp_out])
        scores_perdim_out = pd.concat([scores_perdim_out, sc_perdim_temp_out])

        if "pca" not in cfg_ass["avoid"]:
            _, sc_temp, sc_temp_perdim, pca_model = (
                assess_PCA_benchmarck(block, X_outer_train_scmix_fill, X_outer_test_scmix_fill, fnames,
                                      zdims_innBest, beta_innBest, None, idx_outer_kf, vae_inn_vs, v_tag,
                                      avoid=cfg_ass["avoid"], seed=42, verbose=False))

            scores_out = pd.concat([scores_out, sc_temp])
            scores_perdim_out = pd.concat([scores_perdim_out, sc_temp_perdim])

        # Clean up
        del trainer, vae_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # CV Folding effects
    plot_inner_folding_effects(block, scores.loc[scores["model"]=="mVAE"], vae_inn_dir, var="recon_main_mean")
    plot_inner_folding_effects(block, scores.loc[scores["model"]=="mVAE"], vae_inn_dir, var="kl")

    # Latent Stability across repetitions
    stability, stab_perdim = (  # uses r_zX for stability
        assess_latent_stability(block, scores_perdim, metric="corr", topk=3, pairs_mode="all",
                                sim_thr=0.7, jacc_thr=0.5,  aggfunc="mean"))

    # Plot assessment summary
    plot_assessment([block], df_scores=scores,  df_stab=stability,
                    select_vars=cfg_plot["select_vars"], select_down=cfg_plot["select_down"],
                    dir=vae_inn_dir, params=cfg_plot["params"],)

    # Save INNER results
    scores.to_csv(os.path.join(vae_inn_dir, "scores.csv"), index=False)
    scores_perdim.to_csv(os.path.join(vae_inn_dir, "scores_perdim.csv"), index=False)

    stability.to_csv(os.path.join(vae_inn_dir, "stability.csv"), index=False)
    stab_perdim.to_csv(os.path.join(vae_inn_dir, "stability_perdim.csv"), index=False)

    plt.close("all")  # por si algún fig quedó abierto
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Save OUTER results
    scores_out.to_csv(os.path.join(vae_out_dir,  f"scores_out.csv"), index=False)
    scores_perdim_out.to_csv(os.path.join(vae_out_dir, f"scores_perdim_out.csv"), index=False)


    print(f"\n\t\tnested-CV ended :: Block {block.upper()} . Time {round((time.time() - tic0) / 60, 2)}min\n")

    return scores, scores_perdim, stability, stab_perdim




def fTrain_mmVAE_block(data, block, vars_meta,  zdims, beta, max_epochs=50, batch_size=30,
                      lr=1e-3, patience=10, delta=1e-3, beta_warmup=0.3, logs_dir=None,
                      cfg_ass=None,  alpha_num=1, alpha_ord=1, alpha_bin=1):

    # Data
    X = data.values
    fnames = data.columns.tolist()

    # Define var types
    meta = vars_meta.set_index("name")

    idx_num = [i for i, var in enumerate(fnames) if meta.at[var, "type"] == "numeric"]
    idx_ord = [i for i, var in enumerate(fnames) if meta.at[var, "type"] == "ordinal"]
    idx_bin = [i for i, var in enumerate(fnames) if meta.at[var, "type"] == "binary"]
    ord_ncat = [int(meta.at[var, "ord_ncat"]) for var in fnames if meta.at[var, "type"] == "ordinal"]


    # Create aux dirs
    vae_fin_dir = os.path.join(logs_dir, block, f"VAE_final")
    os.makedirs(vae_fin_dir, exist_ok=True)


    # %% Hyper parameters (those depending on data structure): hiddendims, warm-up steps.
    hiddendims, hd_info = compute_hidden_dims(data)
    warmup_steps = int((X.shape[0] / batch_size) * max_epochs * beta_warmup)


    tic0 = time.time()

    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42, shuffle=True)


    # 1) Fit preprocessing ONLY on train
    scaler = StandardScaler()

    X_train_scmix_fill = X_train.copy().astype(float)
    X_val_scmix_fill = X_val.copy().astype(float)

    if len(idx_num) > 0:
        X_train_scmix_fill[:, idx_num] = scaler.fit_transform(X_train[:, idx_num])
        X_val_scmix_fill[:, idx_num] = scaler.transform(X_val[:, idx_num])

    # Fill NaNs with zero; also categoricals. Add use mask in encoder.
    X_train_scmix_fill = np.nan_to_num(X_train_scmix_fill, nan=0.0)
    X_val_scmix_fill = np.nan_to_num(X_val_scmix_fill, nan=0.0)

    # Get the MASKS for NaNs
    M_train = (~np.isnan(X_train)).astype(np.float32)
    M_val = (~np.isnan(X_val)).astype(np.float32)

    # ORD data for ordinal loss. Fill SOLO para evitar NaNs en tensores
    X_train_ord = X_train[:, idx_ord].copy()
    M_train_ord = M_train[:, idx_ord].copy()
    X_train_ord_fill = np.nan_to_num(X_train_ord, nan=0).astype(np.int64)
    X_val_ord = X_val[:, idx_ord].copy()
    X_val_ord_fill = np.nan_to_num(X_val_ord, nan=0).astype(np.int64)

    # BIN data for binary loss.
    X_train_bin = X_train[:, idx_bin].copy()
    M_train_bin = M_train[:, idx_bin].copy()


    # DataLoader for efficient network feeding (training)
    X_train_dl = DataLoader(TensorDataset(
        torch.tensor(X_train_scmix_fill, dtype=torch.float32),
        torch.tensor(M_train, dtype=torch.float32),
        torch.tensor(X_train_ord_fill, dtype=torch.long), ),
        batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)

    X_val_dl = DataLoader(TensorDataset(
        torch.tensor(X_val_scmix_fill, dtype=torch.float32),
        torch.tensor(M_val, dtype=torch.float32),
        torch.tensor(X_val_ord_fill, dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)

    # Loss weights per variable (+info, -info)
    var_weights = torch.ones(X_train.shape[1], dtype=torch.float32)  # TODO ¿variable entropy?

    # Loss weights per response class (+freq, rare)
    bin_class_w = compute_binary_class_weights(
        X_train_bin, M_train_bin, max_weight=2, eps=1.0) if len(idx_bin) > 0 else None

    ord_class_w = compute_ordinal_class_weights(
        X_train_ord, M_train_ord, max_weight=2, eps=1.0, ordinal_n_classes=ord_ncat, ) if len(
        idx_ord) > 0 else None


    print(f"[FINAL] Final model z{zdims} - b{beta}  ::  Data ready. Training ...", end="\r")
    # %% 2) Create VAE MODEL and TRAIN it
    vae_model = mVAE_mixed(input_dim=X_train.shape[1], latent_dim=zdims, hidden_dims=hiddendims, lr=lr,
                     beta_end=beta, beta_warmup_steps=warmup_steps, missing_dropout_p=0, use_mask_in_encoder=True,
                     numeric_idx=idx_num, ordinal_idx=idx_ord, binary_idx=idx_bin, ordinal_n_classes=ord_ncat,
                     ordinal_class_weights=ord_class_w, binary_class_weights=bin_class_w,
                     alpha_num=alpha_num, alpha_ord=alpha_ord, alpha_bin=alpha_bin
                     )

    v_tag = f"z{zdims}-b{beta}"
    logger = CSVLogger(vae_fin_dir, name=None, version=v_tag)
    n_batch, wu_epochs = np.floor(len(X_train) / batch_size), int(warmup_steps / len(X_train_dl))
    # Training Callbacks :: stopping criteria. Include them in trainer.
    early = EarlyStopping(monitor="val_loss_fbeta", mode="min", patience=patience, min_delta=delta)  # or "val_recon"
    trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                         accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                         enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=True,)
    trainer.fit(vae_model, X_train_dl, X_val_dl)

    print(f"[FINAL] {block.upper()} training :: done - time {round((time.time() - tic0) / 60, 2)} minutes")


    # %% Training Assessment of VAE on full data
    X_scmix_fill = X.copy().astype(float)
    if len(idx_num) > 0:
        X_scmix_fill[:, idx_num] = scaler.transform(X[:, idx_num])
    X_scmix_fill = np.nan_to_num(X_scmix_fill, nan=0.0)
    M = (~np.isnan(X)).astype(np.float32)

    Z_mu, sc_temp, sc_temp_per_dim = (
        assess_mVAE_sdh_performance(block, X_scmix_fill, M, fnames, vae_model,
                                zdims, beta, np.nan, np.nan, vae_fin_dir, v_tag,
                                X_scmix_fill, M, vars_meta=cfg_ass["vars_meta"],
                                avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))


    # Save results
    sc_temp.to_csv(os.path.join(vae_fin_dir, "scores.csv"), index=False)
    sc_temp_per_dim.to_csv(os.path.join(vae_fin_dir, "scores_per_dim.csv"), index=False)
    np.savez_compressed(
        os.path.join(vae_fin_dir, "final_training_arrays.npz"),
        X_scmix_fill=X_scmix_fill.astype(np.float32),
        M=M.astype(np.float32),
    )
    torch.save(
        {
            "artifact_version": 1,
            "block": block,
            "v_tag": v_tag,
            "model_state_dict": vae_model.state_dict(),
            "input_dim": int(X.shape[1]),
            "latent_dim": int(zdims),
            "hidden_dims": tuple(hiddendims),
            "beta": float(beta),
            "lr": float(lr),
            "feature_names": fnames,
            "idx_num": idx_num,
            "idx_ord": idx_ord,
            "idx_bin": idx_bin,
            "ord_ncat": ord_ncat,
            "scaler": scaler,
            "alpha_num": float(alpha_num),
            "alpha_ord": float(alpha_ord),
            "alpha_bin": float(alpha_bin),
            "use_mask_in_encoder": True,
            "seed": cfg_ass.get("seed", 42) if cfg_ass is not None else 42,
        },
        os.path.join(vae_fin_dir, "final_training_artifact.pt"),
    )


    # clean up
    plt.close("all")  # por si algún fig quedó abierto
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[FINAL] {block.upper()} evaluation :: ended.  {round((time.time() - tic0) / 60, 2)} additional mins. ")

    return Z_mu, sc_temp, sc_temp_per_dim # Z_ni, down_temp, down_temp_perdim, down_temp_curve

# %% Training functions for Social Determinants
def nCV_mVAE_fc(data, hparams, kf, max_epochs=50, batch_size=30, lr=1e-3, patience=10, delta=1e-3,
                   beta_warmup=0.3, logs_dir=None, cfg_ass=None):
    '''

    :param data:
    :param block:
    :param vars_meta:
    :param hparams:
    :param kf:
    :param max_epochs:
    :param batch_size:
    :param lr:
    :param patience:
    :param delta:
    :param beta_warmup:
    :param logs_dir:
    :param cfg_ass:
    :param cfg_plot:
    :param alpha_num:
    :param alpha_ord:
    :param alpha_bin:
    :return:
    '''

    # Data
    fnames = data.columns.values.tolist()

    scores, scores_perdim, = pd.DataFrame(), pd.DataFrame()
    scores_out, scores_perdim_out = pd.DataFrame(), pd.DataFrame()

    # Create aux dirs
    vae_inn_dir = os.path.join(logs_dir, f"VAE_inner")
    vae_inn_vs = os.path.join(vae_inn_dir, f"versions")
    vae_out_dir = os.path.join(logs_dir, f"VAE_outer")

    _ = [os.makedirs(path, exist_ok=True) for path in [vae_out_dir, vae_inn_vs]]


    pr_end = "\n" if cfg_ass["verbose"] else "\r"

    # %% Hyper parameters (those depending on data structure): warm-up steps.
    warmup_steps = int((data.shape[0] / batch_size) * max_epochs * beta_warmup)

    tic0 = time.time()

    for idx_outer_kf, (idx_outer_train, idx_outer_test) in enumerate(kf.split(data)):

        X_outer_train, X_outer_test = data.iloc[idx_outer_train].values, data.iloc[idx_outer_test].values
        pass


        tic_out = time.time()

        #  ---- INNER LOOP (model selection) ----
        data_inn = data.iloc[idx_outer_train].copy()

        for idx_inner_kf, (idx_inner_train, idx_inner_val) in enumerate(kf.split(data_inn)):

            X_inner_train, X_inner_val = data_inn.iloc[idx_inner_train].values, data_inn.iloc[idx_inner_val].values
            pass


            tic_inn = time.time()

            # 1) Fit preprocessing ONLY on inner_train
            scaler = StandardScaler()

            X_inner_train_sc = scaler.fit_transform(X_inner_train)
            X_inner_val_sc = scaler.transform(X_inner_val)

            # Fill NaNs with zero
            X_inner_train_sc_fill = np.nan_to_num(X_inner_train_sc, nan=0.0)
            X_inner_val_sc_fill = np.nan_to_num(X_inner_val_sc, nan=0.0)

            # Get the MASKS for NaNs
            M_inner_train = (~np.isnan(X_inner_train)).astype(np.float32)
            M_inner_val = (~np.isnan(X_inner_val)).astype(np.float32)

            # DataLoader for efficient network feeding (training)
            X_train_dl = DataLoader(TensorDataset(
                torch.tensor(X_inner_train_sc_fill, dtype=torch.float32),
                torch.tensor(M_inner_train, dtype=torch.float32)),
                batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
            X_val_dl = DataLoader(TensorDataset(
                torch.tensor(X_inner_val_sc_fill, dtype=torch.float32),
                torch.tensor(M_inner_val, dtype=torch.float32)),
                batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)


            print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf} :: data ready. Running first training ...", end="\r")

            for v_id, (hiddendims, beta) in enumerate(hparams):
                pass

                tic_hp = time.time()

                # %% 2) Create VAE MODEL and TRAIN it; mask in encoder.
                vae_model = mVAE(
                    input_dim=X_inner_train.shape[1], latent_dim=2, hidden_dims=hiddendims, lr=lr,
                    beta_end=beta, beta_warmup_steps=warmup_steps, missing_dropout_p=0, use_mask_in_encoder=False)

                v_tag = f"h{hiddendims[0]}_{hiddendims[1]}-b{beta}_o{idx_outer_kf}i{idx_inner_kf}"
                logger = CSVLogger(vae_inn_vs, name=None, version=v_tag)
                n_batch, wu_epochs = np.floor(len(X_inner_train) / batch_size), int(warmup_steps / len(X_train_dl))
                # Training Callbacks :: stopping criteria. Include them in trainer.
                early = EarlyStopping(monitor="val_loss_fbeta", mode="min", patience=patience, min_delta=delta)  # or "val_recon"
                trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                                     accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                                     enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,)
                trainer.fit(vae_model, X_train_dl, X_val_dl)


                print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf}  ({v_id+1}/{len(hparams)}) h{hiddendims} - b{beta}  :: "
                      f"time (hp){round((time.time()-tic_hp)/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() - tic_out)/60, 2)} minutes", end=pr_end)


                # %% Training Assessment
                _, sc_temp, sc_temp_per_dim = (
                    assess_mVAE_fc(X_inner_val_sc_fill, M_inner_val, fnames, vae_model,
                                    2, beta, hiddendims, idx_inner_kf, idx_outer_kf, vae_inn_vs, v_tag,
                                    avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                                    seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

                scores = pd.concat([scores, sc_temp])
                scores_perdim = pd.concat([scores_perdim, sc_temp_per_dim])


            # si usas torch, vacía caché (aunque estés en CPU no molesta)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[out{idx_outer_kf}] INNER kf{idx_inner_kf}  _completed :: "
                  f"time (hp){round(( time.time()-tic_hp )/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() -tic_out) / 60, 2)} minutes", end=pr_end)

        print(f"[out{idx_outer_kf}] INNER loop __completed :: "
              f"time (hp){round(( time.time()-tic_hp )/60, 2)} / (kf){round((time.time() - tic_inn)/60, 2)} // {round((time.time() -tic_out) / 60, 2)} minutes")



        # 3.2 Average inner performance per hparams, and select best to process further
        scores_innAvg = scores.loc[scores["okf"] == idx_outer_kf].groupby(["hdims", "beta"]).mean("numeric").reset_index()
        scores_innBest = scores_innAvg.iloc[scores_innAvg["mse"].idxmin()]
        hidden_innBest, beta_innBest = scores_innBest[["hdims", "beta"]].values

        # zdims_innBest = int(zdims_innBest)
        # hp_sel = select_hp(scores.loc[scores["okf"]==idx_outer_kf], scores_per_dim=scores_perdim.loc[scores_perdim["okf"]==idx_outer_kf],
        #                    rule="1se", metric="recon_main_mean", direction="min", group="zb", kl_th=0.01, reduce_z_to_active=False)
        # hidden_innBest, beta_innBest = hp_sel["chosen_z"], hp_sel["chosen_beta"]

        # %% 4. OUTER loop - complete the outer iteration by training the model with the selected hyper-params
        tic_out2 = time.time()
        print(f"[OUTER {idx_outer_kf}] Best model h{hidden_innBest} - b{beta_innBest} :: ", end="")

        # 4.1) Fit preprocessing on outer_train
        scaler = StandardScaler()
        X_outer_train_sc = scaler.fit_transform(X_outer_train)
        X_outer_test_sc = scaler.transform(X_outer_test)

        # Fill NaNs with zero
        X_outer_train_sc_fill = np.nan_to_num(X_outer_train_sc, nan=0.0)
        X_outer_test_sc_fill = np.nan_to_num(X_outer_test_sc, nan=0.0)

        # Get the MASKS for NaNs
        M_outer_train = (~np.isnan(X_outer_train)).astype(np.float32)
        M_outer_test = (~np.isnan(X_outer_test)).astype(np.float32)

        # DataLoader for efficient network feeding (training)
        X_train_dl = DataLoader(TensorDataset(
            torch.tensor(X_outer_train_sc_fill, dtype=torch.float32),
            torch.tensor(M_outer_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
        X_test_dl = DataLoader(TensorDataset(
            torch.tensor(X_outer_test_sc_fill, dtype=torch.float32),
            torch.tensor(M_outer_test, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)


        # 4.2) Create VAE MODEL and TRAIN it
        vae_model = mVAE(input_dim=X_outer_train_sc.shape[1], latent_dim=2, hidden_dims=hidden_innBest, lr=lr,
                         beta_end=beta_innBest, beta_warmup_steps=warmup_steps, missing_dropout_p=0.1, use_mask_in_encoder=False)


        v_tag = f"o{idx_outer_kf}-innBest_h{hiddendims[0]}_{hiddendims[1]}-b{beta_innBest}"
        logger = CSVLogger(vae_out_dir, name=None, version=v_tag)
        n_batch, wu_epochs = np.floor(len(X_inner_train) / batch_size), int(warmup_steps / len(X_train_dl))
        trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,  callbacks=[early],
                             accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                             enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=True)
        trainer.fit(vae_model, X_train_dl, X_test_dl)

        print(f"\tModel training ({round((time.time() - tic_out2) / 60, 2)}min)")

        # %% Training Assessment
        _, sc_temp_out, sc_temp_perdim_out = (
            assess_mVAE_fc(X_outer_test_sc_fill, M_outer_test, fnames, vae_model,
                           2, beta_innBest, hidden_innBest, None, idx_outer_kf, vae_out_dir, v_tag,
                           avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                           seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

        scores_out = pd.concat([scores_out, sc_temp_out])
        scores_perdim_out = pd.concat([scores_perdim_out, sc_temp_perdim_out])

        # Clean up
        del trainer, vae_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # # Plot assessment summary
    # plot_assessment([block], df_scores=scores,  df_stab=None,
    #                 select_vars=cfg_plot["select_vars"], select_down=cfg_plot["select_down"],
    #                 dir=vae_inn_dir, params=cfg_plot["params"],)

    # Save INNER results
    scores.to_csv(os.path.join(vae_inn_dir, "scores.csv"), index=False)
    scores_perdim.to_csv(os.path.join(vae_inn_dir, "scores_perdim.csv"), index=False)


    plt.close("all")  # por si algún fig quedó abierto
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Save OUTER results
    scores_out.to_csv(os.path.join(vae_out_dir,  f"scores_out.csv"), index=False)
    scores_perdim_out.to_csv(os.path.join(vae_out_dir, f"scores_perdim_out.csv"), index=False)


    print(f"\n\t\tnested-CV ended :: Time  {round((time.time() - tic0) / 60, 2)}min\n")

    return scores, scores_perdim




def fTrain_mVAE_fc(data,  zdims, beta, hdims, max_epochs=50, batch_size=30,
                      lr=1e-3, patience=10, delta=1e-3, beta_warmup=0.3, logs_dir=None,
                      cfg_ass=None):

    # Data
    fnames = data.columns.tolist()

    # Create aux dirs
    vae_fin_dir = os.path.join(logs_dir, f"VAE_final")
    os.makedirs(vae_fin_dir, exist_ok=True)


    # %% Hyper parameters (those depending on data structure): hiddendims, warm-up steps.
    warmup_steps = int((data.shape[0] / batch_size) * max_epochs * beta_warmup)


    tic0 = time.time()

    X_train, X_val = train_test_split(data.values, test_size=0.1, random_state=42, shuffle=True)


    # 1) Fit preprocessing ONLY on train
    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Fill NaNs with zero
    X_train_sc_fill = np.nan_to_num(X_train_sc, nan=0.0)
    X_val_sc_fill = np.nan_to_num(X_val_sc, nan=0.0)

    # Get the MASKS for NaNs
    M_train = (~np.isnan(X_train)).astype(np.float32)
    M_val = (~np.isnan(X_val)).astype(np.float32)

    # DataLoader for efficient network feeding (training)
    X_train_dl = DataLoader(TensorDataset(
        torch.tensor(X_train_sc_fill, dtype=torch.float32),
        torch.tensor(M_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=False)
    X_val_dl = DataLoader(TensorDataset(
        torch.tensor(X_val_sc_fill, dtype=torch.float32),
        torch.tensor(M_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)


    print(f"[FINAL] Final model h{hdims} - b{beta}  ::  Data ready. Training ...", end="\r")
    # %% 2) Create VAE MODEL and TRAIN it; mask in encoder.
    vae_model = mVAE(
        input_dim=X_train.shape[1], latent_dim=2, hidden_dims=hdims, lr=lr,
        beta_end=beta, beta_warmup_steps=warmup_steps, missing_dropout_p=0, use_mask_in_encoder=False)

    v_tag = f"h{hdims[0]}_{hdims[1]}-b{beta}"
    logger = CSVLogger(vae_fin_dir, name=None, version=v_tag)
    n_batch, wu_epochs = np.floor(len(X_train) / batch_size), int(warmup_steps / len(X_train_dl))
    # Training Callbacks :: stopping criteria. Include them in trainer.
    early = EarlyStopping(monitor="val_loss_fbeta", mode="min", patience=patience, min_delta=delta)  # or "val_recon"
    trainer = pl.Trainer(max_epochs=max_epochs, min_epochs=wu_epochs,   callbacks=[early],
                         accelerator="auto", devices="auto", log_every_n_steps=n_batch, logger=logger,
                         enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=True,)
    trainer.fit(vae_model, X_train_dl, X_val_dl)

    print(f"[final Training] :: done - time {round((time.time() - tic0) / 60, 2)} minutes")


    # %% Training Assessment of VAE on full data
    Z_mu, sc_temp, sc_temp_per_dim = (
        assess_mVAE_fc(X_val_sc_fill, M_val, fnames, vae_model,
                       zdims, beta, hdims, None, None, vae_fin_dir, v_tag,
                       avoid=cfg_ass["avoid"], kl_th=cfg_ass["kl_th"], tw_n=cfg_ass["tw_n"],
                       seed=cfg_ass["seed"], verbose=cfg_ass["verbose"]))

    # Save results
    sc_temp.to_csv(os.path.join(vae_fin_dir, "scores.csv"), index=False)
    sc_temp_per_dim.to_csv(os.path.join(vae_fin_dir, "scores_per_dim.csv"), index=False)


    # clean up
    plt.close("all")  # por si algún fig. quedó abierto
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[final Training] evaluation :: ended.  {round((time.time() - tic0) / 60, 2)} additional mins. ")

    return Z_mu, sc_temp, sc_temp_per_dim, vae_model # Z_ni, down_temp, down_temp_perdim, down_temp_curve


