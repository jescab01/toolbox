
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F



# %% VAE classes

class mVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        latent_dim=5,
        hidden_dims=(128, 64),
        lr=1e-3,
        beta_start=0,
        beta_end=1.0,
        beta_warmup_steps=1000,
        use_mask_in_encoder=True,
        missing_dropout_p=0.0,  # optional: additional random masking of observed entries
        var_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        enc_in_dim = input_dim * 2 if use_mask_in_encoder else input_dim

        if var_weights is None:
            var_weights = torch.ones(input_dim)
        self.register_buffer("w_j", var_weights.float())

        # --- Encoder ---
        layers = []
        last_dim = enc_in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.encoder = nn.Sequential(*layers)

        self.mu = nn.Linear(last_dim, latent_dim)
        self.logvar = nn.Linear(last_dim, latent_dim)

        # --- Decoder ---
        dec_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(last_dim, h))
            dec_layers.append(nn.ReLU())
            last_dim = h
        dec_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x, mask=None):
        if self.hparams.use_mask_in_encoder:
            if mask is None:
                raise ValueError("mask is required when use_mask_in_encoder=True")
            enc_in = torch.cat([x, mask], dim=-1)
        else:
            enc_in = x

        h = self.encoder(enc_in)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, mask=None):
        mu, logvar = self.encode(x, mask=mask)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def step(self, batch, stage="train"):
        x, mask = batch  # <- expects (x_filled, mask)

        # Optional: additionally drop some observed entries during training (denoising-style)
        if self.training and self.hparams.missing_dropout_p > 0:
            keep = (torch.rand_like(mask) > self.hparams.missing_dropout_p).float()
            mask_used = mask * keep
            x_used = x * keep  # keep same fill (0) for dropped entries
        else:
            x_used = x
            mask_used = mask

        x_recon, mu, logvar = self(x_used, mask=mask_used if self.hparams.use_mask_in_encoder else None)

        recon_loss = masked_weighted_mse(x_recon, x, mask, self.w_j)  # evaluate recon on *true observed* mask
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        beta_t = beta_schedule_linear(
            global_step=self.global_step,
            warmup_steps=self.hparams.beta_warmup_steps,
            beta_start=self.hparams.beta_start,
            beta_end=self.hparams.beta_end
        )

        loss = recon_loss + beta_t * kl

        fixed = recon_loss + self.hparams.beta_end * kl  # For early stopping criteria

        self.log(f"{stage}_beta", beta_t, on_step=False, on_epoch=True)
        self.log(f"{stage}_recon", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_kl", kl, on_step=False, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_loss_fbeta", fixed, on_step=False, on_epoch=True, prog_bar=True, logger=True) if stage == "val" else None

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class VAE_encoder_wrapper(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model


    def forward(self, x):
        mu, _ = self.vae.encode(x)
        return mu


class mVAE_mixed(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        numeric_idx,
        binary_idx,
        ordinal_idx,
        ordinal_n_classes,
        latent_dim=5,
        hidden_dims=(128, 64),
        lr=1e-3,
        beta_start=0.0,
        beta_end=1.0,
        beta_warmup_steps=1000,
        use_mask_in_encoder=True,
        missing_dropout_p=0.0,
        var_weights=None,
        alpha_num=1.0,
        alpha_bin=1.0,
        alpha_ord=1.0,
        binary_class_weights=None,      # NUEVO: [n_bin, 2]
        ordinal_class_weights=None,     # NUEVO: list[tensor(K_j)]
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["var_weights", "binary_class_weights", "ordinal_class_weights"])

        self.numeric_idx = numeric_idx
        self.binary_idx = binary_idx
        self.ordinal_idx = ordinal_idx
        self.ordinal_n_classes = ordinal_n_classes

        enc_in_dim = input_dim * 2 if use_mask_in_encoder else input_dim

        if var_weights is None:
            var_weights = torch.ones(input_dim)
        self.register_buffer("w_j", var_weights.float())

        # NUEVO: pesos binarios por valor observado
        if len(binary_idx) > 0:
            if binary_class_weights is None:
                binary_class_weights = torch.ones(len(binary_idx), 2, dtype=torch.float32)
            self.register_buffer(
                "binary_class_weights",
                torch.as_tensor(binary_class_weights, dtype=torch.float32)
            )
        else:
            self.binary_class_weights = None

        # NUEVO: pesos ordinales por categoría
        if ordinal_class_weights is None:
            ordinal_class_weights = [
                torch.ones(ncls, dtype=torch.float32)
                for ncls in ordinal_n_classes
            ]

        self.ordinal_class_weights = []
        for i, w in enumerate(ordinal_class_weights):
            w_tensor = torch.as_tensor(w, dtype=torch.float32)
            self.register_buffer(f"ordinal_class_weights_{i}", w_tensor)
            self.ordinal_class_weights.append(getattr(self, f"ordinal_class_weights_{i}"))

        # encoder
        layers = []
        last_dim = enc_in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(last_dim, latent_dim)
        self.logvar = nn.Linear(last_dim, latent_dim)

        # decoder backbone
        dec_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.decoder_backbone = nn.Sequential(*dec_layers)

        # numeric heads
        self.head_num_mu = nn.Linear(last_dim, len(numeric_idx)) if len(numeric_idx) > 0 else None

        # binary head
        self.head_bin = nn.Linear(last_dim, len(binary_idx)) if len(binary_idx) > 0 else None

        # ordinal heads
        self.heads_ord = nn.ModuleList([
            OrdinalLogisticHead(last_dim, ncls) for ncls in ordinal_n_classes
        ])

    def encode(self, x, mask=None):
        if self.hparams.use_mask_in_encoder:
            if mask is None:
                raise ValueError("mask is required when use_mask_in_encoder=True")
            x = torch.cat([x, mask], dim=-1)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_backbone(z)
        out = {
            "num_mu": self.head_num_mu(h) if self.head_num_mu is not None else None,
            "bin": self.head_bin(h) if self.head_bin is not None else None,
            "ord": [head(h) for head in self.heads_ord],   # list of (eta, tau)
        }
        return out

    def forward(self, x, mask=None):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

    def mixed_recon_loss(self, out, x, mask, x_ord, eps=1e-8):
        losses = {
            "num": torch.tensor(0.0, device=x.device),
            "bin": torch.tensor(0.0, device=x.device),
            "ord": torch.tensor(0.0, device=x.device),
        }

        # numeric
        if len(self.numeric_idx) > 0:
            idx = self.numeric_idx
            target = x[:, idx]
            m_t = mask[:, idx]
            w_t = self.w_j[idx].unsqueeze(0)

            mu = out["num_mu"]
            se = (mu - target) ** 2
            total_w = m_t * w_t
            losses["num"] = (se * total_w).sum() / total_w.sum().clamp_min(eps)

        # binary: BCE + peso por valor observado
        if len(self.binary_idx) > 0:
            idx = self.binary_idx
            target = x[:, idx]
            m_t = mask[:, idx]
            w_t = self.w_j[idx].unsqueeze(0)

            bce = F.binary_cross_entropy_with_logits(
                out["bin"],
                target,
                reduction="none"
            )  # [B, n_bin]

            w0 = self.binary_class_weights[:, 0].unsqueeze(0)   # [1, n_bin]
            w1 = self.binary_class_weights[:, 1].unsqueeze(0)   # [1, n_bin]
            class_w = (1.0 - target) * w0 + target * w1         # [B, n_bin]

            total_w = m_t * w_t * class_w
            losses["bin"] = (bce * total_w).sum() / total_w.sum().clamp_min(eps)

        # ordinal: NLL + peso por categoría observada
        if len(self.ordinal_idx) > 0:
            ord_num = torch.tensor(0.0, device=x.device)
            ord_den = torch.tensor(0.0, device=x.device)

            for k, idx in enumerate(self.ordinal_idx):
                eta, tau = out["ord"][k]
                target = x_ord[:, k].long()
                m_t = mask[:, idx]
                w_t = self.w_j[idx]
                class_w = self.ordinal_class_weights[k]

                nll = ordinal_logistic_nll_weighted(
                    eta=eta,
                    tau=tau,
                    target=target,
                    class_weights=class_w,
                    eps=eps
                )

                total_w = m_t * w_t
                ord_num = ord_num + (nll * total_w).sum()
                ord_den = ord_den + total_w.sum()

            losses["ord"] = ord_num / ord_den.clamp_min(eps)

        losses["total_unweighted"] = losses["num"] + losses["bin"] + losses["ord"]
        losses["total"] = (
            self.hparams.alpha_num * losses["num"] +
            self.hparams.alpha_bin * losses["bin"] +
            self.hparams.alpha_ord * losses["ord"]
        )
        return losses

    def step(self, batch, stage="train"):
        x, mask, x_ord = batch

        if self.training and self.hparams.missing_dropout_p > 0:
            keep = (torch.rand_like(mask) > self.hparams.missing_dropout_p).float()
            mask_used = mask * keep
            x_used = x * keep
        else:
            x_used = x
            mask_used = mask

        out, mu, logvar = self(x_used, mask_used if self.hparams.use_mask_in_encoder else None)
        recon = self.mixed_recon_loss(out, x, mask, x_ord)
        recon_loss = recon["total"]

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        beta_t = beta_schedule_linear(
            global_step=self.global_step,
            warmup_steps=self.hparams.beta_warmup_steps,
            beta_start=self.hparams.beta_start,
            beta_end=self.hparams.beta_end,
        )

        loss = recon_loss + beta_t * kl
        fixed = recon_loss + self.hparams.beta_end * kl

        self.log(f"{stage}_beta", beta_t, on_step=False, on_epoch=True)
        self.log(f"{stage}_recon", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_recon_num", recon["num"], on_step=False, on_epoch=True)
        self.log(f"{stage}_recon_bin", recon["bin"], on_step=False, on_epoch=True)
        self.log(f"{stage}_recon_ord", recon["ord"], on_step=False, on_epoch=True)
        self.log(f"{stage}_recon_unweighted", recon["total_unweighted"], on_step=False, on_epoch=True)
        self.log(f"{stage}_kl", kl, on_step=False, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if stage == "val":
            self.log(f"{stage}_loss_fbeta", fixed, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class VAE_encoder_wrapper_masked(torch.nn.Module):
    """
    Wrapper para explicar mu = encoder(x, mask), pero exponiendo a SHAP
    una sola entrada concatenada: [x, mask].
    """
    def __init__(self, vae_model, xdim):
        super().__init__()
        self.vae = vae_model
        self.xdim = int(xdim)

    def forward(self, xm):
        # xm: [batch, 2*xdim] = concat([x, mask], dim=1)
        x = xm[:, :self.xdim]
        m = xm[:, self.xdim:]
        mu, _ = self.vae.encode(x, m)
        return mu


# %% Aux for ordinal mixed VAE
class OrdinalLogisticHead(nn.Module):
    def __init__(self, in_dim, n_classes, init_gap=2.0):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        self.n_classes = n_classes
        self.score = nn.Linear(in_dim, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))

        # Inicialización más separada
        # Para K clases hay K-1 thresholds.
        # Ejemplo K=3 -> [-1, +1]
        init_tau = torch.linspace(-init_gap/2, init_gap/2, steps=n_classes - 1)
        self.raw_thresholds = nn.Parameter(init_tau.clone())

    def ordered_thresholds(self):
        # Forzamos orden estricto y separación positiva
        t0 = self.raw_thresholds[:1]
        if self.raw_thresholds.numel() == 1:
            return t0

        deltas = F.softplus(self.raw_thresholds[1:]) + 1e-3
        return torch.cat([t0, t0 + torch.cumsum(deltas, dim=0)], dim=0)

    def forward(self, h):
        scale = F.softplus(self.scale) + 1e-3
        eta = scale * self.score(h).squeeze(-1)   # (B,)
        tau = self.ordered_thresholds()   # (K-1,)
        return eta, tau


def ordinal_logistic_probs(eta, tau, eps=1e-8):
    """
    eta: (B,)
    tau: (K-1,)
    returns probs: (B, K)
    """
    cdf = torch.sigmoid(tau.unsqueeze(0) - eta.unsqueeze(1))   # (B, K-1)

    p_first = cdf[:, :1]
    p_middle = cdf[:, 1:] - cdf[:, :-1]
    p_last = 1.0 - cdf[:, -1:]

    probs = torch.cat([p_first, p_middle, p_last], dim=1)
    probs = probs.clamp_min(eps)

    # normalización por seguridad numérica
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def ordinal_logistic_predict(eta, tau):
    probs = ordinal_logistic_probs(eta, tau)
    return probs.argmax(dim=1)


def ordinal_logistic_nll(eta, tau, target, eps=1e-8):
    """
    eta:    (B,)
    tau:    (K-1,)
    target: (B,) integer labels 0..K-1
    """

    probs = ordinal_logistic_probs(eta, tau, eps=eps)
    logp = torch.log(probs)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return nll


def ordinal_logistic_nll_weighted(eta, tau, target, class_weights=None, eps=1e-8):
    """
    eta:    (B,)
    tau:    (K-1,)
    target: (B,) integer labels 0..K-1
    class_weights: (K,) or None
    """
    probs = ordinal_logistic_probs(eta, tau, eps=eps)
    logp = torch.log(probs)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)

    if class_weights is not None:
        sample_w = class_weights[target]
        nll = nll * sample_w

    return nll


def compute_binary_class_weights(X_bin, M_bin, max_weight=5.0, eps=1.0):
    """
    X_bin : [N, n_bin] con valores 0/1 (puede contener NaNs o fills)
    M_bin : [N, n_bin] máscara 1 si observado, 0 si missing

    Devuelve:
        class_weights: [n_bin, 2]
            [:, 0] = peso para valor 0
            [:, 1] = peso para valor 1

    Cuenta frecuencias SOLO sobre observados.
    """
    Xb = torch.as_tensor(X_bin, dtype=torch.float32)
    Mb = torch.as_tensor(M_bin, dtype=torch.float32)

    n_vars = Xb.shape[1]
    weights = []

    for j in range(n_vars):
        obs = Mb[:, j] > 0.5
        x_obs = Xb[obs, j]

        if x_obs.numel() == 0:
            # si no hay observaciones, no ponderar
            w0 = torch.tensor(1.0)
            w1 = torch.tensor(1.0)
        else:
            n_obs = float(x_obs.numel())
            n1 = float((x_obs == 1).sum())
            n0 = float((x_obs == 0).sum())

            w0 = n_obs / (2.0 * (n0 + eps))
            w1 = n_obs / (2.0 * (n1 + eps))

            w0 = torch.clamp(torch.tensor(w0), max=max_weight)
            w1 = torch.clamp(torch.tensor(w1), max=max_weight)

        weights.append(torch.stack([w0, w1]))

    return torch.stack(weights, dim=0)   # [n_bin, 2]


def compute_ordinal_class_weights(X_ord, M_ord, ordinal_n_classes, max_weight=5.0, eps=1.0):
    """
    X_ord : [N, n_ord] con categorías enteras 0..K-1 (puede contener NaNs o fills)
    M_ord : [N, n_ord] máscara 1 si observado, 0 si missing
    ordinal_n_classes : lista con K por variable ordinal

    Devuelve:
        lista de tensores, uno por variable ordinal
        cada tensor tiene shape [K_j]
    """
    Xo = torch.as_tensor(X_ord, dtype=torch.float32)
    Mo = torch.as_tensor(M_ord, dtype=torch.float32)

    weights = []

    for j, K in enumerate(ordinal_n_classes):
        obs = Mo[:, j] > 0.5
        x_obs = Xo[obs, j]

        if x_obs.numel() == 0:
            w = torch.ones(K, dtype=torch.float32)
        else:
            x_obs = x_obs.long()
            counts = torch.bincount(x_obs, minlength=K).float()
            n_obs = counts.sum()

            w = n_obs / (K * (counts + eps))
            w = torch.clamp(w, max=max_weight)

        weights.append(w)

    return weights


# %% Aux functions for mVAE

def beta_schedule_linear(global_step, warmup_steps, beta_start=0.0, beta_end=1.0):
    if warmup_steps <= 0:
        return beta_end
    t = min(global_step / warmup_steps, 1.0)
    return beta_start + t * (beta_end - beta_start)


def masked_weighted_mse(x_recon, x, mask, w_j, eps=1e-8):
    """
    x_recon, x, mask: (B, D)
    w_j: (D,)  (per-variable weights)
    """
    # broadcast weights to (B, D)
    w = w_j.unsqueeze(0)  # (1, D)

    se = (x_recon - x) ** 2
    num = (se * mask * w).sum()
    den = (mask * w).sum().clamp_min(eps)
    return num / den


def make_var_weights(p_j, w_max=3.0, eps=1e-6):
    w_j = 1.0 / torch.sqrt(torch.clamp(p_j, min=eps))
    w_j = torch.clamp(w_j, max=w_max)
    return w_j