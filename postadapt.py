# Adapted from Kishore4399/PostTrainingAdpt, condition-variance-cosine.
import os
import copy
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import wandb
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from pta_data import AgeDBPTADataModule
from conr_backbone import load_checkpoint, read_checkpoint
from source_contract import resolve_source_data, resolve_backbone
from dir_data import dataset_paths, check_data


def calculate_regression_metrics(
    y_true,
    y_pred,
    quantile_bins: int = 4,
) -> pd.Series:
 
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
 
    calibration_model = LinearRegression()
    calibration_model.fit(y_true.reshape(-1, 1), y_pred)
 
    slope = float(calibration_model.coef_[0])
    intercept = float(calibration_model.intercept_)
 
    slope_bias = abs(slope - 1.0)
    signed_slope_bias = slope - 1.0
 
    error = y_pred - y_true
    absolute_error = np.abs(error)
 
    eps = 1e-8
    gm = float(np.exp(np.mean(np.log(absolute_error + eps))))
 
    if np.std(y_pred, ddof=1) == 0 or np.std(error, ddof=1) == 0:
        pred_error_corr = np.nan
        pred_error_corr_pvalue = np.nan
    else:
        corr_result = pearsonr(y_pred, error)
        pred_error_corr = float(corr_result.statistic)
        pred_error_corr_pvalue = float(corr_result.pvalue)

    if np.std(y_true, ddof=1) == 0 or np.std(error, ddof=1) == 0:
        true_error_corr = np.nan
        true_error_corr_pvalue = np.nan
    else:
        corr_result = pearsonr(y_true, error)
        true_error_corr = float(corr_result.statistic)
        true_error_corr_pvalue = float(corr_result.pvalue)
 
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    overall_mpe = float(np.mean(error))

    q1 = float(np.quantile(y_true, 0.25))
    q3 = float(np.quantile(y_true, 0.75))
 
    lower_mask = y_true < q1
    upper_mask = y_true > q3
 
    lower_tail_mpe = (
        float(np.mean(error[lower_mask]))
        if lower_mask.any()
        else np.nan
    )
 
    upper_tail_mpe = (
        float(np.mean(error[upper_mask]))
        if upper_mask.any()
        else np.nan
    )

    quantile_codes, quantile_edges = pd.qcut(
        y_true,
        q=quantile_bins,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
 
    quantile_metrics = {}
 
    actual_num_bins = len(quantile_edges) - 1
 
    for bin_idx in range(actual_num_bins):
        mask = quantile_codes == bin_idx
 
        lower_bound = float(quantile_edges[bin_idx])
        upper_bound = float(quantile_edges[bin_idx + 1])
 
        bin_mae = (
            float(np.mean(absolute_error[mask]))
            if np.any(mask)
            else np.nan
        )
 
        # Include the right boundary for the final bin
        interval_symbol = "]" if bin_idx == actual_num_bins - 1 else ")"
 
        prefix = f"Quantile {bin_idx + 1}"
 
        quantile_metrics[f"{prefix} range"] = (
            f"[{lower_bound:.4f}, {upper_bound:.4f}{interval_symbol}"
        )
        quantile_metrics[f"{prefix} N"] = int(np.sum(mask))
        quantile_metrics[f"{prefix} MAE"] = bin_mae
 
    metrics = {
            "N": len(y_true),
            "Intercept": intercept,
            "Slope": slope,
            "Bias(Slope)": slope_bias,
            "Signed Bias(Slope)": signed_slope_bias,
            "Cor(y_pred, residual)": pred_error_corr,
            "Cor(y_pred, residual) p-value": pred_error_corr_pvalue,
            "Cor(y_true, residual)": true_error_corr,
            "Cor(y_true, residual) p-value": true_error_corr_pvalue,
            "RMSE": rmse,
            "MAE": mae,
            "GM": gm,
            "Overall MPE": overall_mpe,
            "Q1": q1,
            "Q3": q3,
            "Lower-tail N": int(lower_mask.sum()),
            "Lower-tail MPE (y_true < Q1)": lower_tail_mpe,
            "Upper-tail N": int(upper_mask.sum()),
            "Upper-tail MPE (y_true > Q3)": upper_tail_mpe,
            **quantile_metrics,
        }
 
    return metrics

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    
    if config['params'] is None:
        return cls()
    else:
        return cls(**config.get("params", dict()))

def plot_prediction_and_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    # Remove NaN / Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Prediction residual
    residual = y_pred - y_true

    reg = LinearRegression()
    reg.fit(y_true.reshape(-1, 1), y_pred)

    slope = reg.coef_[0]
    intercept = reg.intercept_

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Common x range
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    xx = np.linspace(vmin, vmax, 200)

    reg_res = LinearRegression()
    reg_res.fit(y_true.reshape(-1, 1), residual)

    residual_fit = reg_res.predict(xx.reshape(-1, 1))

    residual_slope = reg_res.coef_[0]
    residual_intercept = reg_res.intercept_

    residual_pred = reg_res.predict(y_true.reshape(-1, 1))
    residual_r2 = r2_score(residual, residual_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]

    ax.scatter(
        y_true,
        y_pred,
        s=18,
        alpha=0.6
    )

    ax.plot(
        xx,
        reg.predict(xx.reshape(-1, 1)),
        linewidth=2,
        label='Regression'
    )

    ax.plot(
        xx,
        xx,
        '--',
        linewidth=2,
        label='Ideal (y=x)'
    )

    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title('Prediction vs Ground Truth')

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')

    ax.grid(True, alpha=0.3)

    text = (
        f'R² = {r2:.3f}\n'
        f'MAE = {mae:.3f}\n'
        f'slope = {slope:.3f}\n'
        f'intercept = {intercept:.3f}'
    )

    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.85,
            edgecolor='gray'
        )
    )

    ax.legend(frameon=False)


    ax = axes[1]
    ax.scatter(
        y_true,
        residual,
        s=18,
        alpha=0.6
    )

    ax.plot(
        xx,
        residual_fit,
        linewidth=2,
        label='Trend'
    )

    ax.axhline(
        0,
        linestyle='--',
        linewidth=2,
        label='Zero Error'
    )

    ax.set_xlabel('True Age')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Residual vs True')

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(-40, 40)

    ax.grid(True, alpha=0.3)

    text_res = (
        f'R² = {residual_r2:.3f}\n'
        f'slope = {residual_slope:.3f}\n'
        f'intercept = {residual_intercept:.3f}'
    )

    ax.text(
        0.98,
        0.02,
        text_res,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.85,
        
            edgecolor='gray'
        )
    )

    ax.set_xlabel('True Age')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Residual vs True')

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    
    return fig


EPS = 1e-8
COND_LOSSES = ("w2", "l1", "kl", "skl")
# kl / skl divide by sigma^2, so a near-zero kernel std (an anchor with no label neighbours in the
# batch) makes them explode. These defaults (label units / neighbour count) stabilise them.
KL_LOSSES = ("kl", "skl")
DEFAULT_SIGMA_FLOOR = 1.0
DEFAULT_MIN_NEIGHBORS = 3.0
FROZEN_PREFIXES = ("src_encoder.", "src_fc.")  # frozen source copy, left out of best-model files


def l2sp_penalty(model, ref_model):
    """L2-SP regularization (Li, Grandvalet & Davoine, ICML 2018): ||w - w0||^2.

    Squared L2 distance between the trainable parameters of ``model`` and the matching
    parameters of ``ref_model`` (the frozen pretrained copy). Pulls weights toward their
    pretrained values instead of toward zero; use it with the optimizer's ``weight_decay=0``.
    """
    penalty = 0.0
    for p, p0 in zip(model.parameters(), ref_model.parameters()):
        if p.requires_grad:
            penalty = penalty + (p.float() - p0.detach().float()).pow(2).sum()
    return penalty


def fit_label_bias(y_true, y_pred):
    """OLS fit of the source error trend: (y_pred - y_true) = beta1 * y_true + beta2.

    The regressor is the ground-truth label. Computed in float64. Returns ``beta1``, ``beta2``,
    ``n``, the error-label correlation and the fitted error slope before (``slope_original``)
    and after (``slope_corrected``, ~0) the correction.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    e = np.asarray(y_pred, dtype=np.float64).ravel() - y
    if y.size < 2 or np.var(y) == 0:
        raise ValueError("Need at least two samples with distinct labels to fit the source bias.")
    yc = y - y.mean()
    beta1 = float(np.sum(yc * (e - e.mean())) / np.sum(yc * yc))
    beta2 = float(e.mean() - beta1 * y.mean())
    e_c = e - (beta1 * y + beta2)
    return {
        "beta1": beta1,
        "beta2": beta2,
        "n": int(y.size),
        "error_label_corr": float(np.corrcoef(e, y)[0, 1]) if np.std(e) > 0 else float("nan"),
        "slope_original": beta1,
        "slope_corrected": float(np.sum(yc * (e_c - e_c.mean())) / np.sum(yc * yc)),
    }


def label_kernel(y, bandwidth):
    """Normalized Gaussian label kernel: row i weights the batch by similarity to y_i; rows sum to 1."""
    with torch.autocast(device_type=y.device.type, enabled=False):
        y = y.float()
        return torch.softmax(-(y[:, None] - y[None, :]).square() / (2 * bandwidth ** 2), dim=1)


def kernel_conditional_stats(weights, errors):
    """Kernel-weighted conditional mean and variance of errors, in fp32 (autocast off).

    mu_i  = sum_j w_ij e_j
    var_i = sum_j w_ij (e_j - mu_i)^2   (centered form: no cancellation, never negative)
    """
    with torch.autocast(device_type=weights.device.type, enabled=False):
        w, e = weights.float(), errors.float()
        mu = w @ e
        var = (w * (e[None, :] - mu[:, None]).square()).sum(dim=1).clamp_min(0)
    return mu, var


def effective_neighbors(weights):
    """Kish effective sample size of each kernel row, 1 / sum_j w_ij^2 (1 = only the anchor itself)."""
    with torch.autocast(device_type=weights.device.type, enabled=False):
        return 1.0 / weights.float().square().sum(dim=1)


def gaussian_matching_loss(mu, sigma, sigma_ref, kind="w2", mask=None):
    """Distance between each anchor's conditional error distribution N(mu_i, sigma_i^2) and the
    target N(0, sigma_ref_i^2): no bias, and the spread of the bias-corrected source model.
    Averaged over anchors (only those where ``mask`` is True, if given); ``sigma_ref`` receives
    no gradient. Computed in fp32.

    w2: squared 2-Wasserstein distance, mu^2 + (sigma - sigma_ref)^2
    l1: |mu| + |sigma - sigma_ref|
    kl: KL(N(mu, sigma^2) || N(0, sigma_ref^2)) = log(sigma_ref / sigma) + (sigma^2 + mu^2) / (2 sigma_ref^2) - 1/2
    skl: symmetric KL, the average of both directions (the log terms cancel):
         (KL(P || Q) + KL(Q || P)) / 2 = (sigma^2 + mu^2) / (4 sigma_ref^2) + (sigma_ref^2 + mu^2) / (4 sigma^2) - 1/2
    """
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu, sigma, sigma_ref = mu.float(), sigma.float(), sigma_ref.detach().float()
        if kind == "w2":
            d = mu.square() + (sigma - sigma_ref).square()
        elif kind == "l1":
            d = mu.abs() + (sigma - sigma_ref).abs()
        elif kind == "kl":
            d = torch.log(sigma_ref / sigma) + (sigma.square() + mu.square()) / (2 * sigma_ref.square()) - 0.5
        elif kind == "skl":
            d = ((sigma.square() + mu.square()) / (4 * sigma_ref.square())
                 + (sigma_ref.square() + mu.square()) / (4 * sigma.square()) - 0.5)
        else:
            raise ValueError(f"Unknown condLoss {kind!r}; choose one of {COND_LOSSES}")
        if mask is not None:
            m = mask.to(d.dtype)
            return (d * m).sum() / m.sum().clamp_min(1.0)   # 0 (with zero grad) if no anchor qualifies
        return d.mean()


def delta_correlation(y_true, y_pred):
    """Pearson correlation between the delta (y_pred - y_true) and y_true (DC; ADC for age).

    Negative values mean regression to the mean. NaN when either input has zero variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    delta = y_pred - y_true
    if np.std(delta) == 0 or np.std(y_true) == 0:
        return float("nan")
    return float(np.corrcoef(delta, y_true)[0, 1])


class ValDCTopKCheckpoint(pl.Callback):
    """Model selection on the validation set.

    After every validation epoch (from ``save_start`` on) the ``top_k`` epochs with the highest
    (signed) validation DC are kept as candidate files; ties go to the lower val MAE. DC =
    corr(y_pred - y_true, y_true) is usually negative (regression to the mean), so higher means
    less bias, and positive values are allowed. An epoch that beats the worst kept one replaces
    it. At the end of training the selected model is the candidate with the lowest validation
    MAE (``best``); the test set plays no part in the choice.

    Candidate names are ``filename`` formatted with ``epoch``, ``val_dc``, ``val_abs_dc``,
    ``val_mae``, ``test_dc``, ``test_abs_dc`` and ``test_mae``. Files hold the weights only (no
    optimizer state, no frozen source copy; the source bias coefficients are kept). ``save_best``
    copies the selected one to ``best-model-epochXX-valMAE...-valDC....ckpt``, the name other
    scripts look for.

    ``max_abs_dc`` (optional, e.g. 0.05) adds a constraint: epochs with |val DC| <= max_abs_dc
    always rank above epochs outside it, so once enough qualifying epochs exist the top-k holds
    only those, and the selected model is the lowest-MAE qualifying candidate. If no epoch ever
    meets the limit, selection falls back to the unconstrained rule (``selection`` says so).
    """

    METRICS = ("val_mae", "val_dc", "val_abs_dc")          # required: selection uses validation only
    TEST_METRICS = ("test_mae", "test_dc", "test_abs_dc")  # optional (logger.eval_test_each_epoch); NaN if off

    def __init__(self, dirpath, filename="topk-epoch{epoch:02d}-valDC{val_dc:+.4f}-valMAE{val_mae:.3f}",
                 top_k=5, save_start=0, max_abs_dc=None):
        super().__init__()
        if str(filename).startswith("best-model"):
            raise ValueError("logger.save_name names the top-k candidates; 'best-model*' is reserved for the "
                             "selected model (e.g. use topk-epoch{epoch:02d}-valDC{val_dc:+.4f}-valMAE{val_mae:.3f})")
        if int(top_k) < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.dirpath = str(dirpath)
        self.filename = filename
        self.top_k = int(top_k)
        self.save_start = int(save_start)
        self.max_abs_dc = None if max_abs_dc is None else float(max_abs_dc)
        if self.max_abs_dc is not None and self.max_abs_dc < 0:
            raise ValueError(f"max_abs_dc must be >= 0, got {max_abs_dc}")
        self.kept = []   # one dict per kept epoch: path, epoch and METRICS

    def _within_dc(self, entry):
        return self.max_abs_dc is None or abs(entry["val_dc"]) <= self.max_abs_dc

    def _rank(self, entry):
        """Sort key, lower = better candidate: within the |DC| limit first, then higher val DC,
        then lower val MAE."""
        return (not self._within_dc(entry), -entry["val_dc"], entry["val_mae"])

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.current_epoch < self.save_start:
            return
        metrics = trainer.callback_metrics
        if not all(k in metrics for k in self.METRICS):
            return
        fields = {k: float(metrics[k]) for k in self.METRICS}
        fields.update({k: float(metrics[k]) if k in metrics else float("nan") for k in self.TEST_METRICS})
        if not (np.isfinite(fields["val_dc"]) and np.isfinite(fields["val_mae"])):
            return
        entry = {"epoch": trainer.current_epoch, **fields}
        if len(self.kept) >= self.top_k and self._rank(entry) >= self._rank(max(self.kept, key=self._rank)):
            return

        entry["path"] = os.path.join(self.dirpath, self.filename.format(**entry) + ".ckpt")
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
            weights = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()
                       if not k.startswith(FROZEN_PREFIXES)}
            torch.save({"state_dict": weights, **entry}, entry["path"])
        self.kept = [e for e in self.kept if e["path"] != entry["path"]] + [entry]
        if len(self.kept) > self.top_k:
            worst = max(self.kept, key=self._rank)
            self.kept.remove(worst)
            if trainer.is_global_zero and os.path.exists(worst["path"]):
                os.remove(worst["path"])

    @property
    def best(self):
        """The selected candidate: lowest val MAE among the kept epochs that meet the |DC| limit,
        or among all kept epochs if none does (None if nothing was kept)."""
        pool = [e for e in self.kept if self._within_dc(e)] or self.kept
        return min(pool, key=lambda e: (e["val_mae"], -e["val_dc"])) if pool else None

    @property
    def best_model_path(self):
        return self.best["path"] if self.best else None

    @property
    def best_epoch(self):
        return self.best["epoch"] if self.best else None

    @property
    def selection(self):
        base = f"lowest val MAE among the {self.top_k} epochs with the highest val DC"
        if self.max_abs_dc is None:
            return base
        if any(self._within_dc(e) for e in self.kept):
            return f"{base}, restricted to |val DC| <= {self.max_abs_dc}"
        return f"{base} (WARNING: no epoch reached |val DC| <= {self.max_abs_dc}; constraint ignored)"

    def save_best(self):
        """Copy the selected candidate to best-model-epochXX-valMAE...-valAbsDC....ckpt (only one kept)."""
        import shutil
        best = self.best
        if best is None:
            return None
        for old in Path(self.dirpath).glob("best-model-*.ckpt"):
            old.unlink()
        path = os.path.join(self.dirpath, "best-model-epoch{epoch:02d}-valMAE{val_mae:.3f}-valDC{val_dc:+.4f}.ckpt"
                            .format(**best))
        shutil.copyfile(best["path"], path)
        return path

    def candidates_table(self):
        rows = sorted(self.kept, key=self._rank)
        best = self.best
        return pd.DataFrame([{**{k: e.get(k, float("nan")) for k in ("epoch", "val_dc", "val_mae", "test_dc", "test_mae")},
                              "within_dc_limit": self._within_dc(e),
                              "selected": e is best, "checkpoint": os.path.basename(e["path"])} for e in rows])

    def state_dict(self):
        return {"kept": [dict(e) for e in self.kept]}

    def load_state_dict(self, state_dict):
        # candidates saved under the old |DC| rule have no signed val_dc: they are dropped
        self.kept = [dict(e) for e in state_dict.get("kept", []) if "val_dc" in e]


class PostTrainAdaptation(pl.LightningModule):
    """Adapt a pretrained regressor so its errors are unbiased given the label.

    Loss = condLoss(N(mu_t, sigma_t^2), N(0, sigma_ref^2))  conditional error distribution
         + l2spWeight * ||w - w0||^2                        stay close to the pretrained weights

    mu_t, sigma_t: kernel-weighted conditional mean/std of the student's errors around each label.
    sigma_ref: the same std for the frozen source model after removing its linear error trend
    beta1 * y + beta2 (fitted once on the adaptation split before training).
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        _, checkpoint_name = read_checkpoint(self.hparams["pretrained"])
        params = self.hparams["backbone"]["params"]
        params["name"] = resolve_backbone(params.get("name", "auto"), checkpoint_name)
        self.encoder = instantiate_from_config(self.hparams["backbone"])
        # Feature size: model.feature_dim if given, else from the backbone (model.Encoder sets .dim)
        enc_dim = getattr(self.encoder, "dim", None)
        feature_dim = self.hparams.get("feature_dim") or enc_dim
        if feature_dim is None:
            raise ValueError("Set model.feature_dim: the backbone does not report its feature size")
        if enc_dim is not None and int(feature_dim) != int(enc_dim):
            raise ValueError(f"model.feature_dim={feature_dim} but backbone "
                             f"{self.hparams['backbone']['params']} outputs {enc_dim} features")
        self.fc = nn.Linear(int(feature_dim), 1)
        self.validation_outputs = []
        self.val_slope = float("nan")   # slope of (y_pred - y_true) on y_true, last validation
        self.save_train_predictions = True  # also predict the source 'train' split (logger.save_train_predictions)
        self.eval_test_each_epoch = False   # test metrics after every validation epoch (logger.eval_test_each_epoch)

        optimizer_name(self.hparams)   # fail early on an unknown model.optim.name
        self.cond_loss = str(self.hparams.get("condLoss", "w2")).lower()
        if self.cond_loss not in COND_LOSSES:
            raise ValueError(f"Unknown condLoss {self.cond_loss!r}; choose one of {COND_LOSSES}")
        # Stabilisers for kl / skl (ignored by w2 / l1, whose behaviour is unchanged):
        #   sigmaFloor:   sigma = sqrt(var + sigmaFloor^2), in label units
        #   minNeighbors: anchors whose kernel row has fewer effective samples are skipped
        is_kl = self.cond_loss in KL_LOSSES
        self.sigma_floor = float(self.hparams.get("sigmaFloor") or DEFAULT_SIGMA_FLOOR) if is_kl else 0.0
        self.min_neighbors = float(self.hparams.get("minNeighbors") or DEFAULT_MIN_NEIGHBORS) if is_kl else 0.0

        self.load_pretrained_weights()

        # Trainable parameters (the regression head stays frozen unless "all"; L2-SP then covers it too):
        #   backbone:      all encoder weights
        #   backbone_bn:   BatchNorm affine parameters only (ResNets)
        #   backbone_norm: BatchNorm and LayerNorm affine parameters (ResNets and ViTs)
        #   all:           encoder and regression head (default)
        mode = self.hparams["optim"]["param"]
        norm_types = {"backbone_bn": (_BatchNorm,), "backbone_norm": (_BatchNorm, nn.LayerNorm, nn.GroupNorm)}
        self.requires_grad_(False)
        if mode == "backbone":
            self.encoder.requires_grad_(True)
        elif mode == "all":
            self.requires_grad_(True)
        elif mode in norm_types:
            for m in self.encoder.modules():
                if isinstance(m, norm_types[mode]) and getattr(m, "weight", None) is not None:
                    m.requires_grad_(True)
        else:
            raise ValueError(f"Unknown optim.param: {mode} (backbone, backbone_bn, backbone_norm or all)")
        if not any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError(f"optim.param={mode} leaves nothing to train in this backbone "
                             f"(ViTs have LayerNorm, not BatchNorm: use backbone_norm or backbone)")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(name, p.shape)

        # Frozen source encoder (shares the frozen head self.fc)
        self.src_fc = copy.deepcopy(self.fc).eval()
        self.src_fc.requires_grad_(False)
        self.src_encoder = copy.deepcopy(self.encoder)
        self.src_encoder.eval()
        self.src_encoder.requires_grad_(False)

        # [beta1, beta2] of error_src = beta1 * y + beta2, fitted once before training.
        # Buffers: saved in checkpoints, restored on resume, moved with the module.
        self.register_buffer("src_bias_beta", torch.zeros(2))
        self.register_buffer("src_bias_fitted", torch.zeros((), dtype=torch.bool))

    def forward(self, x):
        return self.fc(self.encoder(x)).squeeze(-1)

    @torch.no_grad()
    def collect_source_predictions(self, dataloader, desc="Source predictions"):
        """Labels and frozen source-model predictions over a dataloader (eval mode, no gradients)."""
        self.src_encoder.eval()
        self.src_fc.eval()
        y_true, y_pred = [], []
        for batch in tqdm(dataloader, desc=desc):
            x = batch["image"].to(self.device, non_blocking=True)
            y_pred.append(self.src_fc(self.src_encoder(x)).flatten().float().cpu())
            y_true.append(torch.as_tensor(batch["label"]).flatten().float())
        return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()

    def setup_source_bias_correction(self):
        """Fit beta1, beta2 once on the adaptation set, or keep them if restored from a checkpoint."""
        if bool(self.src_bias_fitted):
            beta1, beta2 = self.src_bias_beta.tolist()
            print(f"Source bias correction restored from checkpoint: beta1 {beta1:+.5f}  beta2 {beta2:+.4f}")
            return

        # Adaptation split only, each sample once, no augmentation; the test split is never read
        y_true, y_pred = self.collect_source_predictions(self.trainer.datamodule.adaptation_dataloader())
        stats = fit_label_bias(y_true, y_pred)

        beta = torch.tensor([stats["beta1"], stats["beta2"]], dtype=torch.float32)
        beta = self.trainer.strategy.broadcast(beta, src=0)
        self.src_bias_beta.copy_(beta.to(self.src_bias_beta.device))
        self.src_bias_fitted.fill_(True)

        stats["adaptation_split"] = self.trainer.datamodule.train_dataset.adpt_split
        if self.trainer.is_global_zero:
            with open(os.path.join(self.logger.save_dir, "source_bias_correction.json"), "w") as f:
                json.dump(stats, f, indent=2)
        self.logger.log_metrics(
            {f"src_bias/{k}": v for k, v in stats.items() if isinstance(v, (int, float))},
            step=self.global_step)
        print(f"Source bias fit on {stats['n']} '{stats['adaptation_split']}' samples: "
              f"beta1 {stats['beta1']:+.5f}  beta2 {stats['beta2']:+.4f}  "
              f"corr(error, y) {stats['error_label_corr']:+.3f}  "
              f"slope {stats['slope_original']:+.4f} -> {stats['slope_corrected']:+.2e}")

    def on_fit_start(self):
        self.setup_source_bias_correction()
        # *_before.csv: the frozen source model (src_encoder + fc), i.e. the pretrained model.
        # Uses src_encoder explicitly so the files stay correct even if self.encoder was
        # already changed (e.g. weights restored from a checkpoint before this hook).
        dm = self.trainer.datamodule
        loaders = {"val": dm.val_dataloader(), "test": dm.test_dataloader()}
        if self.save_train_predictions:
            loaders["train"] = dm.train_eval_dataloader()
        for split, loader in loaders.items():
            y_true, y_pred = self.collect_source_predictions(loader, desc=f"{split.capitalize()} (pretrained)")
            self.save_predictions_to_csv(y_true, y_pred, f"{split}_before.csv")

    def on_fit_end(self):
        device = self.trainer.strategy.root_device
        self.to(device=device)

        # Predictions of the final-epoch weights
        val_true, val_pred = self.get_validation_prediction()
        test_true, test_pred = self.get_test_prediction()
        self.save_predictions_to_csv(val_true, val_pred, "val_last.csv")
        self.save_predictions_to_csv(test_true, test_pred, "test_last.csv")

        # Load the selected model: lowest val MAE among the top-k epochs by val DC (higher is better)
        ckpt_cb = next((c for c in self.trainer.callbacks if isinstance(c, ValDCTopKCheckpoint)), None)
        best_path = ckpt_cb.best_model_path if ckpt_cb is not None else None
        if best_path is not None:
            state = torch.load(best_path, map_location="cpu", weights_only=False)
            msg = self.load_state_dict(state["state_dict"], strict=False)
            bad = [k for k in msg.missing_keys if not k.startswith(FROZEN_PREFIXES)] + list(msg.unexpected_keys)
            if bad:
                raise RuntimeError(f"Best checkpoint does not match the model: {bad[:10]}")
            self.to(device=device)
            selection, epoch = ckpt_cb.selection, ckpt_cb.best_epoch
            table = ckpt_cb.candidates_table()
            if self.trainer.is_global_zero:
                best_path = ckpt_cb.save_best()
                table.to_csv(os.path.join(self.logger.save_dir, "topk_candidates.csv"), index=False)
            print(f"Top-{ckpt_cb.top_k} epochs by val DC:\n{table.to_string(index=False)}")
            print(f"Loaded best model ({selection}, epoch {epoch}): {best_path}")
        else:
            selection = "last epoch (no epoch was evaluated after logger.save_start)"
            epoch = self.current_epoch
            print(f"{selection}: *_after.csv use the last-epoch weights.")

        # *_after.csv: the selected best model (read by main_lassoUp.py)
        val_true, val_pred = self.get_validation_prediction()
        test_true, test_pred = self.get_test_prediction()
        self.save_predictions_to_csv(val_true, val_pred, "val_after.csv")
        self.save_predictions_to_csv(test_true, test_pred, "test_after.csv")
        splits = {"val": (val_true, val_pred), "test": (test_true, test_pred)}
        if self.save_train_predictions:
            train_true, train_pred = self.get_train_prediction()
            self.save_predictions_to_csv(train_true, train_pred, "train_after.csv")
            splits["train"] = (train_true, train_pred)

        summary = {split: {"Selection": selection, "Selected epoch": epoch, "Checkpoint": best_path,
                           "DC": delta_correlation(t, p), **calculate_regression_metrics(t, p)}
                   for split, (t, p) in splits.items()}
        pd.DataFrame(summary).to_csv(os.path.join(self.logger.save_dir, "metrics_best.csv"))
        print(f"Best model  val MAE {summary['val']['MAE']:.3f}  DC {summary['val']['DC']:+.3f} | "
              f"test MAE {summary['test']['MAE']:.3f}  DC {summary['test']['DC']:+.3f}")

    def training_step(self, batch, batch_idx):
        # BatchNorm in eval mode for both encoders: running statistics are never updated
        # (Lightning puts every submodule back in train mode after each validation)
        self.encoder.eval()
        self.src_encoder.eval()
        self.src_fc.eval()

        x, y = batch["image"], batch["label"].flatten().float()
        with torch.no_grad():
            src_feat = self.src_encoder(x)
        feat = self.encoder(x)
        error_t = self.fc(feat).flatten().float() - y

        # Conditional error distribution of the student vs the bias-corrected source
        weights = label_kernel(y, float(self.hparams["bandwidth"]))
        mu_t, var_t = kernel_conditional_stats(weights, error_t)
        with torch.no_grad():
            beta1, beta2 = self.src_bias_beta
            error_s = self.src_fc(src_feat).flatten().float() - y
            _, var_ref = kernel_conditional_stats(weights, error_s - (beta1 * y + beta2))
        floor2 = max(self.sigma_floor ** 2, EPS)
        sigma_t, sigma_ref = (var_t + floor2).sqrt(), (var_ref + floor2).sqrt()
        mask = effective_neighbors(weights) >= self.min_neighbors if self.min_neighbors > 0 else None
        cond_loss = gaussian_matching_loss(mu_t, sigma_t, sigma_ref, self.cond_loss, mask=mask)

        # The head term is zero unless optim.param=all trains the head (only trainable params count).
        l2sp = l2sp_penalty(self.encoder, self.src_encoder) + l2sp_penalty(self.fc, self.src_fc)

        loss = cond_loss + self.hparams["l2spWeight"] * l2sp

        self.log_dict({
            "lr": self.optimizers().param_groups[0]["lr"],
            "cond_loss": cond_loss,
            "l2sp": l2sp,
            "cond_mean_abs": mu_t.detach().abs().mean(),                   # |conditional bias|
            "cond_std_gap": (sigma_t - sigma_ref).detach().abs().mean(),   # |sigma_t - sigma_ref|
            "slope": self.val_slope,
            **({"cond_anchor_frac": mask.float().mean()} if mask is not None else {}),
        }, prog_bar=True)
        return loss

    def validation_step(self, batch):
        y_pred = self(batch["image"]).float().cpu()
        y_true = batch["label"].float().cpu()
        outputs = {"y_true": y_true, "y_pred": y_pred}
        self.validation_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        y_true = torch.cat([x["y_true"] for x in self.validation_outputs]).flatten().numpy()
        y_pred = torch.cat([x["y_pred"] for x in self.validation_outputs]).flatten().numpy()
        self.validation_outputs.clear()

        reg_fig = plot_prediction_and_bias(y_true, y_pred)
        self.logger.experiment.log({f"Val epoch_{self.current_epoch:04d}": wandb.Image(reg_fig)})
        plt.close(reg_fig)

        results = calculate_regression_metrics(y_true, y_pred)
        self.val_slope = results["Signed Bias(Slope)"]
        val_dc = delta_correlation(y_true, y_pred)
        self.log("val_mae", results["MAE"], prog_bar=True)
        self.log("val_dc", val_dc, prog_bar=True)
        self.log("val_abs_dc", abs(val_dc) if np.isfinite(val_dc) else float("inf"))

        # Test metrics each epoch (logger.eval_test_each_epoch): only logged and listed for the top-k
        # candidates, never used for selection. test_before/after.csv are saved either way.
        if not self.eval_test_each_epoch:
            return
        test_true, test_pred = self.get_test_prediction()
        reg_fig = plot_prediction_and_bias(test_true, test_pred)
        self.logger.experiment.log({f"Test epoch_{self.current_epoch:04d}": wandb.Image(reg_fig)})
        plt.close(reg_fig)
        test_dc = delta_correlation(test_true, test_pred)
        self.log("test_mae", float(mean_absolute_error(test_true, test_pred)), prog_bar=True)
        self.log("test_dc", test_dc, prog_bar=True)
        self.log("test_abs_dc", abs(test_dc) if np.isfinite(test_dc) else float("inf"))

    def save_predictions_to_csv(self, y_true, y_pred, filename):
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            os.path.join(self.logger.save_dir, filename), index=False)

    @torch.no_grad()
    def predict_loader(self, loader, desc):
        self.encoder.eval()
        y_true, y_pred = [], []
        for batch in tqdm(loader, desc=desc):
            y_pred.extend(self(batch["image"].to(self.device)).cpu().tolist())
            y_true.extend(batch["label"].flatten().tolist())
        self.encoder.train()
        return np.array(y_true), np.array(y_pred)

    def get_test_prediction(self):
        return self.predict_loader(self.trainer.datamodule.test_dataloader(), "Test")

    def get_train_prediction(self):
        return self.predict_loader(self.trainer.datamodule.train_eval_dataloader(), "Train")

    def get_validation_prediction(self):
        return self.predict_loader(self.trainer.datamodule.val_dataloader(), "Validation")

    def load_pretrained_weights(self):
        load_checkpoint(self.encoder, self.fc, self.hparams['pretrained'])

    def configure_optimizers(self):
        cfg = self.hparams["optim"]
        params = [p for p in self.parameters() if p.requires_grad]
        name = optimizer_name(self.hparams)
        if name == "sgd":
            optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
        elif name == "adam":
            optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:  # adamw
            optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = {
            "scheduler": WarmupCosineSchedule(
                optimizer=optimizer,
                warmup_steps=0,
                t_total=self.trainer.estimated_stepping_batches,
                end_lr=1e-6,
                cycles=0.5,
                warmup_multiplier=0,
            ),
            "interval": "step",
            "frequency": 1,
            "name": "lr",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


OPTIMIZERS = ("adam", "sgd", "adamw")


def optimizer_name(model_cfg):
    """model.optim.name: adam (default) | sgd (momentum 0.9) | adamw."""
    name = str(model_cfg["optim"].get("name") or "adam").lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown model.optim.name {name!r}; choose one of {OPTIMIZERS}")
    return name


def run_name(model_cfg, batch_size):
    """Output folder name: the settings that change the result (batch_size is dataset.batch_size)."""
    kind = str(model_cfg.get('condLoss', 'w2')).lower()
    kl_tag = (f"_sf:{model_cfg.get('sigmaFloor') or DEFAULT_SIGMA_FLOOR}"
              f"_mn:{model_cfg.get('minNeighbors') or DEFAULT_MIN_NEIGHBORS}") if kind in KL_LOSSES else ""
    return (f"PostTrainAdpt_{kind}"
            f"_h:{model_cfg['bandwidth']}{kl_tag}"
            f"_l2spWeight:{model_cfg['l2spWeight']}"
            f"_lr:{model_cfg['optim']['lr']}"
            f"_bsz:{batch_size}"
            f"_{model_cfg['optim']['param']}"
            f"_{optimizer_name(model_cfg)}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args, overrides = parser.parse_known_args()

    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    config.dataset = resolve_source_data(config.dataset, config.model.pretrained)
    pl.seed_everything(int(config.get('seed', 0)), workers=True)
    dataset = config.dataset.pop('name')
    images, manifest = dataset_paths(dataset, config.dataset.data_folder, config.dataset.data_csv)
    config.dataset.data_folder = str(images)
    config.dataset.data_csv = str(manifest)
    check_data(images, manifest, config.dataset.label_name)
    if config.trainer.devices != 1:
        raise ValueError("Post-adaptation currently supports one device")
    if config.dataset.adaptation_split == 'test':
        raise ValueError("Choose a labeled adaptation split other than test")
    if config.dataset.batch_size < 2:
        raise ValueError("dataset.batch_size must be at least 2 for conditional statistics")
    if float(config.model.bandwidth) <= 0:
        raise ValueError("model.bandwidth must be positive")
    print(OmegaConf.to_yaml(config))

    # Results go next to the pretrained checkpoint
    folder = run_name(config['model'], config['dataset']['batch_size'])
    balance = config['dataset'].get('balance') or {}
    if balance.get('enabled', False):   # unbalanced runs keep their old folder names
        folder += f"_balance=a{balance.get('alpha', 1.0)}b{balance.get('num_bins', 10)}"
    save_dir = (Path(config['logger']['save_dir']).expanduser().resolve()
                if config['logger'].get('save_dir')
                else Path(config['model']['pretrained']).resolve().parent / folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    config['logger']['save_dir'] = str(save_dir)
    with open(save_dir / 'source_data_config.json', 'w') as handle:
        json.dump({"name": dataset, **OmegaConf.to_container(config.dataset, resolve=True)}, handle, indent=2)

    data_module = AgeDBPTADataModule(**config['dataset'])
    model = PostTrainAdaptation(config['model'])
    model.save_train_predictions = bool(config['logger'].get('save_train_predictions', True))
    model.eval_test_each_epoch = bool(config['logger'].get('eval_test_each_epoch', False))

    ckpt_dir = str(save_dir / 'checkpoints')
    best_checkpoint_callback = ValDCTopKCheckpoint(
        dirpath=ckpt_dir,
        filename=config['logger'].get('save_name') or 'topk-epoch{epoch:02d}-valDC{val_dc:+.4f}-valMAE{val_mae:.3f}',
        top_k=config['logger'].get('top_k', 5),
        save_start=config['logger'].get('save_start', 0),
        max_abs_dc=config['logger'].get('max_abs_dc'),
    )
    last_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    wandb_logger = WandbLogger(
        project=config['logger']['project_name'],
        name=config['logger']['run_name'],
        save_dir=config['logger']['save_dir'],
        config=OmegaConf.to_container(config, resolve=True),
        offline=True,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, lr_monitor],
        **config['trainer'],
    )
    trainer.fit(model, datamodule=data_module)

