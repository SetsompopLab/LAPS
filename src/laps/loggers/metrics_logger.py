import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

import wandb
from laps.metrics import compute_inference_metrics, compute_patch_based_metrics
from laps.utils import normalize


class PatchBasedMetricsLogger:
    def __init__(
        self,
        recon_types: Sequence[str],
        metric_types: Sequence[str],
        sim_percentile: float = 90.0,  # this % of patch similarities is considered "similar"
        dissim_percentile: float = 10.0,  # this % of patch similarities is considered "dissimilar"
        comp_types: Sequence[str] = ["samp_targ", "samp_prior", "targ_prior"],
        wandb_enable: bool = True,
        export_dir: Optional[str] = None,
    ):
        """
        New Patch-based metrics logger which computes the metrics based on percentiles, rather than fixed thresholds.
        """

        assert (
            0 < dissim_percentile < sim_percentile < 100
        ), f"Percentiles must be in (0, 1) and dissim_percentile < sim_percentile. "

        for c in comp_types:
            assert c in [
                "samp_targ",
                "samp_prior",
                "targ_prior",
            ], f"Comparison type {c} not supported. Use one of: samp_targ, samp_prior, targ_prior"

        self.recon_types = recon_types
        self.metric_types = [m.lower() for m in metric_types]
        self.mask_types = [
            "sim_top",
            "sim_mid",
            "sim_low",
        ]
        self.comp_types = comp_types
        self.running_metric_count = 0
        self.batch_index = 0
        self.wandb_enable = wandb_enable
        self.sim_percentile = sim_percentile
        self.dissim_percentile = dissim_percentile

        # init dataframe for patch-based metrics
        self.df_patch = pd.DataFrame()

        # also log metrics when computed over full images
        self.df_full = pd.DataFrame()

        self.index = 0

        if export_dir is not None:
            os.makedirs(export_dir, exist_ok=True)
            self.export_dir = export_dir
            self.export_path = os.path.join(export_dir, "metrics.csv")
            self.export_path_patch = os.path.join(export_dir, "metrics_patch.csv")

        self.export = export_dir is not None

    def update_metrics(
        self,
        recon_dict: Dict[str, torch.Tensor],
        targ: torch.Tensor,
        prior: torch.Tensor,
        metadata: Dict = {},
    ):
        """
        Update metrics for batch of images all with shape (*im_size)
        """

        recon_types = list(recon_dict.keys())
        for r in recon_types:
            assert (
                r in self.recon_types
            ), f"Reconstruction type {r} not in {self.recon_types}"

        mets_full = {}
        mets_patch = {}

        targ = targ.abs() / targ.abs().quantile(0.99)
        prior = normalize(prior.abs(), targ, mag=True, ofs=False)

        for i, (r, rec) in enumerate(recon_dict.items()):

            rec = normalize(rec.abs(), targ, mag=True, ofs=False)

            # Patch-based metrics
            mets = compute_patch_based_metrics(
                methods=self.metric_types,
                samps=rec[
                    None,
                ].clone(),
                targs=targ[
                    None,
                ].clone(),
                priors=prior[
                    None,
                ].clone(),
            )
            for m in mets.keys():
                if m == "Similarity":
                    mets_patch[m] = mets[m].tolist()
                elif (i == 0) and "targ_prior" in m:
                    mets_patch[m] = mets[m].tolist()
                elif "targ_prior" not in m:
                    mets_patch[f"{m}:{r}"] = mets[m].tolist()

            # Full metrics
            mets_full_img = compute_inference_metrics(
                methods=self.metric_types,
                samps=rec[
                    None,
                ].clone(),
                targs=targ[
                    None,
                ].clone(),
                priors=prior[
                    None,
                ].clone(),
                return_averages=True,
            )
            for m in mets_full_img.keys():
                if (i == 0) and "targ_prior" in m:
                    mets_full[m] = [mets_full_img[m]]
                elif "targ_prior" not in m:
                    mets_full[f"{m}:{r}"] = [mets_full_img[m]]

        # match the index of this example
        mets_patch["Index"] = [self.index] * len(mets_patch["Similarity"])
        new_df = pd.DataFrame(mets_patch)
        if len(self.df_patch) == 0:
            self.df_patch = new_df
        else:
            self.df_patch = pd.concat([self.df_patch, new_df], ignore_index=True)

        # Concat the full metrics
        if metadata is not None:
            for k, v in metadata.items():
                mets_full[k] = [v]
        mets_full["Index"] = [self.index]
        new_df_full = pd.DataFrame(mets_full)
        if len(self.df_full) == 0:
            self.df_full = new_df_full
        else:
            self.df_full = pd.concat([self.df_full, new_df_full], ignore_index=True)

        self.index += 1

    def update_batch_metrics(
        self,
        recons_dict: Dict[str, torch.Tensor],
        targs: torch.Tensor,
        priors: torch.Tensor,
        metadata: Optional[Sequence[Dict]] = None,
    ):
        """
        Update metrics for batch of images all with shape (B, *im_size)
        """
        B = targs.shape[0]

        if metadata is None:
            metadata = [{} for _ in range(B)]

        for i in range(B):
            # Update metrics for each image in the batch
            self.update_metrics(
                {k: v[i] for k, v in recons_dict.items()},
                targs[i],
                priors[i],
                metadata[i],
            )

    def log_batch_metrics(self, step: Optional[int] = None):
        """
        Log metrics to logger and wandb.
        """

        # Ensure we are ready to log
        assert (
            len(self.df_patch) > 0
        ), "No metrics to log. Call `update_batch_metrics` first."

        # Compute percentiles for similarity
        sim_values = np.array(self.df_patch["Similarity"].values)
        sim_top = np.percentile(sim_values, self.sim_percentile)
        sim_mid = np.percentile(sim_values, self.dissim_percentile)

        inds_top = np.where(sim_values >= sim_top)[0]
        inds_mid = np.where((sim_values < sim_top) & (sim_values >= sim_mid))[0]
        inds_low = np.where(sim_values < sim_mid)[0]

        # Log to wandb
        logdict = {"Mean_Similarity": sim_values.mean()}
        for m in self.metric_types:
            for p in self.mask_types:
                if p == "sim_top":
                    inds = inds_top
                elif p == "sim_mid":
                    inds = inds_mid
                elif p == "sim_low":
                    inds = inds_low
                else:
                    raise ValueError(f"Unknown mask type {p}")

                for c in self.comp_types:
                    if c == "targ_prior":
                        logdict[f"{m}/{c}/{p}"] = np.array(
                            self.df_patch[f"{c}:{m}"].values[inds]
                        ).mean()
                    else:
                        for r in self.recon_types:
                            logdict[f"{m}/{c}/{p}/{r}"] = np.array(
                                self.df_patch[f"{c}:{m}:{r}"].values[inds]
                            ).mean()

        # Log full metrics
        for m in self.metric_types:
            for c in self.comp_types:
                if c == "targ_prior":
                    logdict[f"{m}/{c}/full"] = np.array(
                        self.df_full[f"{c}:{m}"].values
                    ).mean()
                else:
                    for r in self.recon_types:
                        logdict[f"{m}/{c}/{r}/full"] = np.array(
                            self.df_full[f"{c}:{m}:{r}"].values
                        ).mean()

        if self.wandb_enable:
            try:
                if step is not None:
                    wandb.log(logdict, step=step)
                else:
                    wandb.log(logdict)
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}.")

        if self.export:
            self.df_patch.to_csv(self.export_path_patch, index=False)
            self.df_full.to_csv(self.export_path, index=False)
