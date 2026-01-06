"""
Wrapper to match VAE interface to that of SD VAE.
"""

from types import SimpleNamespace

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from medvae.models import AutoencoderKL_2D
from medvae.utils.factory import (
    FILE_DICT_ASSOCIATIONS,
    create_model,
    download_model_weights,
)
from omegaconf import OmegaConf

from laps import PROJECT_ROOT


class LatentDist:
    def __init__(self, dist):
        self.latent_dist = dist

    def sample(self):
        return self.latent_dist.mode()

    # alias
    def mode(self):
        return self.latent_dist.mode()


class MedVAEWrapper(ModelMixin, ConfigMixin):
    config_name = "config.json"
    ignore_for_config = ["vae"]

    @register_to_config
    def __init__(self, vae=None, scaling_factor=1.0, downsampling_factor=4):
        super().__init__()
        assert downsampling_factor in [
            4,
            8,
        ], "Only 4x and 8x downsampling are currently supported"
        if vae is None:
            model_name = (
                "medvae_4_4_2d_c" if downsampling_factor == 4 else "medvae_8_4_2d_c"
            )
            config_fpath = download_model_weights(
                FILE_DICT_ASSOCIATIONS[model_name]["config"]
            )
            if model_name == "medvae_8_4_2d_c":
                config_fpath = (
                    PROJECT_ROOT
                    / "submodules"
                    / "laps-medvae"
                    / "configs"
                    / "ours-8x1-new.yaml"
                )

            conf = OmegaConf.load(config_fpath)
            conf.embed_dim = 4
            conf.ddconfig.z_channels = 4
            conf["ddconfig"]["in_channels"] = 2
            conf["ddconfig"]["out_ch"] = 2

            vae = AutoencoderKL_2D(
                ddconfig=conf.ddconfig,
                embed_dim=conf.embed_dim,
            )

        self.vae = vae
        # When using SD pipeline it uses `block_out_channels` to determine the size of the image based on
        # 2 ** (len(block_out_channels) - 1)
        n_blocks = int(np.log2(downsampling_factor)) + 1
        self.register_to_config(
            block_out_channels=[
                1,
            ]
            * n_blocks,
            in_channels=2,
            scaling_factor=scaling_factor,
            downsampling_factor=downsampling_factor,
        )

    def encode(self, x):
        dist = self.vae.encode(x)

        return SimpleNamespace(latent_dist=LatentDist(dist))

    def decode(self, x, return_dict=False, generator=None):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.vae.decode(x)
        return (x,)
