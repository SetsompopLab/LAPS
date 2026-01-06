from typing import Dict

from loguru import logger

from laps.recon.adadiff import AdaDiffParams, AdaDiffReconstructor
from laps.recon.linops import linop
from laps.recon.nerp import NERPParams, NERPReconstructor
from laps.recon.reconstructor import (
    AHbReconstructor,
    CGParams,
    CGReconstructor,
    LACSParams,
    LACSReconstructor,
    LDMParams,
    ModlParams,
    MoDLReconstructor,
    ReconParams,
    Reconstructor,
    StableDiffusionReconstructor,
    TextConditionalLDMParams,
)


def get_reconstructors(
    recon_typs: Dict[str, ReconParams],
    forward_model: linop,
    idle_sd_models_on_cpu: bool = False,
) -> Dict[str, Reconstructor]:
    reconstructors = {}
    for recon_typ, params in recon_typs.items():
        recon_cls = AHbReconstructor
        kwargs = dict()

        if isinstance(params, CGParams):
            logger.info(f"Instantiating CG Reconstructor with params: {params}")
            recon_cls = CGReconstructor
        elif isinstance(params, LACSParams):
            logger.info(f"Instantiating LACS Reconstructor with params: {params}")
            recon_cls = LACSReconstructor
        elif isinstance(params, ModlParams):
            logger.info(f"Instantiating MoDL Reconstructor with params: {params}")
            recon_cls = MoDLReconstructor
            kwargs = dict(load_model_to_cpu=idle_sd_models_on_cpu)
        elif isinstance(params, LDMParams):
            if isinstance(params, TextConditionalLDMParams):
                logger.info(
                    f"Instantiating Text Conditional LDM Reconstructor with params: {params}"
                )
            else:
                logger.info(
                    f"Instantiating Unconditional LDM Reconstructor with params: {params}"
                )
            recon_cls = StableDiffusionReconstructor
            kwargs = dict(load_model_to_cpu=idle_sd_models_on_cpu)
        elif isinstance(params, NERPParams):
            logger.info(f"Instantiating NERP Reconstructor with params: {params}")
            recon_cls = NERPReconstructor
        elif isinstance(params, AdaDiffParams):
            logger.info(f"Instantiating AdaDiff Reconstructor with params: {params}")
            recon_cls = AdaDiffReconstructor
        elif isinstance(params, ReconParams):
            logger.info("Couldn't determine model type, using AHb Reconstructor")
            recon_cls = AHbReconstructor
        else:
            raise ValueError(
                f"Unknown reconstructor type: {recon_typ}. "
                "Please provide a valid reconstructor type."
            )

        reconstructors[recon_typ] = recon_cls(
            forward_model=forward_model,
            params=params,
            **kwargs,
        )

    return reconstructors
