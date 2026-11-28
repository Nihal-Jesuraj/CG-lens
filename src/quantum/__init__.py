"""Quantum-computing extensions for CG Lens (exact classical simulation only).

Modules
-------
``simulator``     statevector primitives (Walsh-Hadamard, ZZ phases, RY states)
``feature_maps``  product and IQP/ZZ feature maps, training-fold angle scaler
``kernels``       Gram matrices, PSD diagnostics, shot noise, depolarizing model

Nothing here is run on quantum hardware and no speedup is claimed.
"""
from src.quantum.feature_maps import (
    AngleScaler,
    feature_map,
    features_to_angles,
    product_map,
    zz_map,
)
from src.quantum.kernels import (
    depolarize,
    gram,
    kernel_target_alignment,
    psd_project,
    psd_report,
    shot_noise,
)

__all__ = [
    "AngleScaler",
    "feature_map",
    "features_to_angles",
    "product_map",
    "zz_map",
    "depolarize",
    "gram",
    "kernel_target_alignment",
    "psd_project",
    "psd_report",
    "shot_noise",
]
