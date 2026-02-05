"""
coordinatecleaner-py: Clean occurrence record coordinates.

Python port of R's CoordinateCleaner package.
"""

from .flags import (
    cc_val,
    cc_zero,
    cc_equ,
    cc_dupl,
    cc_round,
    cc_gbif,
    cc_inst,
    cc_cap,
    cc_outl,
    cc_iucn,
    clean_coordinates,
    GBIF_HQ,
    CAPITALS,
    NATURAL_HISTORY_MUSEUMS,
)

__version__ = "0.1.0"
__all__ = [
    "cc_val",
    "cc_zero", 
    "cc_equ",
    "cc_dupl",
    "cc_round",
    "cc_gbif",
    "cc_inst",
    "cc_cap",
    "cc_outl",
    "cc_iucn",
    "clean_coordinates",
    "GBIF_HQ",
    "CAPITALS",
    "NATURAL_HISTORY_MUSEUMS",
]
