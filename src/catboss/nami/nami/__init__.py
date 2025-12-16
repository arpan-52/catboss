"""
NAMI: Nonlinear Automated Monotonous filter for Interference

A fast RFI flagging tool for radio astronomy data with C++ acceleration.
C++ is mandatory for performance - no Python fallback.
"""

__version__ = "0.2.0"
__author__ = "Arpan Pal"
__email__ = "apal@ncra.tifr.res.in"

# C++ extension is mandatory
import _nami_core

# Export main entry point
from nami.navigator import main

__all__ = ['main', '__version__']
