"""
Aranya Prime: High-Performance OIH Computational Engine
=======================================================

Aranya Prime is a modular, native computation engine designed to solve the "Entropy Reduction"
problem in high-performance computing. It replaces dynamic Python interpretation with 
statically compiled, highly-optimized "Fused kernels" (C++, Fortran, V).

It follows the OIH (Ontological Information Hypothesis) by maximizing computation density
and minimizing state-space degrees of freedom.

Metadata
--------
Version:    0.1 alpha
Author:     Aditya (OIH Research)
License:    Proprietary / Research
Architecture: Standard Native (Source -> Build -> Link -> Load)
"""

from .version import __version__, __author__, get_banner

# Expose core functionality - lazy loading means import won't fail before build
from .wrapper import (
    polynomial, add, sub, mul, div,
    sin, cos, tan,
    dot, magnitude, normalize,
    scale, rotate_2d
)
