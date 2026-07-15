"""Kernel implementations for Gaussian-process transitions."""

from .kernel import Kernel, CompositeKernel
from .rbf import RBFKernel, get_rbf_kernel, build_sq_dist_mat
from .linear import LinearKernel, get_linear_kernel
from .periodic import PeriodicKernel, get_periodic_kernel, build_abs_dist_mat
from .registry import KERNEL_REGISTRY, resolve_kernel

__all__ = [
    "Kernel",
    "CompositeKernel",
    "RBFKernel",
    "LinearKernel",
    "PeriodicKernel",
    "resolve_kernel",
    "KERNEL_REGISTRY",
    "get_rbf_kernel",
    "get_linear_kernel",
    "get_periodic_kernel",
    "build_sq_dist_mat",
    "build_abs_dist_mat",
]
