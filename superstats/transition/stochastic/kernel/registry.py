"""Kernel name registry and resolution."""

from .kernel import Kernel
from .rbf import RBFKernel
from .linear import LinearKernel
from .periodic import PeriodicKernel

KERNEL_REGISTRY = {
    "rbf": RBFKernel(),
    "linear": LinearKernel(),
    "periodic": PeriodicKernel(),
}

__all__ = ["KERNEL_REGISTRY", "resolve_kernel"]


def resolve_kernel(kernel: str | Kernel) -> Kernel:
    """Resolve a kernel name or instance to a `Kernel`.

    Parameters
    ----------
    kernel : str or Kernel
        Either a registered name (currently "rbf", "linear", or
        "periodic") or a
        `Kernel` instance, including composites built via `+`/`*`.

    Returns
    -------
    kernel : Kernel
    """
    if isinstance(kernel, str):
        try:
            return KERNEL_REGISTRY[kernel]
        except KeyError:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Available: {list(KERNEL_REGISTRY)}, or pass a custom Kernel instance."
            )
    if not isinstance(kernel, Kernel):
        raise TypeError(f"kernel must be a str or Kernel instance, got {type(kernel)}")
    return kernel
