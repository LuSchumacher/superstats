"""Kernel base classes and composition helpers."""

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import numpy as np


class Kernel(ABC):
    """Interface a kernel must satisfy to be used by `GaussianProcess`.

    Subclasses set `_local_hyperparam_names` (the hyperparameter names
    they need) and implement `build`. Use `self.local_hyperparams(...)`
    inside `build` to strip this kernel's name prefix off the incoming
    `hyperparams` dict before passing values to the underlying kernel
    math.

    By default (`name=None`), `hyperparam_names` are unprefixed, e.g.
    `RBFKernel()` exposes `("length_scale", "amplitude")` — matching
    `DEFAULT_HYPER_PRIORS` directly, same as `RandomWalk`'s `sigma`/
    `delta`. Pass `name=` to prefix them (`"trend_length_scale"`),
    which is required when combining two kernels that would otherwise
    expose the same hyperparameter names.

    Kernels combine with `+` (sum) and `*` (elementwise product) into
    a `CompositeKernel` — both operations preserve
    positive-semidefiniteness, so any combination is itself a valid
    kernel.

    Parameters
    ----------
    name : str, optional
        Prefix for this kernel's hyperparameter names. Leave unset
        for a single kernel, or when combining kernels of different
        types. Required (and must be unique) when combining two
        kernels that share hyperparameter names, e.g.
        `RBFKernel(name="trend") + RBFKernel(name="local")`.

    Attributes
    ----------
    hyperparam_names : tuple of str
        Hyperparameter names this kernel expects in `build`, e.g.
        `("length_scale", "amplitude")` when unnamed, or
        `("trend_length_scale", "trend_amplitude")` when
        `name="trend"`. These become `Transition.hyper_specs` entries.
    """

    _local_hyperparam_names: Sequence[str] = ()

    def __init__(self, name: str | None = None):
        self.name = name
        prefix = f"{name}_" if name is not None else ""
        self.hyperparam_names = tuple(f"{prefix}{n}" for n in self._local_hyperparam_names)

    def local_hyperparams(self, **hyperparams: np.ndarray) -> Dict[str, np.ndarray]:
        """Strip this kernel's name prefix off `hyperparams`.

        Parameters
        ----------
        **hyperparams : np.ndarray
            Must contain every name in `self.hyperparam_names`. Extra
            keys (from sibling kernels in a composite) are ignored.

        Returns
        -------
        local : dict - `hyperparams` re-keyed by the unprefixed names
            in `self._local_hyperparam_names`
        """
        prefix = f"{self.name}_" if self.name is not None else ""
        return {local: hyperparams[f"{prefix}{local}"] for local in self._local_hyperparam_names}

    @abstractmethod
    def build(self, num_steps: int, **hyperparams: np.ndarray) -> np.ndarray:
        """Construct the (batch_size, num_steps, num_steps) kernel matrix.

        Parameters
        ----------
        num_steps : int
        **hyperparams : np.ndarray
            Must contain every name in `self.hyperparam_names`, each of
            shape (batch_size,). Extra keys (from sibling kernels in a
            composite) are ignored.

        Returns
        -------
        kernel_mat : np.ndarray of shape (batch_size, num_steps, num_steps)
        """
        raise NotImplementedError

    def __add__(self, other: "Kernel") -> "CompositeKernel":
        return CompositeKernel(self, other, op="add")

    def __mul__(self, other: "Kernel") -> "CompositeKernel":
        return CompositeKernel(self, other, op="mul")


class CompositeKernel(Kernel):
    """Elementwise sum or product of two kernels' covariance matrices.

    Not constructed directly in normal use — created by `+` / `*` on
    `Kernel` instances, e.g. `RBFKernel() + LinearKernel()`.

    Parameters
    ----------
    left, right : Kernel
        Kernels to combine. Must have disjoint `hyperparam_names` —
        combining two kernels of the same type requires giving at
        least one of them an explicit `name`.
    op : {"add", "mul"}
    """

    def __init__(self, left: Kernel, right: Kernel, op: str):
        if op not in ("add", "mul"):
            raise ValueError(f"op must be 'add' or 'mul', got {op!r}")

        overlap = set(left.hyperparam_names) & set(right.hyperparam_names)
        if overlap:
            raise ValueError(
                f"Combined kernels have colliding hyperparameter name(s) {overlap}. "
                "Give one of them an explicit name, e.g. "
                "RBFKernel(name='short') + RBFKernel(name='long')."
            )

        self.left = left
        self.right = right
        self.op = op
        self.name = None
        self.hyperparam_names = left.hyperparam_names + right.hyperparam_names

    def build(self, num_steps: int, **hyperparams: np.ndarray) -> np.ndarray:
        left_mat = self.left.build(num_steps, **hyperparams)
        right_mat = self.right.build(num_steps, **hyperparams)
        return left_mat + right_mat if self.op == "add" else left_mat * right_mat
