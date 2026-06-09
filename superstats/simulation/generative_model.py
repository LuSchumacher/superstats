from typing import Callable, Dict
import inspect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superstats.prior.joint_prior import JointPrior

class GenerativeModel:
    """
    A generative model that combines a joint prior with a simulation function.

    This class facilitates sampling parameters from a joint prior distribution
    and generating simulated data using a user-provided model function. It handles
    parameter broadcasting, flattening, and reshaping to support batched simulations
    with time-varying parameters.

    Parameters
    ----------
    prior : JointPrior
        The joint prior distribution over model parameters, which may include
        both time-varying transitions and time-invariant priors.
    model : Callable
        The simulation function that takes parameter values and returns simulated data.
        The function signature determines the expected parameter names and order.
    """

    def __init__(
        self,
        prior: JointPrior,
        model: Callable
    ):
        """
        Initialize the generative model.

        Parameters
        ----------
        prior : JointPrior
            The joint prior distribution for model parameters.
        model : Callable
            The simulation function to generate data from parameters.
        """
        self.prior = prior
        self.model = model

        # Inspect simulator signature
        self.signature = inspect.signature(model)
        self.param_order = list(self.signature.parameters.keys())

        # Run a pilot draw to determine key groups once
        pilot = self.prior.sample(batch_size=1, steps=1)
        self.local_keys = list(pilot["local_params"].keys()) if pilot.get("local_params") else []
        self.hyper_keys = list(pilot["hyper_params"].keys()) if pilot.get("hyper_params") else []
        self.shared_keys = list(pilot["shared_params"].keys()) if pilot.get("shared_params") else []
        self.fixed_keys = list(pilot["fixed_params"].keys()) if pilot.get("fixed_params") else []

    def _prepare_flat_params(self, combined_params, batch_size, steps):
        # Broadcast and flatten parameters for vectorized simulation.
        flat_params = {}
        for name in self.param_order:
            if name not in combined_params:
                param = self.signature.parameters[name]
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{name}' required by model but missing in prior."
                    )
                else:
                    # skip parameters with defaults
                    continue
            p = combined_params[name]
            p = np.asarray(p)
            # shared parameters
            if p.ndim == 1:
                p = np.broadcast_to(p[:, None], (batch_size, steps))
                flat_params[name] = p.reshape(batch_size * steps)
            # local parameters
            elif p.ndim == 2:
                flat_params[name] = p.reshape(batch_size * steps)
            # fixed parameters
            elif p.ndim == 0:
                p = np.full((batch_size, steps), p.item(), dtype=np.asarray(p).dtype)
                flat_params[name] = p.reshape(batch_size * steps)
            else:
                raise ValueError(
                    f"Unexpected shape for parameter '{name}': {p.shape}"
                )

        return flat_params

    def _normalize_local_params(self, params, batch_size, steps):
        # Normalize time-varying parameters to shape (batch_size, steps, 1).
        if not params:
            return None

        normalized = {}
        for name, value in params.items():
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError(
                    f"Local parameter '{name}' must have shape (batch_size, steps), got {arr.shape}"
                )
            normalized[name] = arr.reshape(batch_size, steps, 1)
        return normalized

    def _normalize_batch_params(self, params, batch_size):
        if not params:
            return None

        normalized = {}

        for name, value in params.items():
            arr = np.asarray(value)

            # scalar per batch -> (B,1 )
            if arr.ndim == 1:
                normalized[name] = arr.reshape(batch_size, 1)

            # already batched scalar parameters -> (B, 1)
            elif arr.ndim == 2 and arr.shape[1] == 1:
                normalized[name] = arr

            # vector-valued hyperparameter (e.g. mixture weights)
            elif arr.ndim == 2:
                normalized[name] = arr  # keep (B, K)

            # scalar constant
            elif arr.ndim == 0:
                normalized[name] = np.full((batch_size, 1), arr.item(), dtype=arr.dtype)

            else:
                raise ValueError(
                    f"Parameter '{name}' has invalid shape {arr.shape}"
                )

        return normalized

    def sample(
        self,
        batch_size: int,
        steps: int,
        include_fixed: bool = False,
        tile_to_steps: bool = False
    ):
        """
        Sample parameters from the prior and generate simulated data.

        This method performs a complete generative process:
        1. Samples parameters from the joint prior distribution
        2. Prepares parameters for vectorized simulation
        3. Runs the simulation model
        4. Reshapes outputs back to trajectory format

        Parameters
        ----------
        batch_size : int
            Number of independent simulation batches to generate.
        steps : int
            Number of time steps per trajectory.
        include_fixed : bool, optional
            If True, include ``fixed_params`` in the returned dictionary.
            Default is False.
        tile_to_steps : bool, optional
            If True, tile ``hyper_params`` and ``shared_params`` from shape
            (batch_size, 1) to (batch_size, steps, 1), aligning them with
            the time axis of local parameters. Default is False.

        Returns
        -------
        dict
            Flat dictionary with ``'data'`` plus one entry per sampled parameter.
            Local (time-varying) params have shape ``(batch_size, steps, 1)``;
            hyper and shared params have shape ``(batch_size, 1)``, or
            ``(batch_size, steps, 1)`` when ``tile_to_steps`` is True.
            Fixed params are included only when ``include_fixed`` is True.

            The instance attributes ``local_keys``, ``hyper_keys``,
            ``shared_keys``, and ``fixed_keys`` are updated each call to
            record which keys belong to which parameter group.

        Raises
        ------
        ValueError
            If required parameters are missing from the prior or have invalid shapes.
        """

        # Sample parameters
        prior_draws = self.prior.sample(batch_size=batch_size, steps=steps)
        local_params = prior_draws["local_params"]
        shared_params = prior_draws.get("shared_params", {})
        fixed_params = prior_draws.get("fixed_params", {})

        # Combine parameter dictionaries
        combined_params = dict(local_params)
        combined_params.update(shared_params)
        # Include fixed params that are used by the model
        for name in self.param_order:
            if name in fixed_params:
                combined_params[name] = fixed_params[name]

        # Broadcast + flatten params
        flat_params = self._prepare_flat_params(
            combined_params, batch_size, steps
        )

        # Order parameters according to model signature
        ordered_params = []
        for name in self.param_order:
            if name in flat_params:
                ordered_params.append(flat_params[name])
            else:
                # fallback to default value from function signature
                default = self.signature.parameters[name].default
                if default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{name}' required by model but missing in prior and has no default."
                    )
                ordered_params.append(default)

        # Run simulator
        sim_data = self.model(*ordered_params)
        sim_data = np.asarray(sim_data)

        # Reshape back to trajectories
        output_shape = sim_data.shape[1:] if sim_data.ndim > 1 else ()

        sim_data = sim_data.reshape(
            batch_size,
            steps,
            *output_shape
        )

        local_params = self._normalize_local_params(local_params, batch_size, steps)
        hyper_params = self._normalize_batch_params(prior_draws.get("hyper_params", {}), batch_size)
        shared_params = self._normalize_batch_params(shared_params, batch_size)

        if tile_to_steps:
            if hyper_params is not None:
                hyper_params = {
                    k: np.tile(v[:, np.newaxis, :], (1, steps, 1))
                    for k, v in hyper_params.items()
                }
            if shared_params is not None:
                shared_params = {
                    k: np.tile(v[:, np.newaxis, :], (1, steps, 1))
                    for k, v in shared_params.items()
                }
    
        result = {"data": sim_data}
        if local_params:
            result.update(local_params)
        if hyper_params:
            result.update(hyper_params)
        if shared_params:
            result.update(shared_params)
        if include_fixed and fixed_params:
            result.update(fixed_params)

        return result

    def plot_push_forward(
        self,
        batch_size: int = 6,
        steps: int = 100,
        data_dim: int = 0,
        type: str = "hist",
        aggregate: bool = False,
        stats_fun: callable = None,
        n_cols: int = 3,
        color: str = "#822621",
        title_fontsize: int = 14,
        tick_fontsize: int = 10,
        alpha: float = 0.8,
    ):
        data = self.sample(batch_size=batch_size, steps=steps)["data"]
        x = data[:, :, data_dim]  # (batch_size, steps)

        COL_WIDTH, ROW_HEIGHT = 4.0, 3.0

        # ----------------------------------------------------------------
        # trajectory + aggregate: mean line + 95% CI
        # ----------------------------------------------------------------
        if type == "trajectory" and aggregate:
            mean = x.mean(axis=0)
            lower = np.percentile(x, 2.5, axis=0)
            upper = np.percentile(x, 97.5, axis=0)
            t = np.arange(steps)

            fig, ax = plt.subplots(figsize=(COL_WIDTH * 2, ROW_HEIGHT))
            ax.plot(t, mean, color=color, linewidth=2.0)
            ax.fill_between(t, lower, upper, color=color, alpha=0.25)
            ax.set_xlabel("step", fontsize=tick_fontsize)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        # ----------------------------------------------------------------
        # hist + aggregate: stats_fun per batch item -> hist of results
        # ----------------------------------------------------------------
        elif type == "hist" and aggregate:
            fn = stats_fun if stats_fun is not None else lambda x: x.mean(axis=-1)
            stats = fn(x)
            stats = np.asarray(stats).reshape(-1)

            fig, ax = plt.subplots(figsize=(COL_WIDTH * 2, ROW_HEIGHT))
            sns.histplot(
                stats,
                bins=30,
                stat="density",
                kde=True,
                line_kws={"linewidth": 2.0},
                ax=ax,
                color=color,
                alpha=alpha,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        # ----------------------------------------------------------------
        # trajectory + no aggregate: one panel per batch item
        # ----------------------------------------------------------------
        elif type == "trajectory":
            n_rows = int(np.ceil(batch_size / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
            )
            axes = np.atleast_1d(axes).ravel()
            t = np.arange(steps)

            for i in range(batch_size):
                ax = axes[i]
                ax.plot(t, x[i], color=color, alpha=alpha, linewidth=1.5)
                ax.set_title(f"dataset {i}", fontsize=title_fontsize)
                ax.set_xlabel("step", fontsize=tick_fontsize)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

            for j in range(batch_size, len(axes)):
                axes[j].axis("off")

        # ----------------------------------------------------------------
        # hist + no aggregate: one panel per batch item
        # ----------------------------------------------------------------
        else:
            n_rows = int(np.ceil(batch_size / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
            )
            axes = np.atleast_1d(axes).ravel()

            for i in range(batch_size):
                ax = axes[i]
                sns.histplot(
                    x[i],
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 2.0},
                    ax=ax,
                    color=color,
                    alpha=alpha,
                )
                ax.set_title(f"dataset {i}", fontsize=title_fontsize)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

            for j in range(batch_size, len(axes)):
                axes[j].axis("off")

        sns.despine()
        plt.tight_layout()
        
        return fig