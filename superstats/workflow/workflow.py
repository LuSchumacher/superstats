"""High-level workflow wrapper around BayesFlow."""

from typing import Literal, Callable
from collections.abc import Mapping, Sequence
from numbers import Integral

import functools
import os
import pickle
import numpy as np
import pandas as pd

import bayesflow as bf
import keras
import logging

from bayesflow.adapters import Adapter
from bayesflow.utils.numpy_utils import credible_interval

from superstats.defaults import (
    BASE_COLOR,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
)
from superstats.simulation import Model
from superstats.utils.dispatch import find_inference_network, find_embedding_network
from superstats.utils.indexing import normalize_data_indices
from superstats.utils.logging import warning as log_warning
from superstats.diagnostics.plots import (
    plot_time_varying_verification,
    plot_recovery,
    plot_calibration,
    plot_z_score_contraction,
    plot_time_varying_posterior,
    plot_time_invariant_posterior,
)


class _SuppressCheckpointExistsWarning(logging.Filter):
    def filter(self, record):
        return "Checkpoint file exists" not in record.getMessage()


class Workflow:
    """Lightweight amortized Bayesian inference workflow wrapper.

    Wraps `bf.BasicWorkflow` with sensible defaults for the embedding and
    inference networks, an auto-built adapter when one isn't supplied,
    and optional checkpoint/history restoration.

    Parameters
    ----------
    model            : Model or None, optional, default: None
        The model used for training and, when `adapter` is not
        provided, for building a default adapter. Required in that case.
    adapter              : Adapter or None, optional, default: None
        Data adapter for the workflow. If None, a default adapter is
        built from the stochastic `model.local_keys`, `model.hyper_keys`,
        and `model.shared_keys` (which requires `model` to be set).
    embedding_network      : {"recurrent", "transformer"} or keras.Layer, optional, default: "recurrent".
        String names build a default embedding network; otherwise, an already-created Keras layer is used directly.
    inference_network    : {"coupling", "coupling_flow"} or keras.Layer, optional, default: "coupling".
        String names build a default inference network; otherwise, an already-created Keras
        layer is used directly.
    checkpoint_filepath  : str or None, optional, default: None
        Directory for saving/restoring the approximator and training
        history.
    restore_approximator : bool, optional, default: True
        If True and a checkpoint directory exists at
        `checkpoint_filepath`, restore the approximator from it.
        Otherwise, a warning is issued and training starts from scratch.
    restore_history      : bool, optional, default: True
        If True and a `history.pkl` file exists at
        `checkpoint_filepath`, restore `self.history` from it.
        Otherwise, a warning is issued.
    **kwargs
        Forwarded to `bf.BasicWorkflow`.
    """

    def __init__(
        self,
        model: Model | None = None,
        adapter: Adapter | None = None,
        embedding_network: Literal["recurrent", "transformer"] | keras.Layer = "recurrent",
        inference_network: Literal["coupling", "coupling_flow"] | keras.Layer = "coupling",
        checkpoint_filepath: str | None = None,
        restore_approximator: bool = True,
        restore_history: bool = True,
        **kwargs,
    ):
        self.model = model

        self.embedding_network = find_embedding_network(embedding_network)
        self.inference_network = find_inference_network(inference_network)

        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = self.default_adapter(model)

        self.checkpoint_filepath = checkpoint_filepath

        logging.getLogger("bayesflow").addFilter(_SuppressCheckpointExistsWarning())

        self.workflow = bf.BasicWorkflow(
            simulator=self.model,
            adapter=self.adapter,
            summary_network=self.embedding_network,
            inference_network=self.inference_network,
            standardize="all",
            checkpoint_filepath=self.checkpoint_filepath,
            **kwargs,
        )

        if restore_approximator and self.checkpoint_filepath is not None and os.path.isdir(self.checkpoint_filepath):
            path = os.path.join(self.checkpoint_filepath, "model.keras")
            self.approximator = keras.saving.load_model(path)

        if restore_history and self.checkpoint_filepath is not None and os.path.isdir(self.checkpoint_filepath):
            self._load_history()

    @staticmethod
    def default_adapter(model):
        local_keys = model.local_keys
        hyper_keys = model.hyper_keys
        shared_keys = model.shared_keys
        data_keys = model.data_keys

        adapter = (
            bf.Adapter()
            .convert_dtype("float64", "float32")
            .as_time_series(["time_steps", *data_keys])
            .concatenate(local_keys + hyper_keys + shared_keys, into="inference_variables")
        )

        summary_keys = ["time_steps", *data_keys]
        if hasattr(model, "has_mask") and model.has_mask:
            adapter = adapter.as_time_series("missing_mask")
            adapter = adapter.concatenate([*summary_keys, "missing_mask"], into="summary_variables")
        else:
            adapter = adapter.concatenate(summary_keys, into="summary_variables")

        return adapter

    def _load_history(self) -> None:
        """Load persisted training history from `checkpoint_filepath`, if present.

        A no-op if `checkpoint_filepath` is None or no `history.pkl`
        file exists there.
        """
        if self.checkpoint_filepath is None:
            return
        path = os.path.join(self.checkpoint_filepath, "history.pkl")
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            self.workflow.history = pickle.load(f)

    def _save_history(self, new_history: keras.callbacks.History) -> None:
        """Merge and persist training history to `checkpoint_filepath`, if present.

        A no-op if `checkpoint_filepath` is None.

        Parameters
        ----------
        new_history : keras.callbacks.History
            History from the most recent training run. Merged into any
            existing `self.workflow.history` before saving.
        """
        if self.checkpoint_filepath is None:
            return
        existing = self.workflow.history
        if existing is not None and existing is not new_history:
            for key, values in new_history.history.items():
                existing.history.setdefault(key, []).extend(values)
            new_history = existing
        os.makedirs(self.checkpoint_filepath, exist_ok=True)
        with open(os.path.join(self.checkpoint_filepath, "history.pkl"), "wb") as f:
            pickle.dump(new_history, f)
        self.workflow.history = new_history

    def _prepare_conditions(self, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Add adapter-required auxiliary condition keys to named observations."""
        if not isinstance(data, Mapping):
            raise TypeError(f"data must be a mapping of named arrays, got {type(data)}.")

        conditions = dict(data)
        if self.model is None:
            return conditions

        data_keys = self.model.data_keys
        missing_keys = [key for key in data_keys if key not in conditions]
        if missing_keys:
            raise KeyError(f"Missing observed data keys {missing_keys!r}. Expected keys: {data_keys!r}.")

        first = conditions[data_keys[0]]
        num_datasets = first.shape[0]
        num_steps = first.shape[1]

        if "time_steps" not in conditions:
            log_warning("No time_steps provided; adding contiguous default time steps.")
            conditions["time_steps"] = np.broadcast_to(np.arange(1, num_steps + 1)[None, :], (num_datasets, num_steps))

        elif conditions["time_steps"].shape != (num_datasets, num_steps):
            raise ValueError(
                f"'time_steps' must have shape {(num_datasets, num_steps)}, got {conditions['time_steps'].shape}."
            )

        if getattr(self.model, "has_mask", False):
            if "missing_mask" not in conditions:
                log_warning("No missing_mask provided although model has missingness; assuming no missings.")
                conditions["missing_mask"] = np.zeros((num_datasets, num_steps), dtype=bool)

            elif conditions["missing_mask"].shape != (num_datasets, num_steps):
                raise ValueError(
                    f"'missing_mask' must have shape {(num_datasets, num_steps)}, "
                    f"got {conditions['missing_mask'].shape}."
                )

        remaining_keys = data_keys + ["missing_mask", "time_steps"]
        conditions = {k: v for k, v in conditions.items() if k in remaining_keys}

        return conditions

    def fit_offline(
        self, data, validation_data, epochs: int = 100, batch_size: int = 32, save_history: bool = True, **kwargs
    ) -> keras.callbacks.History:
        """Train the approximator on a fixed, pre-simulated dataset.

        Parameters
        ----------
        data            : Any
            Training data, in the format expected by
            `bf.BasicWorkflow.fit_offline`.
        validation_data : Any
            Validation data, in the same format as `data`.
        epochs          : int, optional, default: 100
            Number of training epochs.
        batch_size      : int, optional, default: 32
            Training batch size.
        save_history    : bool, optional, default: True
            If True, merge this run's history into `self.history` and
            persist it to `checkpoint_filepath` (if set).
        **kwargs
            Forwarded to `bf.BasicWorkflow.fit_offline`.

        Returns
        -------
        history : keras.callbacks.History - the training history for
            this run
        """
        history = self.workflow.fit_offline(
            data=data, epochs=epochs, batch_size=batch_size, validation_data=validation_data, **kwargs
        )

        if save_history:
            self._save_history(history)

        return history

    def fit_online(
        self,
        num_steps: int,
        epochs: int = 100,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
        save_history: bool = True,
        **kwargs,
    ) -> keras.callbacks.History:
        """Train the approximator by simulating data on the fly.

        Temporarily binds `self.model.sample` to always draw
        trajectories of length `num_steps` with `tile_to_steps=True`,
        then restores the original method afterward (even if training
        raises).

        Parameters
        ----------
        num_steps             : int
            Number of time steps per simulated trajectory during
            training.
        epochs                : int, optional, default: 100
            Number of training epochs.
        num_batches_per_epoch : int, optional, default: 100
            Number of simulated batches per epoch.
        batch_size            : int, optional, default: 32
            Training batch size.
        save_history          : bool, optional, default: True
            If True, merge this run's history into `self.history` and
            persist it to `checkpoint_filepath` (if set).
        **kwargs
            Forwarded to `bf.BasicWorkflow.fit_online`.

        Returns
        -------
        history : keras.callbacks.History - the training history for
            this run
        """
        original_sample = self.model.sample
        self.model.sample = functools.partial(original_sample, num_steps=num_steps, tile_to_steps=True)
        try:
            history = self.workflow.fit_online(
                epochs=epochs, num_batches_per_epoch=num_batches_per_epoch, batch_size=batch_size, **kwargs
            )
        finally:
            self.model.sample = original_sample

        if save_history:
            self._save_history(history)

        return history

    @property
    def history(self):
        """keras.callbacks.History or None - the workflow's training history."""
        return self.workflow.history

    @property
    def approximator(self):
        """The underlying trained BayesFlow approximator object.

        Reads through to `self.workflow.approximator` by default (kept in
        sync automatically by `bf.BasicWorkflow` during training), but can
        be explicitly assigned - e.g. when restoring a checkpoint from
        disk in `__init__`.
        """
        return self.workflow.approximator

    @approximator.setter
    def approximator(self, value):
        self.workflow.approximator = value

    def sample(
        self,
        data: dict[str, np.ndarray],
        num_samples: int = 500,
        batch_size: int = 4,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Run inference on observed data.

        Parameters
        ----------
        data                 : dict of np.ndarray
            Observed data to condition on, keyed by the model's
            named observation variables. Each value should have shape
            (num_datasets, num_steps). If `time_steps` is omitted, it is
            generated automatically. If the model was configured with
            missingness and `missing_mask` is omitted, an all-observed
            mask is generated automatically.
        num_samples          : int, optional, default: 500
            Number of posterior samples per dataset.
        batch_size : int, optional, default: 4
            Datasets per GPU batch, to avoid out-of-memory errors.
        **kwargs
            Forwarded to `self.approximator.sample`.

        Returns
        -------
        samples : dict of {param_name: np.ndarray} - posterior samples
            per parameter
        """
        conditions = self._prepare_conditions(data)
        samples = self.approximator.sample(
            conditions=conditions,
            num_samples=num_samples,
            batch_size=batch_size,
            **kwargs,
        )
        return samples

    def resimulate(
        self,
        estimates: Mapping[str, np.ndarray],
        num_sims: int = 10,
        rng=None,
        data_idx: int | Sequence[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate posterior predictive simulations from posterior parameter draws.

        Parameters
        ----------
        estimates : dict of np.ndarray
            Posterior estimates returned by `self.sample`. Each array
            should have shape (batch_size, num_samples, num_steps, dim)
            or (batch_size, num_samples, num_steps).
        num_sims          : int, optional, default: 10
            Number of posterior predictive trajectories to simulate
            per dataset.
        rng               : int or np.random.Generator or None, optional, default: None
            Random seed or generator for sampling posterior indices.
        data_idx          : int, sequence of int, or None, optional, default: None
            Dataset indices to resimulate. None selects all datasets.
            A single integer still preserves the dataset axis, and a
            sequence preserves the requested order.

        Returns
        -------
        sim_data : dict of np.ndarray
            Named simulated variables. Each value has shape
            (num_selected_datasets, num_sims, num_steps).

        Raises
        ------
        ValueError
            If `estimates` is empty, if a posterior array has
            fewer than 3 dimensions, if a parameter's batch size
            doesn't match the others, if a parameter has an unsupported
            number of dimensions, or if a parameter's shape can't be
            reshaped to collapse the sample axis into the batch axis.
        """
        rng = np.random.default_rng(rng)

        if not estimates:
            raise ValueError("estimates must be a non-empty dict.")

        # Infer shape from one posterior parameter
        example = np.asarray(next(iter(estimates.values())))
        if example.ndim < 3:
            raise ValueError(
                "Posterior sample arrays must have at least 3 dimensions: (batch_size, num_samples, num_steps, ...)."
            )

        original_batch_size, num_draws = example.shape[:2]
        selected_indices = normalize_data_indices(data_idx, original_batch_size)
        selected_estimates = {}
        for name, value in estimates.items():
            arr = np.asarray(value)
            if arr.shape[0] != original_batch_size:
                raise ValueError(
                    f"Posterior parameter '{name}' has batch size {arr.shape[0]} but expected {original_batch_size}."
                )
            selected_estimates[name] = arr[selected_indices]

        estimates = selected_estimates
        example = next(iter(estimates.values()))
        batch_size = len(selected_indices)
        time_varying_keys = set(self.model.local_keys) | set(self.model.deterministic_keys)
        time_varying_arrays = [np.asarray(value) for name, value in estimates.items() if name in time_varying_keys]
        num_steps = time_varying_arrays[0].shape[2] if time_varying_arrays else example.shape[2]

        sample_idx = rng.integers(num_draws, size=(batch_size, num_sims))

        simulation_params = {}
        fixed_params = self.model.get_fixed_params()

        for name, arr in estimates.items():
            arr = np.asarray(arr)
            if arr.shape[0] != batch_size:
                raise ValueError(
                    f"Posterior parameter '{name}' has batch size {arr.shape[0]} but expected {batch_size}."
                )

            if arr.ndim == 3:
                selected = arr[np.arange(batch_size)[:, None], sample_idx, :]
                simulation_params[name] = selected[..., 0] if name not in time_varying_keys else selected
            elif arr.ndim == 4:
                selected = arr[
                    np.arange(batch_size)[:, None, None],
                    sample_idx[:, :, None],
                    np.arange(num_steps)[None, None, :],
                    :,
                ]
                if name not in time_varying_keys:
                    while selected.ndim > 2:
                        selected = selected[..., 0]
                simulation_params[name] = selected
            else:
                raise ValueError(
                    f"Unexpected posterior shape for '{name}': {arr.shape}. "
                    "Expected (batch, samples, steps, dim) or (batch, samples, steps)."
                )

        # Collapse sample axis into batch axis for simulation
        expanded_params = {}
        for name, arr in simulation_params.items():
            if name not in time_varying_keys:
                expanded_params[name] = arr.reshape(batch_size * num_sims)
            elif arr.ndim == 2:
                expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps)
            elif arr.ndim == 3:
                if arr.shape[2] == num_steps:
                    expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps)
                else:
                    expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps, arr.shape[2])
            elif arr.ndim == 4:
                expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps, arr.shape[3])
            else:
                raise ValueError(f"Cannot reshape posterior parameter '{name}' with shape {arr.shape}.")

        for name, value in fixed_params.items():
            expanded_params[name] = np.broadcast_to(np.asarray(value), (batch_size * num_sims,))

        for name in self.model.deterministic_keys:
            if name in expanded_params:
                continue
            transition = self.model.prior.params[name]
            prefix = f"{name}_"
            transition_params = {
                key[len(prefix) :]: value for key, value in expanded_params.items() if key.startswith(prefix)
            }
            transition_fixed = transition.sample(batch_size=1, num_steps=1).get("fixed_params", {})
            transition_params.update(
                {key: value for key, value in transition_fixed.items() if key not in transition_params}
            )
            expanded_params[name] = transition.sample_from_parameters(
                transition_params,
                batch_size=batch_size * num_sims,
                num_steps=num_steps,
            )

        raw_sim = self.model.simulate_from_parameters(
            expanded_params,
            batch_size=batch_size * num_sims,
            num_steps=num_steps,
        )

        return {name: value.reshape(batch_size, num_sims, num_steps) for name, value in raw_sim.items()}

    def plot_history(
        self,
        history,
        title_fontsize: int = TITLE_FONTSIZE,
        label_fontsize: int = LABEL_FONTSIZE,
        tick_fontsize: int = TICK_FONTSIZE,
        **kwargs,
    ):
        """Plot training loss curves.

        Parameters
        ----------
        history : keras.callbacks.History
            Training history, e.g. from `fit_offline`, `fit_online`, or
            `self.history`.
        title_fontsize : int, optional, default: 22
            Font size for panel titles.
        label_fontsize : int, optional, default: 18
            Font size for axis labels and the legend.
        tick_fontsize : int, optional, default: 16
            Font size for axis tick labels.
        **kwargs
            Additional keyword arguments forwarded to
            `bf.diagnostics.plots.loss`.

        Returns
        -------
        fig : plt.Figure - the loss curve figure
        """
        kwargs.setdefault("train_color", BASE_COLOR)
        kwargs.setdefault("title_fontsize", title_fontsize)
        kwargs.setdefault("label_fontsize", label_fontsize)
        kwargs.setdefault("legend_fontsize", label_fontsize)

        fig = bf.diagnostics.plots.loss(history, **kwargs)
        for ax in fig.axes:
            ax.tick_params(labelsize=tick_fontsize)
        return fig

    def verify_time_varying(
        self,
        targets: dict,
        estimates: dict,
        variable_keys: list | None = None,
        variable_names: list | None = None,
        aggregation: Callable = np.median,
        **kwargs,
    ):
        """Plot recovery diagnostics over steps for time-varying parameters.

        Parameters
        ----------
        targets        : dict
            Ground-truth local parameter trajectories, keyed by
            parameter name; each value has shape
            (batch_size, num_steps, 1).
        estimates      : dict
            Posterior estimates for the same parameters, keyed by name;
            each value has shape
            (batch_size, num_post_samples, num_steps, 1).
        variable_keys  : list of str or None, optional, default: None
            Which parameters to select and plot, and in what order.
            Defaults to `self.model.local_keys` when not supplied.
        variable_names : list of str or None, optional, default: None
            Display names for the plotted columns, in the same order as
            `variable_keys`. Defaults to `variable_keys` when not
            supplied.
        aggregation    : callable, optional, default: np.median
            Aggregation function forwarded to
            `plot_time_varying_verification`, used to collapse each
            metric across simulations. Typically np.mean or np.median.
        **kwargs
            Additional keyword arguments forwarded to
            `plot_time_varying_verification` (e.g. `colors`,
            `title_fontsize`).

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        local_keys = self.model.local_keys

        if variable_keys is None:
            variable_keys = local_keys

        targets_squeezed = {k: targets[k][..., 0] for k in variable_keys}
        estimates_squeezed = {k: estimates[k][..., 0] for k in variable_keys}

        return plot_time_varying_verification(
            estimates=estimates_squeezed,
            targets=targets_squeezed,
            variable_keys=variable_keys,
            variable_names=variable_names,
            aggregation=aggregation,
            **kwargs,
        )

    def _prepare_time_varying_at_steps(
        self,
        targets: Mapping[str, np.ndarray],
        estimates: Mapping[str, np.ndarray],
        time_steps: int | Sequence[int],
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Select local parameters at specific zero-based time-step indices."""
        keys = list(variable_keys) if variable_keys is not None else list(self.model.local_keys)
        if not keys:
            raise ValueError("No time-varying parameters found.")

        missing = [key for key in keys if key not in estimates or key not in targets]
        if missing:
            raise ValueError(f"variable_keys not found in both estimates and targets: {missing}")

        names = list(variable_names) if variable_names is not None else keys
        if len(names) != len(keys):
            raise ValueError(f"variable_names has {len(names)} entries but there are {len(keys)} variables.")

        target_arrays = {}
        estimate_arrays = {}
        expected_shape = None
        for key in keys:
            target = np.asarray(targets[key])
            estimate = np.asarray(estimates[key])
            if target.ndim != 3 or target.shape[-1] != 1:
                raise ValueError(f"Target '{key}' must have shape (num_datasets, num_steps, 1), got {target.shape}.")
            if estimate.ndim != 4 or estimate.shape[-1] != 1:
                raise ValueError(
                    f"Estimate '{key}' must have shape (num_datasets, num_samples, num_steps, 1), got {estimate.shape}."
                )

            shape = (
                target.shape[0],
                estimate.shape[1],
                target.shape[1],
            )
            if estimate.shape[0] != shape[0] or estimate.shape[2] != shape[2]:
                raise ValueError(
                    f"Estimate and target shapes for '{key}' are inconsistent: {estimate.shape} and {target.shape}."
                )
            if expected_shape is not None and shape != expected_shape:
                raise ValueError("All selected variables must have matching dataset, sample, and time-step dimensions.")
            expected_shape = shape
            target_arrays[key] = target
            estimate_arrays[key] = estimate

        num_steps = expected_shape[2]
        if isinstance(time_steps, Integral) and not isinstance(time_steps, bool):
            selected_steps = [int(time_steps)]
        elif isinstance(time_steps, Sequence) and not isinstance(
            time_steps,
            (str, bytes),
        ):
            selected_steps = list(time_steps)
            if not selected_steps:
                raise ValueError("time_steps must contain at least one index.")
            if any(not isinstance(step, Integral) or isinstance(step, bool) for step in selected_steps):
                raise TypeError("time_steps must be an int or a sequence of ints.")
            selected_steps = [int(step) for step in selected_steps]
        else:
            raise TypeError("time_steps must be an int or a sequence of ints.")

        normalized_steps = [step + num_steps if step < 0 else step for step in selected_steps]
        invalid = [step for step in normalized_steps if step < 0 or step >= num_steps]
        if invalid:
            raise ValueError(f"time_steps contains out-of-range index {invalid[0]} for {num_steps} steps.")

        target_columns = []
        estimate_columns = []
        resolved_names = []
        show_steps = len(normalized_steps) > 1
        for step in normalized_steps:
            for key, name in zip(keys, names):
                target_columns.append(target_arrays[key][:, step, 0])
                estimate_columns.append(estimate_arrays[key][:, :, step, 0])
                resolved_names.append(f"{name} (step {step})" if show_steps else name)

        targets_arr = np.stack(target_columns, axis=-1)
        estimates_arr = np.stack(estimate_columns, axis=-1)
        return estimates_arr, targets_arr, resolved_names

    def recovery_at_steps(
        self,
        targets: Mapping[str, np.ndarray],
        estimates: Mapping[str, np.ndarray],
        time_steps: int | Sequence[int],
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        **kwargs,
    ):
        """Plot local-parameter recovery at selected time steps.

        `time_steps` contains zero-based indices and may be a single
        integer or a sequence. Plot arguments are forwarded to
        `plot_recovery`.
        """
        estimates_arr, targets_arr, names = self._prepare_time_varying_at_steps(
            targets,
            estimates,
            time_steps,
            variable_keys,
            variable_names,
        )
        return plot_recovery(
            estimates=estimates_arr,
            targets=targets_arr,
            variable_names=names,
            **kwargs,
        )

    def calibration_at_steps(
        self,
        targets: Mapping[str, np.ndarray],
        estimates: Mapping[str, np.ndarray],
        time_steps: int | Sequence[int],
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        **kwargs,
    ):
        """Plot local-parameter calibration at selected time steps.

        `time_steps` contains zero-based indices and may be a single
        integer or a sequence. Plot arguments are forwarded to
        `plot_calibration`.
        """
        estimates_arr, targets_arr, names = self._prepare_time_varying_at_steps(
            targets,
            estimates,
            time_steps,
            variable_keys,
            variable_names,
        )
        return plot_calibration(
            estimates=estimates_arr,
            targets=targets_arr,
            variable_names=names,
            **kwargs,
        )

    def z_score_contraction_at_steps(
        self,
        targets: Mapping[str, np.ndarray],
        estimates: Mapping[str, np.ndarray],
        time_steps: int | Sequence[int],
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        **kwargs,
    ):
        """Plot local-parameter z-scores and contraction at selected steps.

        `time_steps` contains zero-based indices and may be a single
        integer or a sequence. Plot arguments are forwarded to
        `plot_z_score_contraction`.
        """
        estimates_arr, targets_arr, names = self._prepare_time_varying_at_steps(
            targets,
            estimates,
            time_steps,
            variable_keys,
            variable_names,
        )
        return plot_z_score_contraction(
            estimates=estimates_arr,
            targets=targets_arr,
            variable_names=names,
            **kwargs,
        )

    def verify_time_invariant(
        self,
        targets: Mapping[str, np.ndarray] | np.ndarray,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        uncertainty_agg: Callable | None = credible_interval,
        **kwargs,
    ):
        """Plot time-invariant recovery, calibration, and contraction.

        Parameters
        ----------
        targets        : Mapping[str, np.ndarray] or np.ndarray
            If a dict, mapping from parameter name to an np.ndarray of
            shape (num_sims, dim). If an array, the fully-prepared target
            array of shape (num_sims, num_params) directly (e.g. mixture
            components already expanded).
        estimates      : Mapping[str, np.ndarray] or np.ndarray
            If a dict, mapping from parameter name to an np.ndarray of
            shape (num_sims, num_samples, steps, dim). If an array, the
            fully-prepared estimate array of shape
            (num_sims, num_pooled_samples, num_params) directly. Must use
            the same input type (dict or array) as `targets`.
        variable_keys  : sequence of str or None, optional, default: None
            Which time-invariant parameters to include, and in what
            order, when `targets`/`estimates` are dicts. Defaults to
            `self.model.hyper_keys + self.model.shared_keys` when
            not supplied. Mixture parameters (dim > 1) are expanded into
            one column per component regardless of this selection.
            Ignored for array input.
        variable_names : sequence of str or None, optional, default: None
            Display names for the final, expanded columns. For dict
            input, must match the number of expanded columns (not
            `len(variable_keys)`) and defaults to the auto-derived
            per-component names. For array input, defaults to `param_0`,
            `param_1`, ...
        uncertainty_agg : callable or None, optional, default: credible_interval
            Uncertainty aggregation passed only to `plot_recovery`. By
            default, draws 95% credible intervals. Pass `None` to suppress
            recovery uncertainty intervals.
        **kwargs
            Shared options forwarded to `plot_recovery`, `plot_calibration`,
            and `plot_z_score_contraction` (e.g. `label_fontsize`,
            `title_fontsize`, `tick_fontsize`, or `color`).

        Returns
        -------
        figs : tuple
            `(fig_recovery, fig_calibration, fig_z_score_contraction)`.

        Raises
        ------
        ValueError
            If no time-invariant parameters are found for dict input.
        """
        recovery_kwargs = dict(kwargs)
        recovery_kwargs["uncertainty_agg"] = uncertainty_agg

        if not isinstance(estimates, Mapping):
            fig_recovery = plot_recovery(
                estimates=estimates,
                targets=targets,
                variable_keys=variable_keys,
                variable_names=variable_names,
                **recovery_kwargs,
            )
            fig_calibration = plot_calibration(
                estimates=estimates,
                targets=targets,
                variable_keys=variable_keys,
                variable_names=variable_names,
                **kwargs,
            )
            fig_z_score_contraction = plot_z_score_contraction(
                estimates=estimates,
                targets=targets,
                variable_keys=variable_keys,
                variable_names=variable_names,
                **kwargs,
            )
            return fig_recovery, fig_calibration, fig_z_score_contraction

        if variable_keys is None:
            variable_keys = self.model.hyper_keys + self.model.shared_keys
        if not variable_keys:
            raise ValueError("No time-invariant parameters found.")
        missing = [k for k in variable_keys if k not in estimates or k not in targets]
        if missing:
            raise ValueError(f"variable_keys not found in both estimates and targets: {missing}")

        target_list = []
        estimate_list = []
        expanded_names = []

        for k in variable_keys:
            e_arr = estimates[k]
            B, S, T, dim = e_arr.shape
            t_arr = self._normalize_time_invariant_target(
                k,
                targets[k],
                batch_size=B,
                num_components=dim,
            )

            e_agg = e_arr.reshape(B, S * T, dim)

            if dim > 1:
                param_key = k.split("_mixture_weights")[0]
                mixture_obj = self.model.prior.params.get(param_key)
                if hasattr(mixture_obj, "names") and len(mixture_obj.names) == dim:
                    comp_names = [f"{k}_{n}" for n in mixture_obj.names]
                else:
                    comp_names = [f"{k}_{i}" for i in range(dim)]

                for i, name in enumerate(comp_names):
                    target_list.append(t_arr[:, i : i + 1])
                    estimate_list.append(e_agg[:, :, i : i + 1])
                    expanded_names.append(name)
            else:
                target_list.append(t_arr)
                estimate_list.append(e_agg)
                expanded_names.append(k)

        target_arr = np.concatenate(target_list, axis=-1)
        estimate_arr = np.concatenate(estimate_list, axis=-1)

        if variable_names is not None:
            if len(variable_names) != len(expanded_names):
                raise ValueError(
                    f"variable_names has {len(variable_names)} entries but there are "
                    f"{len(expanded_names)} expanded columns."
                )
            expanded_names = list(variable_names)

        fig_recovery = plot_recovery(
            estimates=estimate_arr,
            targets=target_arr,
            variable_names=expanded_names,
            **recovery_kwargs,
        )

        fig_calibration = plot_calibration(
            estimates=estimate_arr,
            targets=target_arr,
            variable_names=expanded_names,
            **kwargs,
        )

        fig_z_score_contraction = plot_z_score_contraction(
            estimates=estimate_arr,
            targets=target_arr,
            variable_names=expanded_names,
            **kwargs,
        )

        return fig_recovery, fig_calibration, fig_z_score_contraction

    def plot_time_varying_posterior(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        aggregation: Callable | None = None,
        aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
        uncertainty_fun: Literal["std", "ci", "mad", "hdi"] | Callable | None = "ci",
        smoothing: Literal["sma", "ema"] | None = None,
        smoothing_window: int = 5,
        marginal: bool = True,
        dist_type: Literal["hist", "kde", "both"] = "hist",
        num_bins: int | None = None,
        dist_alpha: float | None = None,
        num_cols: int | None = None,
        alpha: float = 0.5,
        color: str = BASE_COLOR,
        title_fontsize: int = TITLE_FONTSIZE,
        label_fontsize: int = LABEL_FONTSIZE,
        tick_fontsize: int = TICK_FONTSIZE,
        figsize: tuple[float, float] | None = None,
        data_idx: int | Sequence[int] | None = None,
        **kwargs,
    ):
        """Plot time-varying posterior diagnostics.

        Parameters
        ----------
        estimates          : Mapping[str, np.ndarray] or np.ndarray
            Posterior samples. If a dict, values of shape
            (num_datasets, num_post_samples, num_steps, 1), keyed by
            variable. If an array, shape
            (num_datasets, num_post_samples, num_steps, num_params)
            directly.
        targets            : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
            Ground-truth trajectories, matching the input type of
            `estimates`. If a dict, values of shape
            (num_datasets, num_steps, 1). If an array, shape
            (num_datasets, num_steps, num_params) directly. If given,
            drawn as a black dashed line on top of each panel: the raw
            per-dataset trajectory when `aggregation` is None, or
            aggregated across datasets (using `aggregation`) when
            `aggregation` is not None.
        variable_keys      : sequence of str or None, optional, default: None
            Which variables to select and plot, and in what order, when
            `estimates`/`targets` are dicts. Defaults to
            `self.model.local_keys` when not supplied. Ignored for
            array input.
        variable_names     : sequence of str or None, optional, default: None
            Display names (used for panel labels/titles), in the same
            order as `variable_keys` (or the array's last axis). Defaults
            to `variable_keys` for dict input, or `param_0`, `param_1`,
            ... for array input.
        aggregation        : callable or None, optional, default: None
            None: one panel per (param, dataset).
            callable: one panel per param, aggregated across datasets.
            Called as `aggregation(trajectories, axis=0)` and must return
            a (T,) center. The same function aggregates `targets` across
            datasets when both `targets` and `aggregation` are given.
        aggregate_strategy : {"full_uncertainty", "no_epistemic"}, optional, default: "full_uncertainty"
            Only used when `aggregation` is not None.
            "full_uncertainty": flatten only the dataset and posterior-sample
            axes into one trajectory pool, retaining posterior and
            between-dataset variation.
            "no_epistemic": take the posterior median within each dataset,
            preserving only between-dataset variation.
        uncertainty_fun    : {"std", "ci", "mad", "hdi"} or callable or None, optional, default: "ci"
            Named methods draw nested outer/inner ribbons: ±1/±0.5 SD,
            95%/65% CI, ±1.48/±0.74 MAD, or 95%/65% HDI. A callable
            receives (N, T) trajectories and draws the single `(lo, hi)`
            interval it returns, with each bound shaped (T,).
        smoothing          : {"sma", "ema"} or None, optional, default: None
            Applied to each trajectory before computing the center,
            uncertainty, and marginal.
        smoothing_window   : int, optional, default: 5
            Window size for `sma`, or span parameter for `ema`.
        marginal           : bool, optional, default: True
            Attach a marginal distribution panel to the right of each
            time-series axis. It uses the same strategy-specific trajectory
            pool as the uncertainty band.
        dist_type          : {"hist", "kde", "both"}, optional, default: "hist"
            Distribution type used for marginal panels.
        num_bins           : int or None, optional, default: None
            Number of histogram bins. If None, Seaborn selects the bins.
        dist_alpha         : float or None, optional, default: None
            Opacity of marginal distributions. If None, uses 1.0 for a
            single distribution and 0.5 when targets are overlaid.
        num_cols           : int or None, optional, default: None
            Exact number of grid columns. If None, non-aggregated plots
            use one column per selected dataset and aggregated plots use
            the shared compact dynamic layout.
        alpha              : float, optional, default: 0.5
            Opacity of the darker inner uncertainty ribbon. The outer
            ribbon uses half this opacity.
        color              : str, optional, default: BASE_COLOR
            Color used for posterior centers, bands, and marginals.
        title_fontsize     : int, optional, default: 22
            Font size for panel titles.
        label_fontsize     : int, optional, default: 18
            Font size for axis labels and the figure legend.
        tick_fontsize      : int, optional, default: 16
            Font size for axis tick labels.
        figsize            : tuple of two floats or None, optional, default: None
            Explicit figure size in inches.
        data_idx           : int, sequence of int, or None, optional, default: None
            Dataset indices to plot. None selects all datasets. A single
            integer preserves the dataset axis, and a sequence preserves
            the requested order.
        **kwargs
            Additional arguments forwarded to
            `plot_time_varying_posterior`.

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        if variable_keys is None:
            variable_keys = self.model.local_keys

        return plot_time_varying_posterior(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            aggregation=aggregation,
            aggregate_strategy=aggregate_strategy,
            uncertainty_fun=uncertainty_fun,
            smoothing=smoothing,
            smoothing_window=smoothing_window,
            marginal=marginal,
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            num_cols=num_cols,
            alpha=alpha,
            color=color,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            figsize=figsize,
            data_idx=data_idx,
            **kwargs,
        )

    def plot_time_invariant_posterior(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        aggregation: Callable | None = None,
        mixture_names: dict | None = None,
        dist_type: Literal["hist", "kde", "both"] = "hist",
        num_bins: int | None = None,
        dist_alpha: float | None = None,
        num_cols: int | None = None,
        color: str = BASE_COLOR,
        title_fontsize: int = TITLE_FONTSIZE,
        label_fontsize: int = LABEL_FONTSIZE,
        tick_fontsize: int = TICK_FONTSIZE,
        figsize: tuple[float, float] | None = None,
        data_idx: int | Sequence[int] | None = None,
        **kwargs,
    ):
        """Plot time-invariant posterior diagnostics.

        Parameters
        ----------
        estimates      : Mapping[str, np.ndarray] or np.ndarray
            Posterior samples. If a dict, values of shape
            (num_datasets, num_post_samples, num_steps, num_components),
            keyed by variable. If an array, shape
            (num_datasets, num_post_samples, num_steps, num_params)
            directly.
        targets        : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
            Ground-truth values, matching the input type of `estimates`.
            If given, drawn as black dashed vertical lines - per dataset
            when `aggregation` is None, or collapsed with `aggregation`
            into a single line per panel otherwise.
        variable_keys  : sequence of str or None, optional, default: None
            Which variables to select and plot, and in what order, when
            `estimates` is a dict. Defaults to
            `self.model.hyper_keys + self.model.shared_keys` when
            not supplied. Ignored for array input.
        variable_names : sequence of str or None, optional, default: None
            Display names for the plotted panels. Defaults to
            `variable_keys` (dict input) or `param_0`, `param_1`, ...
            (array input).
        aggregation    : callable or None, optional, default: None
            Controls both the posterior layout and the target summary. If
            None: one panel per (dataset, parameter) pair. If a callable
            (e.g. np.mean, np.median): posterior samples (and `targets`,
            if given) are pooled/aggregated across datasets into one
            panel per parameter.
        mixture_names  : dict or None, optional, default: None
            Mapping from parameter name to a list of component names.
            Defaults to `self.model.prior._mixture_names()` when not
            supplied.
        dist_type      : {"hist", "kde", "both"}, optional, default: "hist"
            Distribution type used for posterior distributions.
        num_bins       : int or None, optional, default: None
            Number of histogram bins. If None, Seaborn selects the bins.
        dist_alpha     : float or None, optional, default: None
            Opacity of posterior distributions. If None, uses 1.0 for one
            distribution and 0.5 for overlaid mixture components.
        num_cols       : int or None, optional, default: None
            Exact number of grid columns. If None, non-aggregated plots
            use one column per selected dataset and aggregated plots use
            the shared compact dynamic layout.
        color          : str, optional, default: BASE_COLOR
            Base color used for non-mixture distributions.
        title_fontsize : int, optional, default: 22
            Font size for panel titles.
        label_fontsize : int, optional, default: 18
            Font size for axis labels and the figure legend.
        tick_fontsize  : int, optional, default: 16
            Font size for axis tick labels.
        figsize        : tuple of two floats or None, optional, default: None
            Explicit figure size in inches.
        data_idx       : int, sequence of int, or None, optional, default: None
            Dataset indices to plot. None selects all datasets. A single
            integer preserves the dataset axis, and a sequence preserves
            the requested order.
        **kwargs
            Additional arguments forwarded to
            `plot_time_invariant_posterior`.

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        if variable_keys is None:
            variable_keys = self.model.hyper_keys + self.model.shared_keys
        if mixture_names is None:
            mixture_names = self.model.prior._mixture_names()

        return plot_time_invariant_posterior(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            aggregation=aggregation,
            mixture_names=mixture_names,
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            num_cols=num_cols,
            color=color,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            figsize=figsize,
            data_idx=data_idx,
            **kwargs,
        )

    @staticmethod
    def _normalize_time_invariant_target(
        name: str,
        values: np.ndarray,
        batch_size: int,
        num_components: int,
    ) -> np.ndarray:
        """Return a time-invariant target as `(batch_size, num_components)`."""
        arr = np.asarray(values)
        if arr.shape[0] != batch_size:
            raise ValueError(f"Target '{name}' has batch size {arr.shape[0]}, expected {batch_size}.")

        if arr.ndim == 1:
            if num_components != 1:
                raise ValueError(f"Target '{name}' must have {num_components} components, got shape {arr.shape}.")
            return arr[:, None]

        if arr.ndim == 2 and arr.shape[1] == num_components:
            return arr

        if arr.ndim == 2 and num_components == 1:
            tiled = arr[..., None]
        elif arr.ndim == 3 and arr.shape[2] == num_components:
            tiled = arr
        else:
            raise ValueError(
                f"Target '{name}' must have shape (batch_size, {num_components}) or "
                f"(batch_size, num_steps, {num_components}), got {arr.shape}."
            )

        if not np.allclose(tiled, tiled[:, :1, :], equal_nan=True):
            raise ValueError(
                f"Target '{name}' varies across steps but verify_time_invariant requires a time-invariant target."
            )
        return tiled[:, 0, :]

    def prepare_data(
        self,
        df: pd.DataFrame,
        id_col: str,
        data_mapping: Mapping[str, str],
        missing_value: int | float | None = None,
        time_col: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Convert a long-format DataFrame into the model's dict-of-arrays format.

        Groups `df` by `id_col` and reshapes the columns named in
        `data_mapping` into arrays of shape (batch_size, num_steps).

        If `time_col` is given, it must contain discrete integer-like
        values. The actual labels may be negative or non-contiguous; they
        are normalized to positions `1..num_steps` in sorted time order.
        Otherwise, rows are placed by order of appearance within each
        `id_col` group. Missing or padded positions are flagged in
        `"missing_mask"` and filled with the model's missing-value
        convention when one exists.

        Parameters
        ----------
        df           : pd.DataFrame
            Long-format data with one row per (dataset, step). Must contain
            `id_col`, `time_col` (if given), and every key in
            `data_mapping`.
        id_col       : str
            Name of the column in `df` identifying which dataset/sequence
            each row belongs to. Rows are grouped by this column, in order
            of first appearance, to form the batch dimension.
        data_mapping : Mapping[str, str]
            Maps a column name in `df` to the corresponding key expected by
            the model, e.g. `{"rt": "response_time", "correct":
            "choice"}`. The set of values (not keys) must exactly match
            `self.model.data_keys`.
        missing_value : int or float
            Sentinel value marking a missing observation, and used to
            initialize/pad positions with no corresponding row.
        time_col     : str or None, optional, default: None
            Name of the column in `df` giving each row's discrete time
            label. If None, rows are placed by order of appearance within
            their `id_col` group instead.

        Returns
        -------
        data : dict of np.ndarray
            One entry per model data key, each of shape
            (batch_size, num_steps), plus `"missing_mask"` (1 where any
            mapped column equals `missing_value` at that step, 0 otherwise)
            and `"time_steps"` (each row equal to `1..num_steps`).
        """
        model = getattr(self, "model", None)
        if model is None:
            raise AttributeError("prepare_data needs a Workflow with a model.")

        required_cols = [id_col, *data_mapping]
        if time_col is not None:
            required_cols.append(time_col)
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise KeyError(f"df is missing required column(s): {missing_cols}")
        if df.empty:
            raise ValueError("prepare_data requires at least one row.")

        mapped_keys = list(data_mapping.values())
        expected_keys = list(model.data_keys)
        if sorted(mapped_keys) != sorted(expected_keys):
            raise ValueError(
                f"data_mapping values {sorted(mapped_keys)!r} do not match model.data_keys {sorted(expected_keys)!r}."
            )

        if missing_value is None:
            missing_value = -1

        groups = df.groupby(id_col, sort=False)
        dataset_ids = list(groups.groups.keys())

        if time_col is not None:
            time_float = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(time_float)) or not np.all(np.isclose(time_float, np.rint(time_float))):
                raise ValueError(f"time_col '{time_col}' must contain discrete integer-like values.")

            # Map arbitrary discrete labels, including negatives, onto dense columns.
            time_values = pd.Series(np.rint(time_float).astype(np.int64), index=df.index)
            time_lookup = {value: idx for idx, value in enumerate(sorted(time_values.unique()))}
            num_steps = len(time_lookup)
        else:
            time_values = None
            time_lookup = None
            num_steps = int(groups.size().max())

        batch_size = len(dataset_ids)
        data = {data_key: np.full((batch_size, num_steps), missing_value, dtype=float) for data_key in expected_keys}

        def _col_idx(group: pd.DataFrame, dataset_id) -> np.ndarray:
            if time_col is None:
                return np.arange(len(group))
            idx = np.array([time_lookup[value] for value in time_values.loc[group.index]], dtype=int)
            if len(np.unique(idx)) != len(idx):
                raise ValueError(f"Duplicate '{time_col}' values found within id '{dataset_id}'.")
            return idx

        def _as_float_values(group: pd.DataFrame, col: str) -> np.ndarray:
            try:
                return group[col].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Column '{col}' must be numeric.") from exc

        for i, dataset_id in enumerate(dataset_ids):
            group = groups.get_group(dataset_id)
            col_idx = _col_idx(group, dataset_id)
            for col, data_key in data_mapping.items():
                data[data_key][i, col_idx] = _as_float_values(group, col)

        missing_mask = np.zeros((batch_size, num_steps), dtype=bool)
        for data_key in expected_keys:
            missing_mask |= pd.isna(data[data_key])
            if not pd.isna(missing_value):
                missing_mask |= data[data_key] == missing_value

        model_missing_value = getattr(getattr(model, "missing", None), "missing_value", missing_value)

        def _missing_fill(data_key: str, index: int):
            if isinstance(model_missing_value, Mapping):
                return model_missing_value[data_key]
            value = np.asarray(model_missing_value)
            if value.ndim == 0:
                return model_missing_value
            if value.shape == (len(expected_keys),):
                return value[index]
            raise ValueError(f"model missing_value must be scalar, mapping, or shape ({len(expected_keys)},).")

        # A missing value in any observed variable drops the whole time step.
        for i, data_key in enumerate(expected_keys):
            data[data_key][missing_mask] = _missing_fill(data_key, i)

        data["missing_mask"] = missing_mask
        data["time_steps"] = np.broadcast_to(np.arange(1, num_steps + 1)[None, :], (batch_size, num_steps))

        return data
