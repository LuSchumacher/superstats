"""Benchmark a grid of RecurrentNet (LSTM) parameterizations.

Trains one `superstats.workflow.Workflow` per network configuration
defined in `benchmark/config.py` on a shared DDM simulator, then saves,
per configuration, under `benchmark/checkpoints/<config.name>/`:

- the trained approximator + training history (handled by `Workflow`
  via `checkpoint_filepath`)
- `loss.png`                    - training/validation loss curves
- `time_varying_recovery.png`   - local (time-varying) parameter recovery
- `time_invariant_recovery.png` - hyper/shared parameter recovery
- `time_invariant_calibration.png` - hyper/shared parameter calibration

Run from the repository root, e.g.:

    python -m benchmark.run_benchmark

To extend the sweep, edit `SIZE_TO_UNITS` / `LAYER_COUNTS` in
`benchmark/config.py` - no changes are needed here.
"""

from pathlib import Path

import numpy as np

import superstats as sup
import keras

from benchmark.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    INFERENCE_BATCH_SIZE,
    NUM_EPOCHS,
    NUM_POSTERIOR_SAMPLES,
    NUM_STEPS,
    NUM_TRAIN_SIMS,
    NUM_VAL_SIMS,
    NUM_VERIFY_SIMS,
    SEED,
    ModelConfig,
    build_model_configs,
)
from benchmark.model_setup import build_generative_model


def build_shared_data(generative_model: sup.simulation.GenerativeModel, seed: int = SEED) -> dict:
    """Simulate the training/validation/verification data shared across all configs.

    Sampling this once (rather than per-config) ensures every network
    parameterization is trained and evaluated on identical data, so
    differences in performance are attributable to the network alone.

    Parameters
    ----------
    generative_model : sup.simulation.GenerativeModel
        The shared DDM simulator.
    seed              : int, optional
        Seed for `np.random`, applied before each sampling call.

    Returns
    -------
    data : dict - `"train"`, `"val"` (used during `fit_offline`), and
        `"verify"` (used for posterior-recovery verification) datasets
    """
    np.random.seed(seed)
    train_data = generative_model.sample(NUM_TRAIN_SIMS, num_steps=NUM_STEPS, tile_to_steps=True)
    val_data = generative_model.sample(NUM_VAL_SIMS, num_steps=NUM_STEPS, tile_to_steps=True)
    test_data = generative_model.sample(NUM_VERIFY_SIMS, num_steps=NUM_STEPS, tile_to_steps=True)
    return {"train": train_data, "val": val_data, "test": test_data}


def run_single_benchmark(
    config: ModelConfig,
    generative_model: sup.simulation.GenerativeModel,
    data: dict,
    skip_existing: bool = True,
):
    """Train, verify, and save results for a single network configuration.

    Parameters
    ----------
    config           : ModelConfig
        The network parameterization to benchmark.
    generative_model : sup.simulation.GenerativeModel
        The shared DDM simulator.
    data             : dict
        Shared datasets, as returned by `build_shared_data`.
    skip_existing    : bool, optional, default: True
        If True and a `history.pkl` already exists in this config's
        checkpoint directory, skip the run (useful for resuming a
        partially completed sweep).
    """
    checkpoint_filepath = config.checkpoint_filepath
    figures_dir = Path(checkpoint_filepath) / "figures"

    if skip_existing and (Path(checkpoint_filepath) / "history.pkl").exists():
        print(f"[{config.name}] checkpoint already exists, skipping.")
        return

    print(f"[{config.name}] building workflow (hidden_dim={config.hidden_dim}, layers={config.num_layers})")
    summary_network = sup.networks.RecurrentNet(**config.network_kwargs())

    workflow = sup.workflow.Workflow(
        simulator=generative_model,
        summary_network=summary_network,
        checkpoint_filepath=checkpoint_filepath,
        restore_approximator=False,
        restore_history=False,
    )

    print(f"[{config.name}] training for {NUM_EPOCHS} epochs on {NUM_TRAIN_SIMS} sims")
    history = workflow.fit_offline(
        data=data["train"],
        validation_data=data["val"],
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_loss = workflow.plot_history(history)
    fig_loss.savefig(figures_dir / "loss.png", dpi=150, bbox_inches="tight")

    print(f"[{config.name}] sampling posterior for verification")
    verify_data = data["test"]
    samples = workflow.sample(
        data=verify_data["data"],
        num_samples=NUM_POSTERIOR_SAMPLES,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    fig_time_varying = workflow.verify_time_varying(verify_data, samples)
    fig_time_varying.savefig(figures_dir / "time_varying_recovery.png", dpi=150, bbox_inches="tight")

    fig_recovery, fig_calibration = workflow.verify_time_invariant(verify_data, samples)
    fig_recovery.savefig(figures_dir / "time_invariant_recovery.png", dpi=150, bbox_inches="tight")
    fig_calibration.savefig(figures_dir / "time_invariant_calibration.png", dpi=150, bbox_inches="tight")

    print(f"[{config.name}] done. Results saved to {checkpoint_filepath}")


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    generative_model = build_generative_model()
    data = build_shared_data(generative_model)

    configs = build_model_configs()
    print(f"Benchmarking {len(configs)} network configurations: {[c.name for c in configs]}")

    for config in configs:
        try:
            run_single_benchmark(config, generative_model, data)
            keras.backend.clear_session()

        except Exception as err:
            print(f"[{config.name}] FAILED:")
            print(err)


if __name__ == "__main__":
    main()
