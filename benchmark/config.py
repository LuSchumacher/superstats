"""Configuration for the network-parameterization benchmark.

Edit `SIZE_TO_UNITS` / `LAYER_COUNTS` to add or remove parameterizations,
and edit the constants below to change simulation / training budgets.
Everything else (`benchmark/run_benchmark.py`) is written to be agnostic
to how many configs are produced.
"""

from dataclasses import dataclass
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BENCHMARK_DIR / "checkpoints"
SEED = 0
NUM_STEPS = 100
NUM_TRAIN_SIMS = 25_000
NUM_VAL_SIMS = 100
NUM_VERIFY_SIMS = 256
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_POSTERIOR_SAMPLES = 500
INFERENCE_BATCH_SIZE = 4
SIZE_TO_UNITS = {
    "tiny": 32,
    "small": 64,
    "base": 128,
    "large": 256,
    "xlarge": 512,
}
RECURRENT_TYPES = ("gru", "lstm")

# Number of stacked (bidirectional) LSTM layers to try for each size.
LAYER_COUNTS = (1, 2)


@dataclass(frozen=True)
class ModelConfig:
    """A single network parameterization to benchmark.

    Parameters
    ----------
    name       : str
        Unique, filesystem-safe identifier used for the checkpoint
        subdirectory and figure filenames.
    size       : str
        Named size bucket (e.g. "tiny", "base"), for grouping/reporting.
    num_layers : int
        Number of stacked recurrent layers.
    hidden_dim : int or tuple of int
        Per-layer hidden dimensionality, as expected by `RecurrentNet`.
    recurrent_type : str
        Recurrent cell type, forwarded to `RecurrentNet`.
    """

    name: str
    size: str
    num_layers: int
    hidden_dim: int | tuple[int, ...]
    recurrent_type: str = "lstm"

    def network_kwargs(self) -> dict:
        """Keyword arguments to construct the corresponding `RecurrentNet`."""
        return {"hidden_dim": self.hidden_dim, "recurrent_type": self.recurrent_type}

    @property
    def checkpoint_filepath(self) -> str:
        """Directory used for this config's checkpoint and saved figures."""
        return str(CHECKPOINT_DIR / self.name)


def build_model_configs(
    size_to_units: dict[str, int] = SIZE_TO_UNITS,
    layer_counts: tuple[int, ...] = LAYER_COUNTS,
) -> list[ModelConfig]:
    """Build the full grid of network parameterizations to benchmark.

    Parameters
    ----------
    size_to_units : dict of {str: int}, optional
        Mapping from named size to per-layer hidden units.
    layer_counts  : tuple of int, optional
        Numbers of stacked recurrent layers to try for each size.

    Returns
    -------
    configs : list of ModelConfig - one entry per (size, num_layers)
        combination
    """
    configs = []
    for size, units in size_to_units.items():
        for num_layers in layer_counts:
            for recurrent_type in RECURRENT_TYPES:
                hidden_dim = units if num_layers == 1 else tuple([units] * num_layers)
                configs.append(
                    ModelConfig(
                        name=f"{recurrent_type}_{size}_{num_layers}layer",
                        size=size,
                        num_layers=num_layers,
                        hidden_dim=hidden_dim,
                        recurrent_type=recurrent_type,
                    )
                )
    return configs
