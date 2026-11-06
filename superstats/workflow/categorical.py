"""Serializable categorical encoding at the inference-network boundary."""

import numpy as np

from bayesflow.adapters.transforms import Transform
from bayesflow.utils.serialization import serializable

from superstats.simulation.categorical import Categorical


@serializable("superstats.workflow")
class EncodeCategorical(Transform):
    """Encode raw context labels without changing the model's public data keys."""

    def __init__(self, categorical):
        self.categorical = {
            name: schema if isinstance(schema, Categorical) else Categorical(**schema)
            for name, schema in categorical.items()
        }

    def get_config(self):
        return {"categorical": {name: schema.get_config() for name, schema in self.categorical.items()}}

    def forward(self, data, **kwargs):
        result = dict(data)
        for name, schema in self.categorical.items():
            if name in data:
                values = np.asarray(data[name], dtype=object)
                if values.ndim != 2:
                    raise ValueError(f"Categorical context {name!r} must have shape (batch_size, num_steps).")
                result[name] = schema.encode(values, name=name, missing_mask=data.get("missing_mask"))
        return result

    def inverse(self, data, **kwargs):
        # Posterior outputs contain inference targets only. When summary data
        # is also inverted, recover labels; masked labels have been discarded.
        result = dict(data)
        for name, schema in self.categorical.items():
            if name in data:
                encoded = np.asarray(data[name])
                levels = np.asarray(schema.levels, dtype=object)
                labels = levels[np.argmax(encoded, axis=-1)]
                result[name] = np.where(np.any(encoded != 0, axis=-1), labels, None)
        return result
