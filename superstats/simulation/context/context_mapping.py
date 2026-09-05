from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class ContextMapping:
    transition_context: tuple[str, ...] = ()
    simulator_context: tuple[str, ...] = ()
    design_context: tuple[str, ...] = ()

    def split(
        self,
        context: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        available = set(context)

        requested = {
            "transition_context": set(self.transition_context),
            "simulator_context": set(self.simulator_context),
            "design_context": set(self.design_context),
        }

        missing = set().union(*requested.values()) - available
        if missing:
            raise KeyError(f"Context variables requested by the model but not generated: {sorted(missing)}")

        return {name: {variable: context[variable] for variable in variables} for name, variables in requested.items()}
