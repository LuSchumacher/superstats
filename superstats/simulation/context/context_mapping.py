from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class ContextMapping:
    """Route named context variables to formulas and simulators.

    A variable may be listed for both consumers. Generated or fixed context
    variables that are not listed remain in model outputs but are not passed
    to either consumer.
    """

    simulator_context: tuple[str, ...] = ()
    formula_context: tuple[str, ...] = ()

    def split(
        self,
        context: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        available = set(context)

        requested = {
            "simulator_context": set(self.simulator_context),
            "formula_context": set(self.formula_context),
        }

        missing = set().union(*requested.values()) - available
        if missing:
            raise KeyError(f"Context variables requested by the model but not generated: {sorted(missing)}")

        return {name: {variable: context[variable] for variable in variables} for name, variables in requested.items()}
