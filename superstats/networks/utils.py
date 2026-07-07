from collections.abc import Sequence
from typing import Any


def expand_singletons_to_common_length(**kwargs: Any) -> dict[str, list[Any]]:
    """Expand scalar and single-element arguments to the common sequence length.

    Strings and bytes are treated as scalar values, not as sequences.
    """
    values = {name: _as_list(value) for name, value in kwargs.items()}
    lengths = {name: len(value) for name, value in values.items()}

    empty = [name for name, length in lengths.items() if length == 0]
    if empty:
        names = ", ".join(empty)
        raise ValueError(f"Arguments must not be empty sequences: {names}.")

    sequence_lengths = {length for length in lengths.values() if length > 1}
    if len(sequence_lengths) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items() if length > 1)
        raise ValueError(f"Sequence arguments with more than one element must have the same length; got {details}.")

    target_length = max(lengths.values(), default=1)
    return {name: value * target_length if len(value) == 1 else value for name, value in values.items()}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)

    return [value]
