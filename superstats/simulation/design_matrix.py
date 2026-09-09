"""Safe formula evaluation for regressing parameters on context variables."""

import ast
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class DesignMatrix:
    """Resolve parameter-regression formulas against sampled parameters and context.

    A formula assigns one simulator parameter, for example
    ``"v = v_0 + b_v * covariate"``.  The right-hand side accepts numeric
    literals, parameter and context-variable names, parentheses, and the
    ``+``, ``-``, ``*``, ``/``, and ``**`` operators.  Formulas are evaluated
    in order, so a later formula may reference an earlier target.

    Parameters
    ----------
    formulas
        Non-empty sequence of assignment formulas.

    Notes
    -----
    ``resolve`` preserves all sampled parameters and overlays formula targets,
    making its result directly suitable as input to a simulator. Context names
    must not shadow parameter names, as that would make a formula ambiguous.
    """

    def __init__(self, formulas: Sequence[str]):
        if isinstance(formulas, str) or not isinstance(formulas, Sequence) or not formulas:
            raise ValueError("formulas must be a non-empty sequence of strings.")

        self.formulas = tuple(formulas)
        self.targets: list[str] = []
        self._expressions: list[ast.expr] = []

        for formula in self.formulas:
            target, expression = self._parse_formula(formula)
            if target in self.targets:
                raise ValueError(f"Formula target {target!r} is assigned more than once.")
            self.targets.append(target)
            self._expressions.append(expression)

    def resolve(
        self,
        parameters: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate formulas using sampled parameters and context covariates.

        The returned mapping starts with every entry from ``parameters`` and
        then overlays each formula target. This means it can be passed
        directly to :class:`~superstats.simulation.Model`: auxiliary terms
        such as ``v_0`` and ``b_v`` remain available for later formulas, while
        the simulator receives the resolved target ``v``.

        Formulas are evaluated in the order supplied to :class:`DesignMatrix`.
        Consequently, a formula may refer to a target produced by an earlier
        formula. Parameter names and context names share one namespace; the
        same name in both mappings is rejected as ambiguous.

        Parameters
        ----------
        parameters
            Mapping of parameter names to scalars or arrays. Typical model
            draws have shape ``(batch_size, num_steps)`` for local parameters
            and ``(batch_size,)`` for shared regression coefficients.
        context
            Optional mapping of covariate names to scalars or arrays. A
            trial-level covariate normally has shape
            ``(batch_size, num_steps)``. It is commonly the
            ``design_context`` supplied by :class:`ContextMapping`.

        Returns
        -------
        dict[str, Any]
            A copy of ``parameters`` with formula targets added or replaced.
            Target values are NumPy scalars or arrays produced by the formula.

        Notes
        -----
        Standard NumPy broadcasting applies. When any value is a two- or
        higher-dimensional batched array, one-dimensional values with the same
        leading batch size are treated as ``(batch_size, 1)``. Thus a shared
        coefficient of shape ``(batch_size,)`` can multiply a trial-level
        covariate of shape ``(batch_size, num_steps)``.

        Raises
        ------
        TypeError
            If ``parameters`` or a provided ``context`` is not a mapping.
        ValueError
            If a context name shadows a parameter name.
        KeyError
            If a formula references a name not supplied by either mapping or
            by an earlier formula.
        """
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping or None.")

        context = context or {}
        overlap = set(parameters) & set(context)
        if overlap:
            raise ValueError(f"Context variables shadow parameter names: {sorted(overlap)}")

        namespace = self._prepare_namespace({**parameters, **context})
        resolved = dict(parameters)
        for target, expression in zip(self.targets, self._expressions):
            try:
                value = self._evaluate(expression, namespace)
            except KeyError as error:
                raise KeyError(f"Formula for {target!r} references unknown name {error.args[0]!r}.") from error
            namespace[target] = value
            resolved[target] = value
        return resolved

    @staticmethod
    def _parse_formula(formula: str) -> tuple[str, ast.expr]:
        if not isinstance(formula, str):
            raise TypeError(f"Each formula must be a string, got {type(formula).__name__}.")

        try:
            statement = ast.parse(formula, mode="exec")
        except SyntaxError as error:
            raise ValueError(f"Invalid formula {formula!r}: {error.msg}") from error

        if len(statement.body) != 1 or not isinstance(statement.body[0], ast.Assign):
            raise ValueError(f"Formula must be one assignment, got {formula!r}.")
        assignment = statement.body[0]
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
            raise ValueError(f"Formula target must be a single variable name in {formula!r}.")

        expression = assignment.value
        DesignMatrix._validate_expression(expression, formula)
        return assignment.targets[0].id, expression

    @staticmethod
    def _validate_expression(expression: ast.expr, formula: str) -> None:
        for node in ast.walk(expression):
            if isinstance(node, (ast.Load, ast.operator, ast.unaryop)):
                continue
            if isinstance(node, ast.Name):
                continue
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            ):
                continue
            if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
                continue
            if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
                continue
            raise ValueError(f"Unsupported expression in formula {formula!r}: {ast.dump(node)}")

    @staticmethod
    def _prepare_namespace(values: Mapping[str, Any]) -> dict[str, Any]:
        arrays = {name: np.asarray(value) for name, value in values.items()}
        temporal_batch_sizes = {array.shape[0] for array in arrays.values() if array.ndim >= 2}
        if len(temporal_batch_sizes) != 1:
            return arrays

        batch_size = temporal_batch_sizes.pop()
        return {
            name: array[:, None] if array.ndim == 1 and array.shape[0] == batch_size else array
            for name, array in arrays.items()
        }

    def _evaluate(self, node: ast.expr, namespace: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Name):
            return namespace[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return UNARY_OPERATORS[type(node.op)](self._evaluate(node.operand, namespace))
        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left, namespace)
            right = self._evaluate(node.right, namespace)
            return BINARY_OPERATORS[type(node.op)](left, right)
        raise RuntimeError("Formula validation failed to reject an unsupported AST node.")
