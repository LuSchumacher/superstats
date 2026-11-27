"""Safe formula evaluation for regressing parameters on context variables."""

import ast
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .categorical import Categorical, _level_kind


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class Formula:
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
    categorical
        Optional mapping from context names to :class:`Categorical` schemas.
        Use ``C(condition, low=b_low, high=b_high)`` to select an effect by
        level. For numeric labels or labels that are not Python identifiers,
        use ``C(condition, {1: b_low, 2: b_high})``. Every non-reference level
        requires an effect; the reference contributes zero. Effects may be
        arithmetic expressions, shared coefficients, or time-varying ones.

    Notes
    -----
    ``resolve`` preserves all sampled parameters and overlays formula targets,
    making its result directly suitable as input to a simulator. Context names
    must not shadow parameter names, as that would make a formula ambiguous.
    """

    def __init__(self, formulas: Sequence[str], categorical: Mapping[str, Categorical] | None = None):
        if isinstance(formulas, str) or not isinstance(formulas, Sequence) or not formulas:
            raise ValueError("formulas must be a non-empty sequence of strings.")
        if categorical is not None and not isinstance(categorical, Mapping):
            raise TypeError("categorical must be a mapping of context names to Categorical schemas.")
        self._categorical = dict(categorical) if categorical is not None else {}
        if any(not isinstance(name, str) or not name.isidentifier() for name in self._categorical):
            raise ValueError("Categorical predictor names must be valid Python identifiers.")
        if any(not isinstance(schema, Categorical) for schema in self._categorical.values()):
            raise TypeError("Each categorical predictor must have a Categorical schema.")
        self._categorical_calls: dict[ast.Call, tuple[str, list[tuple[Any, ast.expr]]]] = {}

        self.formulas = tuple(formulas)
        self.targets: list[str] = []
        self._expressions: list[ast.expr] = []
        self.dependencies: dict[str, set[str]] = {}

        for formula in self.formulas:
            target, expression = self._parse_formula(formula)
            if target in self.targets:
                raise ValueError(f"Formula target {target!r} is assigned more than once.")
            self.targets.append(target)
            self._expressions.append(expression)
            functions = {node.func for node in ast.walk(expression) if isinstance(node, ast.Call)}
            self.dependencies[target] = {
                node.id for node in ast.walk(expression) if isinstance(node, ast.Name) and node not in functions
            }

        overlap = set(self.targets) & self._categorical.keys()
        if overlap:
            raise ValueError(f"Formula targets cannot overwrite categorical context: {sorted(overlap)}")
        unused = self._categorical.keys() - {name for name, _ in self._categorical_calls.values()}
        if unused:
            raise ValueError(f"Categorical predictors are not used in a C(...) term: {sorted(unused)}")

    @property
    def categorical(self) -> dict[str, Categorical]:
        """A copy of the fixed categorical schemas used by this formula."""
        return dict(self._categorical)

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

        Formulas are evaluated in the order supplied to :class:`Formula`.
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
            context selected by :class:`Model` through ``design_context``.

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

        categories = {}
        for name, schema in self._categorical.items():
            if name not in context:
                raise KeyError(f"Categorical predictor {name!r} must be supplied in context.")
            categories[name] = schema.indices(context[name], name=name)
        namespace = self._prepare_namespace({**parameters, **context, **categories})
        resolved = dict(parameters)
        for target, expression in zip(self.targets, self._expressions):
            try:
                value = self._evaluate(expression, namespace)
            except KeyError as error:
                raise KeyError(f"Formula for {target!r} references unknown name {error.args[0]!r}.") from error
            namespace[target] = value
            resolved[target] = value
        return resolved

    def _parse_formula(self, formula: str) -> tuple[str, ast.expr]:
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
        self._validate_expression(expression, formula)
        return assignment.targets[0].id, expression

    def _validate_expression(self, expression: ast.expr, formula: str) -> None:
        def visit(node):
            if isinstance(node, ast.Name):
                if node.id in self._categorical:
                    raise ValueError(f"Categorical predictor {node.id!r} must be used through C(...).")
                return
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            ):
                return
            if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
                visit(node.left)
                visit(node.right)
                return
            if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
                visit(node.operand)
                return
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "C":
                name, effects = self._validate_categorical_call(node)
                self._categorical_calls[node] = (name, effects)
                for _, effect in effects:
                    visit(effect)
                return
            raise ValueError(f"Unsupported expression in formula {formula!r}: {ast.dump(node)}")

        visit(expression)

    def _validate_categorical_call(self, node: ast.Call) -> tuple[str, list[tuple[Any, ast.expr]]]:
        if not node.args or not isinstance(node.args[0], ast.Name):
            raise ValueError("C(...) requires a declared categorical predictor name as its first argument.")
        name = node.args[0].id
        if name not in self._categorical:
            raise ValueError(f"C(...) predictor {name!r} has no Categorical declaration.")
        if len(node.args) == 1:
            if any(keyword.arg is None for keyword in node.keywords):
                raise ValueError("C(...) does not support unpacked keyword arguments.")
            effects = [(keyword.arg, keyword.value) for keyword in node.keywords]
        elif len(node.args) == 2 and isinstance(node.args[1], ast.Dict) and not node.keywords:
            effects = []
            for key, value in zip(node.args[1].keys, node.args[1].values):
                try:
                    level = ast.literal_eval(key)
                except (ValueError, TypeError) as error:
                    raise ValueError("C(...) effect keys must be literal category labels.") from error
                effects.append((level, value))
        else:
            raise ValueError("C(...) accepts either named effects or one literal mapping of levels to effects.")

        schema = self._categorical[name]
        labels = [level for level, _ in effects]
        kind = _level_kind(schema.levels[0])
        if any(_level_kind(level) != kind for level in labels):
            raise ValueError(f"C({name}, ...) effect labels must match the declared level types.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"C({name}, ...) has duplicate effect labels.")
        if set(labels) != set(schema.non_reference_levels):
            raise ValueError(
                f"C({name}, ...) requires exactly the non-reference levels {schema.non_reference_levels!r}; "
                f"got {labels!r}. The reference {schema.reference!r} contributes zero."
            )
        return name, effects

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
        if isinstance(node, ast.Call):
            name, effects = self._categorical_calls[node]
            codes = namespace[name]
            result = np.zeros_like(codes, dtype=float)
            for level, effect in effects:
                value = self._evaluate(effect, namespace)
                result = np.where(codes == self._categorical[name].levels.index(level), value, result)
            return result
        raise RuntimeError("Formula validation failed to reject an unsupported AST node.")
