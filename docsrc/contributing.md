# Contributing

Contributions to `superstats` are **very welcome**. Whether you are fixing a bug, improving documentation, proposing a feature, or adding new statistical functionality, we appreciate your help.

## Requesting a feature

Before implementing a major change, please, open a GitHub issue describing:

* The problem or use case.
* The behavior or API you are proposing.
* Any relevant examples or references.

This allows us to agree on the scope and approach before work begins. Small fixes and documentation improvements can be submitted directly.

## Development setup

Fork and clone the repository:

```bash
gh repo fork YOUR_ORG/superstats --clone
cd superstats
```

Install [`uv`](https://docs.astral.sh/uv/) if needed, then create the environment and install the project dependencies defined in `pyproject.toml`:

```bash
uv sync --all-extras --dev
```

Install the pre-commit hooks:

```bash
uv run pre-commit install
```

Run the tests and checks:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Use `uv run` for development commands rather than manually activating the virtual environment.

## Expectations

Contributions should:

* Include tests for new behavior na features.
* Keep existing tests passing.
* Include documentation for user-facing changes.
* Follow the existing code style and public API conventions.
* Include references for non-trivial models.

When your changes are ready, open a pull request (PR) with a clear explanation and reference the related issue.

Thank you for contributing to `superstats`!
