# Examples

Example notebooks currently live in the repository's `examples/` directory.
They are intentionally kept outside the first documentation build so the docs
can be generated quickly while the notebook execution policy is still settling.

Good first candidates to promote into rendered documentation are:

- `examples/tutorial.ipynb`
- `examples/minimal_workflow.ipynb`
- `examples/minimal_workflow_net.ipynb`
- `examples/benchmark.ipynb`

The Sphinx configuration already enables MyST-NB with execution disabled, so
these notebooks can be added to the documentation toctree when their outputs and
runtime expectations are ready.
