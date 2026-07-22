"""Sphinx configuration for the superstats documentation."""

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "superstats"
author = "Lukas Schumacher and Stefan Radev"
copyright = "2026, Lukas Schumacher and Stefan Radev"

try:
    release = package_version("superstats")
except PackageNotFoundError:
    with (ROOT / "pyproject.toml").open("rb") as f:
        release = tomllib.load(f)["project"]["version"]

version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

autosummary_generate = True
autodoc_class_signature = "mixed"
autoclass_content = "class"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

myst_heading_anchors = 3
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]
nb_execution_mode = "off"


templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme = "pydata_sphinx_theme"
html_title = "superstats"
html_theme_options = {
    "github_url": "https://github.com/LuSchumacher/superstats",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "logo": {
        "image_light": "_static/superstats-square-dark.svg",
        "image_dark": "_static/superstats-square-dark.svg",
        "text": "",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "bayesflow": ("https://bayesflow.org/v2.0.12/", None),
}
