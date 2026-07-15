SPHINXOPTS ?=
SPHINXBUILD ?= python -m sphinx
O ?=
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help docs html clean linkcheck

help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docs: html

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

linkcheck:
	$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	$(RM) -r "$(BUILDDIR)"
