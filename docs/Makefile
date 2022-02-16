# You can set these variables from the command line.
SPHINXOPTS    = -q
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = dimod
SOURCEDIR     = .
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-rm -rf build/
	-rm -rf build-cpp/
	-rm -rf reference/generated

cpp:
	doxygen Doxyfile

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
