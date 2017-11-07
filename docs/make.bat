
set SPHINXOPTS    =
set SPHINXBUILD   = sphinx-build
set SPHINXPROJ    = dimod
set SOURCEDIR     = .
set BUILDDIR      = build
set GH_PAGES_SOURCES = docs dimod LICENSE.txt

%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% 
