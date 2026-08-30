{ pkgs, lib, config, inputs, ... }:
{
  packages = [
    pkgs.graphviz
    pkgs.pkg-config
    pkgs.ninja
    pkgs.snakemake
    pkgs.apptainer
    pkgs.ghostscript
    pkgs.ruff
    pkgs.python313Packages.packaging
    pkgs.python313Packages.pygraphviz
    pkgs.python313Packages.pikepdf
    pkgs.pandoc
    pkgs.cm_unicode
  ];

  # PyPI's pygraphviz and pikepdf wheels (installed into the uv venv per
  # pyproject.toml) need shared libs the nix devenv doesn't provide (libgraphviz
  # and libz respectively); shadow both with nixpkgs' own builds, already
  # correctly linked, by putting them first on PYTHONPATH. Prepended in
  # enterShell (not env.PYTHONPATH) since languages.python already sets that
  # option to its own venv-activation shim.
  #
  # FONTCONFIG_FILE points graphviz's text renderer at cm_unicode ("CMU
  # Serif", a TrueType Computer/Latin Modern lookalike) without touching the
  # user's global font config, so LDAG figure text matches the manuscript's
  # lmodern body font instead of graphviz's larger-x-height default (Times).
  enterShell = ''
    export PYTHONPATH="${pkgs.python313Packages.pygraphviz}/lib/python3.13/site-packages:${pkgs.python313Packages.pikepdf}/lib/python3.13/site-packages:$PYTHONPATH"
    export FONTCONFIG_FILE="${pkgs.makeFontsConf { fontDirectories = [ pkgs.cm_unicode ]; }}"
  '';

  languages.python = {
    enable = true;
    venv.enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
        allExtras = true;
      };
    };
  };
}
