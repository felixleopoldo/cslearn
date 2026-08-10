{ pkgs, lib, config, inputs, ... }:
{
  packages = [
    pkgs.graphviz
    pkgs.pkg-config
    pkgs.ninja
    pkgs.snakemake
    pkgs.apptainer
    pkgs.python313Packages.packaging
    pkgs.python313Packages.pygraphviz
    pkgs.pandoc
  ];

  # PyPI's pygraphviz wheel (installed into the uv venv per pyproject.toml)
  # needs shared libs the nix devenv doesn't provide; shadow it with
  # nixpkgs' own build, already correctly linked against pkgs.graphviz, by
  # putting it first on PYTHONPATH. Prepended in enterShell (not
  # env.PYTHONPATH) since languages.python already sets that option to its
  # own venv-activation shim.
  enterShell = ''
    export PYTHONPATH="${pkgs.python313Packages.pygraphviz}/lib/python3.13/site-packages:$PYTHONPATH"
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
