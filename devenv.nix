{ pkgs, lib, config, inputs, ... }:
{
  packages = [
    pkgs.graphviz
    pkgs.pkg-config
    pkgs.ninja
    pkgs.snakemake
    pkgs.apptainer
    pkgs.python313Packages.packaging
  ];

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
