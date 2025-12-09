{
  description = "DevEnv for HP_Classifier";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/25.11";
    flake-utils.url = "github:numtide/flake-utils";
    styler-formatter.url =
      "github:NixOS/nixpkgs/870493f9a8cb0b074ae5b411b2f232015db19a65";
  };

  outputs = { self, nixpkgs, ... }@inputs:
    let
      supportedSystems =
        [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});

    in {
      devShells = forAllSystems (system: {
        default = pkgs.${system}.mkShell {
          venvDir = "./.venv";
          packages = with pkgs.${system}; [
            python313Packages.python
            python313Packages.venvShellHook
            python313Packages.requests
            python313Packages.python-lsp-server
            ruff
            python313Packages.biopython
            python313Packages.numpy
            python313Packages.pandas
            python313Packages.matplotlib
            python313Packages.transformers
            python313Packages.torch
            taglib # binaries
            openssl
            git
            libxml2
            libxslt
            libzip
            zlib
            R
            rPackages.pubmedR
            rPackages.tidyverse
            rPackages.languageserver
            rPackages.tidytext
            rPackages.tm
            rPackages.bibliometrix
            rPackages.rsample
            inputs.styler-formatter.legacyPackages."${system}".rPackages.styler
          ];
          shellHook = ''
            python --version
            R --version
            source ./.venv/bin/activate
          '';
          # Run this command, only after creating the virtual environment
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH
            pip install -r requirements.txt
          '';

          # Now we can execute any commands within the virtual environment.
          # This is optional and can be left out to run pip manually.
          postShellHook = ''
            # allow pip to install wheels
            unset SOURCE_DATE_EPOCH
          '';
        };
      });
    };
}
