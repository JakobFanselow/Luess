{
  description = "Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });

    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell.override
          {
            
          }
          {
            
            buildInputs = with pkgs; [
              stdenv
              python3
              kaggle
              stockfish
            ];
            nativeBuildInputs = with pkgs; [
              libgcc
              pkg-config
              stdenv
            ];

            packages = with pkgs; [
		            python3Packages.jupyter
                python3Packages.ipython
                python3Packages.notebook              
                python3Packages.numpy
                python3Packages.pandas
                python3Packages.matplotlib
                python3Packages.scipy
                python3Packages.torch
                python3Packages.torchvision
                python3Packages.tqdm
                python3Packages.torchmetrics
                python3Packages.mlxtend
                python3Packages.scikit-learn
                python3Packages.chess
                python3Packages.kagglehub
                python3Packages.h5py
                python3Packages.wandb
            ] ++ (if stdenv.hostPlatform.system == "aarch64-darwin" then [ ] else [
              
            ]);

            shellHook = ''
              python -m venv .venv
              source .venv/bin/activate
              pip install stockfish
            '';
          };
      });
      
      
    };
}
