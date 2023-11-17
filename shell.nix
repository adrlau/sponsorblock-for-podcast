{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      openai-whisper
      python3
      python3Packages.openai-whisper
      python3Packages.youtube-transcript-api
      python3Packages.numpy
      python3Packages.pandas
      python3Packages.colorama
      python3Packages.torch
      # python310Packages.torchWithRocm
      python3Packages.nltk
      python310Packages.textacy
      python310Packages.spacy
      python310Packages.spacy-transformers
      python310Packages.spacy-legacy
      python3Packages.asteval
      python3Packages.scikit-learn
      python3Packages.scipy
      python310Packages.seaborn
      python310Packages.pip # for installing spacy models
      python310Packages.scikit-learn-extra
      rocm-core
      rocm-runtime
      rocminfo
      rocm-smi
    ];
  shellHook = ''
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    #create a virtual environment
    python3 -m venv venv
    source venv/bin/activate
    #install spacy models
    python -m spacy download en_core_web_sm
  '';
}