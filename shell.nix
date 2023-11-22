{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      openai-whisper
      python310
      python310Packages.openai-whisper
      python310Packages.pyaudio
      python310Packages.pydub
      python310Packages.youtube-transcript-api
      python310Packages.numpy
      python310Packages.mitmproxy
      python310Packages.pandas
      python310Packages.colorama
      python310Packages.torch
      # python310Packages.torchWithRocm
      python310Packages.nltk
      python310Packages.textacy
      python310Packages.spacy
      python310Packages.spacy-transformers
      python310Packages.spacy-legacy
      python310Packages.asteval
      python310Packages.scikit-learn
      python310Packages.scipy
      python310Packages.seaborn
      python310Packages.pip # for installing spacy models and torchtext (not in nixpkgs)
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

    # #not currently in the nixpkgs so we need to install it manually using pip, and because we eigther way use a venv to install spacy models we can just install it there. Should be removed once it is in the nixpkgs, but i did not have time to package a fancy python package for this project
    # pip install torchtext
  '';
}