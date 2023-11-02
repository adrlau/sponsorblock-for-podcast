{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      # openai-whisper
      # openai-whisper-cpp
      # youtube-dl
      # python310Packages.youtube-dl
      python3
      python3Packages.openai-whisper
      python3Packages.youtube-transcript-api
      python3Packages.pandas
      python3Packages.numpy
      python3Packages.colorama
      python3Packages.torch
      python3Packages.nltk
    ];
}