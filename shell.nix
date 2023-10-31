{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      # openai-whisper
      # openai-whisper-cpp
      python310Packages.openai-whisper
      # youtube-dl
      # python310Packages.youtube-dl
      python310Packages.youtube-transcript-api
      python310Packages.pandas
      python310Packages.numpy
      python310Packages.colorama
    ];
}