# Sponsor-block-podcast

Sponsor-block-podcast is a ai adblocking application developoed for use on podcasts.

It is trained on datasets like the one fron sponsorblock for youtube, and run through a machine learning network to detect differences in text between sponsor reads and regular freeflowing text.

This project is a framework you would need to gather the dataset, process the data, train and run the model.

# dependecies and build

This projects dependencies are managed using [nix](https://nixos.org/download#nix-more) the crossplattform declarative package manager.

After installing and setting up nix with nixpackages, a development shell withneeded dependencies can be created by running ``ǹix-shell`` in this directory. or by pointing to shell.nix as an argument. 

The first time this is run it will download and build all dependencies requred to develop and run the application. It migth take a while the first, to download from cache and build non cahced dependencies, but it will be cached in the local nix store after use.

further work to make this into a flake for final packaging is needed, but currently the dev shell suits my needs.

# Language

Working with neural networks and laguage is hard.
therefore i have limited my scope to only english based texts

I hope this will improve performance, but at the cost of other languages.
As this is open source feel free to modify the code to fit other languages. 
A suboptimal but probably workable solution will be to pass all text input after speech to text through translate before handing to the model.

# Dataset and training

as making a ad detection application for podcasts is not trivial, and i could not imediateley find a usable dataset with podcasts and ads in them i remembered sponsorblock for youtube and thougth of using that dataset for classification and training.
i know that youtube videos and podcasts often have diffrent formats, I assume that the adspots are similar and identifiable enough that we could get a usable training data out of it.


data github for [sb-mirror](https://github.com/mchangrh/sb-mirror)
clone and run to download data.

    docker run --rm -it -v "${PWD}/sb-mirror:/mirror" mchangrh/sb-mirror:alpine

## Pull data

now we need to take the data and pull the text from youtube. This can be done using the datagenerator.py script

## process dataset

after the dataset is pulled into sb-mirror you can run dataconverter2.py  this will convert the data from json to our designated training format.

## train

the trainer2.py is used to train and build the weigths. If you want to modify something here please do. There are a lstm model and a regular model utilizing linears and relus.  Learningrate and epochs are also specified inline in this program.

## evaluate.

evaluator.py can be used to evaluate the trained model on some texts.  if you want to test on text from a audio clip. use stt-test.py and change the variable to the correct filename and save the output to a file.

# How to run

For final project with premade weigths, just running the following script and pointing your internet trafick to localhost:8090

    start-proxy.sh

This will intercept all requests with audio, wait until it is loaded, transcribe and filter it before passing the filtered file onwards. 
Transcribing alone takes enough time and resources to timeout a lot of stuff so consider this feature a work in progress. 
I used this site for simple testing on audio files https://www.cbvoiceovers.com/audio-samples/.
and also on a regular podcast i listen to https://darknetdiaries.com/episode/138/ but this is to long to respond within reasonable time and times out, but you should be able to get the filtered file in the .tmp folder that gets generated.

# further work.

There are a lot of unfinished parts in this project.
To begin with it is obvious that the current model has not gotten enough information to be useful. It is not really any better than semi random. 
The machiune learning aspect, probably could have use of a larger model with more data, and a better structured input where previous inputs ahve more to say in the current estimate except for in the current chunk. Also i could need a improved tokenization and cleanup by utilizing something like spacy so that i do not need to keep a file with token information.
Also the detected adspot to timestamp for removal could be implemented in a better way than just assuming tokens and text are aproximately equally distributed in time.
Evaluation improvements could also be made with utilizing a hashmap in all stages of tokenization instead of only creation, to awoid file seeking.
Evaluation on parts as a stream of audio segments instead of only evaluating the whole.
Classic ad blocking based on ip or location for the same file/site.