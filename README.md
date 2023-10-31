# Sponsor-block-podcast

Sponsor-block-podcast is a ai adblocking application developoed for use on podcasts.

It is trained on datasets like the one fron sponsorblock for youtube. 



# interface 

The application is run using speech to text and then parses the text input to determine if the current audio track is an ad or not. 


# dependecies and build

This projects dependencies are managed using [nix](https://nixos.org/download#nix-more) the crossplattform declarative package manager.

After installing and setting up nix with nixpackages, a development shell withneeded dependencies can be created by running ``ǹix-shell`` in this directory. or by pointing to shell.nix as an argument. 





# Language

Working with neural networks and laguage is hard.
therefore i have limited my scope to only english based texts
I hope this will improve performance drastically, but at the cost of other languages.
As this is open source feel free to modify the code to fit other languages. 
A suboptimal but probably workable solution will be to pass all text input after speech to text through translate before handing to the model.


# Dataset

as making a ad detection application for podcasts is not trivial, and i could not imediateley find a usable dataset with podcasts and ads in them i remembered sponsorblock for youtube and thougth of using that dataset for classification and training.
i know that youtube videos and podcasts often have diffrent formats, I assume that the adspots are similar and identifiable enough that we could get a usable training data out of it.


data github for [sb-mirror](https://github.com/mchangrh/sb-mirror)
clone and run to download data.

    docker run --rm -it -v "${PWD}/sb-mirror:/mirror" mchangrh/sb-mirror:alpine

