#!/bin/bash
conda create -n DeepTarget python=3.7
eval "$(conda shell.bash hook)"
conda activate DeepTarget
git clone https://github.com/pcko1/Deep-Drug-Coder.git --branch moses
git clone https://github.com/EBjerrum/molvecgen.git
conda env update --file env.yml
mv Deep-Drug-Coder/ddc_pub/ .
mv molvecgen/molvecgen tmp/
rm -r -f molvecgen/
mv tmp/ molvecgen/
