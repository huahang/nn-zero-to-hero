#!/bin/sh
conda env update -n micrograd --file environment.yml --prune

if [ "$OS" = "Linux" ]; then
  conda install -n micrograd pytorch-cuda=12.4 -c pytorch
fi
