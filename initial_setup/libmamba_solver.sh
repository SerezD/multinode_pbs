#!/bin/bash

# https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

conda update -n base conda

conda install -n base conda-libmamba-solver
conda config --set solver libmamba