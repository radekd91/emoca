# EMOCA
This repository is the official implementation of the CVPR 2022 paper EMOCA: Emotion Driven Monocular Face Capture and Animation. 

By using the following code, you hereby agree to sell your soul and pledge allegiance to the underworld. 


## Structure 
This repo has two subpackages. `gdl` and `gdl_apps` 

### GDL
`gdl` is a library full of research code. Some things are OK organized, some things are badly organized. It includes but is not limited to the following: 

- `models` is a module with (larger) deep learning modules (pytorch based) 
- `layers` contains individual deep learning layers 
- `datasets` contains base classes and their implementations for various datasets I had to use at some points. It's mostly image-based datasets with various forms of GT if any
- `utils` - various tools

The repo is heavily based on PyTorch and Pytorch Lightning. 

### GDL_APPS 
`gdl_apps` contains prototypes (finished and not finished). These can include scripts on how to train, evaluate, test and analyze models from `gdl` and/or data for various tasks. 

Look for individual READMEs in each sub-projects. 

Projects with a certain level of usability: 
- [EMOCA](gdl_apps/EMOCA) 
- [EmotionRecognition](gdl_apps/EmotionRecognition)


## Installation 

0) Clone the repo with submodules: 
```
git clone --recurse-submodules ...
```

1) Set up a conda environment with one of the provided conda files. I recommend using `conda-environment_py36_cu11_ubuntu.yml`.  This is the one I use for the cluster `conda-environment_py36_cu11_cluster.yml`. The differences between tehse two are probably not important but I include both for completeness. 

`conda env create --file conda-environment_py36_cu11_ubuntu.yml`

I strongly recommend using [mamba](https://github.com/mamba-org/mamba) instead of conda: 

`mamba env create --file conda-environment_py36_cu11_ubuntu.yml`


Note: the environment might contain some packages. If you find an environment is missing then just `conda/mamba`- or  `pip`- install it and please notify me.


2) Install `gdl` using pip install. I recommend using the `-e` option and I have not tested otherwise. 

`pip install -e .`

