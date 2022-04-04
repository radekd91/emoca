# EMOCA
This repository is the official implementation of the [CVPR 2022](https://cvpr2022.thecvf.com/) paper [EMOCA: Emotion-Driven Monocular Face Capture and Animation](https://emoca.is.tue.mpg.de/). 

<p align="center"> 
<img src="gdl_apps/EMOCA/EMOCA_gif_sparse_det.gif">
<img src="gdl_apps/EMOCA/EMOCA_gif_sparse_rec.gif">
</p>

<p align="center"> 
<img src="gdl_apps/EMOCA/emoca.png">
</p>
<p align="center">Top row: input images. Middle row: coarse shape reconstruction. Bottom row: reconstruction with detailed displacements.<p align="center">

 EMOCA takes a single in-the-wild image as input and reconstructs a 3D face with sufficient facial expression detail to convey the emotional state of the input image. EMOCA advances the state-of-the-art monocular face reconstruction in-the-wild, putting emphasis on accurate capture of emotional content. The official project page is [here](https://emoca.is.tue.mpg.de/index.html).
 

## EMOCA project 
The training and testing script for EMOCA can be found in this subfolder: 

[EMOCA](gdl_apps/EMOCA) 


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
`gdl_apps` contains prototypes that use the GDL library. These can include scripts on how to train, evaluate, test and analyze models from `gdl` and/or data for various tasks. 

Look for individual READMEs in each sub-projects. 

Current projects: 
- [EMOCA](gdl_apps/EMOCA) 
- [EmotionRecognition](gdl_apps/EmotionRecognition)


## Installation 

### Dependencies

1) Install [conda](https://docs.conda.io/en/latest/miniconda.html)

2) Install [mamba](https://github.com/mamba-org/mamba)

<!-- 0) Clone the repo with submodules:  -->
<!-- ``` -->
<!-- git clone --recurse-submodules ... -->
<!-- ``` -->
3) Clone this repo

### Short version 

1) Run the installation script: 

```bash
bash install.sh
```

### Long version

1) Pull the relevant submodules using: 
```bash
bash pull_submodules.sh
```


2) Set up a conda environment with one of the provided conda files. I recommend using `conda-environment_py36_cu11_ubuntu.yml`.  
<!-- This is the one I use for the cluster `conda-environment_py36_cu11_cluster.yml`. The differences between tehse two are probably not important but I include both for completeness.  -->

You can use conda:
```bash
conda env create python=3.6 --file conda-environment_py36_cu11_ubuntu.yml
```

but I strongly recommend using [mamba](https://github.com/mamba-org/mamba) instead of conda (much faster): 

```bash
mamba env create python=3.6 --file conda-environment_py36_cu11_ubuntu.yml
```

Note: the environment might contain some packages. If you find an environment is missing then just `conda/mamba`- or  `pip`- install it and please notify me.

2) Activate the environment: 
```bash 
conda activate work36_cu11
```

3) For some reason cython is glitching in the requirements file so install it separately: 
```bash 
pip install Cython==0.29.14
```

4) Install `gdl` using pip install. I recommend using the `-e` option and I have not tested otherwise. 

```bash
pip install -e .
```

## Usage 

1) For running EMOCA examples, go to [EMOCA](gdl_apps/EMOCA) 

2) For running examples of Emotion Recognition, go to [EmotionRecognition](gdl_apps/EmotionRecognition)


## Acknowledgements 
There are many people who deserve to get credited. These include but are not limited to: 
Yao Fend and Haiwen Feng and their original implementation of [DECA](https://github.com/YadiraF/DECA).
Antoine Toisoul and colleagues for [EmoNet](https://github.com/face-analysis/emonet).
