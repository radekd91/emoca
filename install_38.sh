#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh
echo "Installing mamba"
conda install mamba -n base -c conda-forge
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Please install mamba before running this script"
    exit
fi
echo "Creating conda environment"
mamba create -n work38 python=3.8 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate work38
if echo $CONDA_PREFIX | grep work38
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi
echo "Installing conda packages"
mamba env update -n work38 --file conda-environment_py38_cu11_ubuntu.yml 
echo "Installing other requirements"
pip install -r requirements38.txt
pip install Cython==0.29
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
echo "Installing GDL"
pip install -e . 
echo "Installation finished"
