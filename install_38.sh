#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh
echo "Installing mamba"
conda install mamba -n base -c conda-forge
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. The installation must have failed. Please install mamba before running this script."
    exit
fi
echo "Creating conda environment"
mamba env create -n work38 python=3.8 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate work38
echo "Installing conda packages"
mamba env update -n work38 --file conda-environment_py38_cu11_ubuntu.yml 
echo "Installing GDL"
pip install Cython==0.29
pip install -e . 
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
echo "Installing other requirements"
pip install -e . 
cd .. 
pip install -r requirements38.txt
cd ../..
echo "Installation finished"
