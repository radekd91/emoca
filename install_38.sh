#!/bin/bash
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Please install mamba before running this script"
    exit
fi

echo "Pulling submodules"
bash pull_submodules.sh
echo "Creating conda environment"
mamba env create python=3.8 --file conda-environment_py38_cu11_ubuntu.yml 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate work38
echo "Installing GDL"
pip install Cython==0.29
pip install -e . 
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.0
echo "Installing avhubert and fairseq"
cd external/av_hubert/fairseq 
pip install -e . 
cd .. 
pip install -r requirements.txt
cd ../..
echo "Installation finished"
