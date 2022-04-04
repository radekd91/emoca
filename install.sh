#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh
echo "Creating conda environment"
mamba env create python=3.6 --file conda-environment_py36_cu11_ubuntu.yml 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate work36_cu11
echo "Installing GDL"
pip install Cython==0.29.14
pip install -e . 
echo "Installation finished"
