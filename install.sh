echo "Pulling submodules"
bash pull_submodules.sh
echo "Creating conda environment"
mamba env create --file conda-environment_py36_cu11_ubuntu.yml 
echo "Installing GDL"
pip install -e . 
echo "Installation finished"
