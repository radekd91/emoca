cd ../../.. 
mkdir -p assets 
cd assets

echo "Downloading assets to run EMOCA..." 

echo "Downloading EMOCA..."
mkdir -p EMOCA/models 
cd EMOCA/models 
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip -O EMOCA.zip
echo "Extracting EMOCA..."
unzip EMOCA.zip
cd ../../

echo "Downloading DECA related assets"
wget https://download.is.tue.mpg.de/emoca/assets/DECA.zip -O DECA.zip
echo "Extracting DECA..."
unzip DECA.zip

echo "Downloading FLAME related assets"
wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
echo "Extracting FLAME..."
unzip FLAME.zip

echo "Assets for EMOCA downloaded and extracted."
cd ../gdl_apps/EMOCA/demos
