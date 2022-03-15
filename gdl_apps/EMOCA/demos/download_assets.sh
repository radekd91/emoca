cd ../../.. 
mkdir -p assets 
cd assets

echo "Downloading assets to run EMOCA..." 

echo "Downloading EMOCA..."
 wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip -O EMOCA.zip
mkdir -p EMOCA/models 
cd EMOCA/models 
echo "Extracting EMOCA..."
unzip EMOCA.zip
cd ../../

echo "Downloading DECA..."
wget https://download.is.tue.mpg.de/emoca/assets/DECA.zip -O DECA.zip
echo "Extracting DECA..."
unzip DECA.zip

echo "Downloading FLAME..."
wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
echo "Extracting FLAME..."
unzip FLAME.zip

echo "Assets for EMOCA downloaded and extracted."
cd ../../../gdl_apps/EMOCA/demos
