cd ../../.. 
mkdir -p assets 
cd assets

echo "Downloading assets to run EMOCA..." 

echo "Downloading EMOCA..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/NaGoq8Jt4BXcTDN/download -O EMOCA.zip
echo "Extracting EMOCA..."
unzip EMOCA.zip

echo "Downloading DECA..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/Wf5CbTweKE9ap46/download -O DECA.zip
echo "Extracting DECA..."
unzip DECA.zip

echo "Downloading FLAME..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/yZiYCGZjNw37jYw/download -O FLAME.zip
echo "Extracting FLAME..."
unzip FLAME.zip

echo "Assets for EMOCA downloaded and extracted."
cd ../../../gdl_apps/EMOCA/demos
