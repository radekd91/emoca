cd ../../.. 
mkdir -p assets 
cd assets

echo "In order to run EMOCA, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use EMOCA, please register at:" 
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


echo "Downloading assets to run EMOCA..." 

echo "Downloading EMOCA..."
mkdir -p EMOCA/models 
cd EMOCA/models 
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA.zip -O EMOCA.zip
echo "Extracting EMOCA..."
unzip EMOCA.zip
cd ../../

echo "Downloading EMOCA v2..."
mkdir -p EMOCA/models 
cd EMOCA/models 
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_mp.zip -O EMOCA_v2_mp.zip
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip -O EMOCA_v2_lr_mse_20.zip
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_cos_1.5.zip -O EMOCA_v2_lr_cos_1.5.zip
echo "Extracting EMOCA v2..."
unzip EMOCA_v2_mp.zip
unzip EMOCA_v2_lr_mse_20.zip
unzip EMOCA_v2_lr_cos_1.5.zip
cd ../../

echo "Downloading DECA..."
mkdir -p EMOCA/models 
cd EMOCA/models 
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip -O DECA.zip
echo "Extracting DECA..."
unzip DECA.zip
cd ../../


echo "Downloading DECA related assets"
wget https://download.is.tue.mpg.de/emoca/assets/DECA.zip -O DECA.zip
wget https://download.is.tue.mpg.de/emoca/assets/FaceRecognition.zip -O FaceRecognition.zip
echo "Extracting DECA related assets..."
unzip DECA.zip
unzip FaceRecognition.zip

echo "Downloading FLAME related assets"
wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
echo "Extracting FLAME..."
unzip FLAME.zip
echo "Assets for EMOCA downloaded and extracted."

cd ../
mkdir data 
cd data
echo "Downloading example test data"
wget https://download.is.tue.mpg.de/emoca/assets/data/EMOCA_test_example_data.zip -O EMOCA_test_example_data.zip
unzip EMOCA_test_example_data.zip
echo "Example test data downloaded and extracted."

cd ../gdl_apps/EMOCA/demos
