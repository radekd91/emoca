cd ../../.. 
mkdir -p assets 
cd assets

# echo "Downloading assets to run Emotion Recognition" 
# wget https://owncloud.tuebingen.mpg.de/index.php/s/WHjQE7t8BE4Re56/download -O EmotionRecognition.zip
# echo "Extracting  Emotion Recognition models"
# unzip EmotionRecognition.zip

echo "In order to run Emotion Recognition with EMOCA, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use Emotion Recognition with EMOCA, please register at:", 
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Downloading assets to run Emotion Recognition" 

mkdir -p EmotionRecognition/face_reconstruction_based 
cd EmotionRecognition/face_reconstruction_based

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/facerec_based_models/EMOCA-emorec.zip -O EMOCA-emorec.zip
echo "Extracting  EMOCA-emorec"
unzip EMOCA-emorec.zip

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/facerec_based_models/EMOCA_detail-emorec.zip -O EMOCA_detail-emorec.zip
echo "Extracting EMOCA_detail-emorec"
unzip EMOCA_detail-emorec.zip

cd .. 

mkdir -p image_based_networks 
cd image_based_networks

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/ResNet50.zip -O ResNet50.zip 
echo "Extracting ResNet 50"
unzip ResNet50.zip

wget https://download.is.tue.mpg.de/emoca/assets/EmotionRecognition/image_based_networks/SWIN-B.zip -O SWIN-B.zip
echo "Extracting SWIN B"
unzip SWIN-B.zip
cd ..


cd ..
mkdir data 
cd data
echo "Downloading example test data"
wget https://download.is.tue.mpg.de/emoca/assets/data/EMOCA_test_example_data.zip -O EMOCA_test_example_data.zip
unzip EMOCA_test_example_data.zip
echo "Example test data downloaded and extracted."


echo "Assets for  Emotion Recognition downloaded and extracted."
cd ../../gdl_apps/EmotionRecognition/demos
