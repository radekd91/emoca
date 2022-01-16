cd ../../.. 
mkdir -p assets 
cd assets

echo "Downloading assets to run Emotion Recognition" 
wget https://owncloud.tuebingen.mpg.de/index.php/s/WHjQE7t8BE4Re56/download -O EmotionRecognition.zip
echo "Extracting  Emotion Recognition models"
unzip EmotionRecognition.zip
echo "Assets for  Emotion Recognition downloaded and extracted."
cd ../../../gdl_apps/EmotionRecognition/demos
