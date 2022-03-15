# Emotion Recognition 

This project accompanies the EMOCA project. Here we provide training and testing code for: 
- image-based emotion recognition networks 
    - EmoNet 
    - vanilla vision backbones (ResNet, VGG, SWIN)
- EMOCA-based emotion recognotion 

## Installation 
Follow the steps at the root of this repo. If for some reason the environment from there is not valid, create one using a `.yml` file from `envs`. After that you should be able to run the demos. 

## Pre-trained models 
All the provided models have been trained on AffectNet, either from scratch (the image-based backbones) or finetuned based on fixed pretrained face reconstruction nets. 

Use `download_assets.sh` to download and unzip them. 


## Demos 

### Single-image based emotion recognition 

If you want to run Emotion Recognition on images, run the following
```python 
python demos/test_emoca_on_images.py --input_video <path_to_images> --output_folder <set_your_output_path>  --modeltype (image|3dmm) --model_name (ResNet50|SWIN-B|EMOCA-emorec|EMOCA_detail-emorec)
```
The script will detect faces in every image in the folder output the results that you specify with `--save_images`, `--save_codes`, `--save_mesh` to the output folder. 

See `demos/test_emoca_on_images.py` for further details.

<!-- ### Video-based emotion recognition  -->


### Training 
If you want to train your own models, you will need to download the AffectNet dataset from the 

Coming soon

## Citation 
If you use this work in your publication, please cite the following publications (TODO: add bibtex): 
- EMOCA
