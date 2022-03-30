# EMOCA : Emotion-Driven Monocular Face Capture and Animation 

This is the official implementation of [EMOCA: Emotion-Driven Monocular Face Capture and Animation](https://emoca.is.tue.mpg.de/). EMOCA takes a single in-the-wild image as input and reconstructs a 3D face with sufficient facial expression detail to convey the emotional state of the
input image. 

<p align="center"> 
<img src="EMOCA_gif_sparse_det.gif">
<img src="EMOCA_gif_sparse_rec.gif">
</p>

<p align="center"> 
<img src="emoca.png">
</p>
<p align="center">Top row: input images. Middle row: coarse shape reconstruction. Bottom row: reconstruction with detailed displacements.<p align="center">


## Installation 

Follow the steps at the root of this repo. If for some reason the environment from there is not valid, create one using a `.yml` file from `envs`. After that you should be able to run the demos. 


## Demos 
In order to run the demos you'll need to download and unzip a few assets. Run `download_assets.sh` to do that: 
```bash 
cd demos 
bash download_assets.sh
```
<!-- Alternatively, manually download and unzip the following assets into the `asset` folder at the root of the repo: 
- [pretrained EMOCA model](https://owncloud.tuebingen.mpg.de/index.php/s/NaGoq8Jt4BXcTDN)  
- [DECA related assets](https://owncloud.tuebingen.mpg.de/index.php/s/Wf5CbTweKE9ap46)  
- [FLAME related assets](https://owncloud.tuebingen.mpg.de/index.php/s/yZiYCGZjNw37jYw) -->

### Single Image Reconstruction 
If you want to run EMOCA on images, run the following
```python 
python demos/test_emoca_on_images.py --input_folder <path_to_images> --output_folder <set_your_output_path> --model_name EMOCA 
```
The script will detect faces in every image in the folder output the results that you specify with `--save_images`, `--save_codes`, `--save_mesh` to the output folder. 

See `demos/test_emoca_on_images.py` for further details.

### Video Reconstruction 
If you want to be able to create a video of the reconstruction (like the teaser above), just pick your favourite emotional video and run the following:
```python 
python demos/test_emoca_on_video.py --input_video <path_to_your_video> --output_folder <set_your_output_path> --model_name EMOCA 
```
The script will extract the frames from the video, run face detection on it to extract cropped out faces. Then EMOCA will be run, the reconstruction renderings saved and finally 

See `demos/test_emoca_on_video.py` for further details.

## Training 

In order to train EMOCA, you need the following things: 

1) Get training data of DECA (this is crucial for training the detail stage part of EMOCA) 

2) Download AffectNet from http://mohammadmahoor.com/affectnet/ 

3) Process AffectNet using the `data/process_affectnet.py` script. (See the script for details) This dataset is crucial to train the 

5) In order to train the detailed stage, we need the "[DECA dataset](https://github.com/YadiraF/DECA)" which consists of VGGFace2, BUPT-Balanced Face and VoxCeleb images. Unfortunately, VGGFace2 is officially not available anymore and we are not allowed to distribute it. 
We are working towards switching to another face recognition dataset in place of VGGFace2 but pull requests are also welcome. We are also willing to provide assistance and advice in training EMOCA.

4) Train EMOCA. For trainng the coarse version, AffectNet is enough. In order to finetune the  detail stage you will also need the DECA dataset (or other large scale dataset that guarantees multiple images per identity)
```
python training/train_expdeca.py emoca.yaml
```

## Citation 

If you use this work in your publication, please cite the following publications:
```
@inproceedings{EMOCA:CVPR:2021,
  title = {{EMOCA}: {E}motion Driven Monocular Face Capture and Animation},
  author = {Danecek, Radek and Black, Michael J. and Bolkart, Timo},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {},
  year = {2022}
}
```

Further, as EMOCA builds on top of [DECA](https://github.com/YadiraF/DECA) and uses parts of DECA as fixed part of the model, please further cite:
```
@inproceedings{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen and Black, Michael J. and Bolkart, Timo},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)}, 
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}
```
 
 
## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://emoca.is.tue.mpg.de/license.html) file. By downloading and using the code and model you agree to the terms of this license.
