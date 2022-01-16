# EMOCA : Emotion-Driven Monocular Face Capture and Animation 

 ![EMOCA](emoca.png)

This is the official implementation of EMOCA: Emotion-Driven Monocular Face Capture and Animation 
 ![EMOCA](EMOCA_gif_sparse_det.gif)  ![EMOCA](EMOCA_gif_sparse_rec.gif)  <!-- ![EMOCA](EMOCA_gif_sparse_rec_trans.gif)  -->

## Installation 

Follow the steps at the root of this repo. If for some reason the environment from there is not valid, create one using a `.yml` file from `envs`. After that you should be able to run the demos. 


## Demos 
In order to run the demos you'll need to download and unzip a few assets. Run `download_assets.sh` to do that. Alternatively, manually download and unzip the following assets into the `asset` folder at the root of the repo: 
- [pretrained EMOCA model](https://owncloud.tuebingen.mpg.de/index.php/s/NaGoq8Jt4BXcTDN)  
- [DECA related assets](https://owncloud.tuebingen.mpg.de/index.php/s/Wf5CbTweKE9ap46)  
- [FLAME related assets](https://owncloud.tuebingen.mpg.de/index.php/s/yZiYCGZjNw37jYw)

### Single Image Reconstruction 
If you want to run EMOCA on images, run the following
```python 
python demos/test_emoca_on_images.py --input_video <path_to_images> --output_folder <set_your_output_path> --model_name EMOCA 
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

### Training 
Coming soon


## Citation 
If you use this work in your publication, please cite the following publications (TODO: add bibtex): 
- EMOCA
- DECA 
- FLAME