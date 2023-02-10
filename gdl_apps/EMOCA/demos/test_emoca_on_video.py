"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def reconstruct_video(args):
    path_to_models = args.path_to_models
    input_video = args.input_video
    model_name = args.model_name
    output_folder = args.output_folder + "/" + model_name
    image_type = args.image_type
    black_background = args.black_background
    include_original = args.include_original
    include_rec = args.include_rec
    cat_dim = args.cat_dim
    use_mask = args.use_mask
    include_transparent = bool(args.include_transparent)
    processed_subfolder = args.processed_subfolder

    mode = args.mode
    # mode = 'detail'
    # mode = 'coarse'
   
    ## 1) Process the video - extract the frames from video and detected faces
    # processed_subfolder="processed_2022_Jan_15_02-43-06"
    # processed_subfolder=None
    dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, 
        batch_size=4, num_workers=4)
    dm.prepare_data()
    dm.setup()
    processed_subfolder = Path(dm.output_dir).name

    # ## 2) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    if Path(output_folder).is_absolute():
        outfolder = output_folder
    else:
        outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)

    ## 3) Get the data loadeer with the detected faces
    dl = dm.test_dataloader()

    # ## 4) Run the model on the data
    for j, batch in enumerate (auto.tqdm( dl)):

        current_bs = batch["image"].shape[0]
        img = batch
        vals, visdict = test(emoca, img)
        for i in range(current_bs):
            # name = f"{(j*batch_size + i):05d}"
            name =  batch["image_name"][i]

            sample_output_folder = Path(outfolder) /name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            if args.save_mesh:
                save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
            if args.save_images:
                save_images(outfolder, name, visdict, i)
            if args.save_codes:
                save_codes(Path(outfolder), name, vals, i)

    ## 5) Create the reconstruction video (reconstructions overlayed on the original video)
    video_file, video_file_with_sound = dm.create_reconstruction_video(0,  rec_method=model_name, image_type=image_type, overwrite=True, 
            cat_dim=cat_dim, include_transparent=include_transparent, 
            include_original=include_original, 
            include_rec = include_rec,
            black_background=black_background, 
            use_mask=use_mask, 
            out_folder=outfolder)
    print("Video saved to: ", video_file_with_sound)

    if args.logger == "wandb":
        from gdl_apps.EMOCA.training.test_and_finetune_deca import  create_logger, project_name
        import wandb
        # project_name = 'EmotionalDeca'
        cfg_detail = conf.detail
        version = cfg_detail.inout.time
        version += "_" + cfg_detail.inout.random_id
        full_run_dir = cfg_detail.inout.full_run_dir
        name = cfg_detail.inout.name
        logger = create_logger("WandbLogger", 
                        name=name,
                        project_name=project_name,
                        #  config=OmegaConf.to_container(conf),
                        version=version,
                        save_dir=full_run_dir)

        print("Logging the video to wandb", video_file_with_sound)
        logger.experiment.log({f"test_video/{Path(input_video).stem}_{image_type}": wandb.Video(video_file_with_sound, format="mp4", caption="Reconstruction with sound")})
        
    print("Done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default=str(Path(gdl.__file__).parents[1] / "/assets/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
        choices=["geometry_detail", "geometry_coarse", "out_im_detail", "out_im_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=0, 
        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_rec', type=str2bool, default=True, 
        help="The reconstruction (non-transparent) will be in the video if True")
    parser.add_argument('--include_transparent', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--include_original', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--black_background', type=str2bool, default=False, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--use_mask', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--logger', type=str, default="", choices=["", "wandb"], help="Specify how to log the results if at all.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reconstruct_video(args)
    print("Done")


if __name__ == '__main__':
    main()
