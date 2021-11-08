from gdl.models.external.Deep3DFace import Deep3DFaceWrapper
from gdl.datasets.dirty.HavenSet import HavenSet
from tqdm import auto
from torch.utils.data.dataloader import DataLoader
from util.preprocess import align_img
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import numpy as np
from skimage.io import imsave
import imageio.core.util
import sys


def from_tensor(tensor):
    image = tensor.detach().cpu().numpy().transpose([1,2,0]).clip(0,1)
    image *= 255.
    image = image.astype(np.uint8)
    return image

def main(input_folder, output_folder ):

    cfg = OmegaConf.load("../DECA/deca_conf/model/settings/deep3dface.yaml")
    model = Deep3DFaceWrapper(cfg.deep3dface)

    dataset = HavenSet(input_folder)

    workers = 8
    batch_size = 32

    dl = DataLoader(dataset, shuffle=False, num_workers=8,
                    batch_size=batch_size, drop_last=False)

    for bi, batch in enumerate(auto.tqdm(dl)):
        print(bi)
        with torch.no_grad():
            out = model(batch)
            unlit_im = model.compute_unlit_render()
        model.model.pred_vertex
        model.model.pred_tex
        model.model.pred_color
        model.model.pred_lm
        model.model.pred_mask
        model.model.pred_face

        bs = out["shapecode"].shape[0]

        for i in range(bs):
            out_path = Path(batch["image_path"][i]).relative_to(dataset.image_folder)
            sample_folder = out_path.parent / out_path.stem
            final_out_path = output_folder / sample_folder
            final_out_path.mkdir(exist_ok=True, parents=True)
            np.save(final_out_path / "vertices.npy", model.model.pred_vertex[i].detach().cpu().numpy())
            np.save(final_out_path / "texture.npy", model.model.pred_tex[i].detach().cpu().numpy())
            np.save(final_out_path / "color.npy", model.model.pred_color[i].detach().cpu().numpy())
            np.save(final_out_path / "landmarks.npy", model.model.pred_lm[i].detach().cpu().numpy())
            np.save(final_out_path / "lightcode.npy",  out["gamma"][i].detach().cpu().numpy())
            np.save(final_out_path / "shapecode.npy",
                    out["shapecode"][i].detach().cpu().numpy())
            mask = from_tensor(out["mask"][i])
            inverted_mask = (1-mask/255).astype(np.uint8)
            imsave(final_out_path / "mask.png", mask)
            color_rendering = from_tensor(unlit_im[i])
            img = from_tensor( batch["image"][i])
            imsave(final_out_path / "color_rendering.png", color_rendering  + (inverted_mask*img))
            final_rendering = from_tensor( model.model.pred_face[i])
            imsave(final_out_path / "final_rendering.png", final_rendering + (inverted_mask*img))
            imsave(final_out_path / "color_rendering_masked.png", color_rendering)
            imsave(final_out_path / "final_rendering_masked.png", final_rendering)
            imsave(final_out_path / "input_image.png", img)

            im_geometry = from_tensor(out["geometry_coarse"][i])
            imsave(final_out_path / "im_geometry.png", im_geometry + (inverted_mask*img))
            imsave(final_out_path / "im_geometry_masked.png", im_geometry)






if __name__ == '__main__':
    input_root = "/is/cluster/scratch/hfeng/light_albedo/albedo-benchmark/full_benchmark/"
    output_root = "/is/cluster/scratch/rdanecek/for_haiwen/deep3dface"

    if len(sys.argv) > 1:
        subfolders = [sys.argv[1]]
    else:
        subfolders = ["test_soft", "test_hard", "validation_soft", "validation_hard"]

    for subfolder in subfolders:
        main(Path(input_root) / subfolder, Path(output_root) / subfolder  )
