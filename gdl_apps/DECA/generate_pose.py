from gdl.models.DecaFLAME import FLAME
from omegaconf import DictConfig
import torch
import pytorch3d.transforms as trans
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt

def main():
    cfg = {}
    cfg["flame_model_path"] = "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl"
    cfg["flame_lmk_embedding_path"] = "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy"
    cfg["n_shape"] = 100
    cfg["n_tex"] = 50
    cfg["n_exp"] = 50
    cfg["n_pose"] = 6
    cfg["n_cam"] = 3
    cfg["n_light"] = 27
    cfg["batch_size"] = 1
    cfg["use_3D_translation"] = 1

    cfg = DictConfig(cfg)
    flame = FLAME(cfg)

    angles = torch.deg2rad(torch.tensor(np.arange(.0, 50., 10.)))

    # angle_idx = 0
    angle_idx = 1
    # angle_idx = 2

    shape = torch.zeros(1, cfg.n_shape)
    exp = torch.zeros(1, cfg.n_exp)
    pose = torch.zeros(1, 6)
    # pose[:, 3] = torch.deg2rad(torch.tensor([ 20.]))
    # pose[:, 5] = torch.deg2rad(torch.tensor([ 40.]))

    faces = flame.faces_tensor.detach().cpu().numpy()

    for angle in angles:
        jaw_pose = pose[:, 3:]
        jaw_pose[:,angle_idx] = angle
        jaw_pose = trans.quaternion_to_axis_angle(trans.matrix_to_quaternion(trans.euler_angles_to_matrix(jaw_pose, "XYZ")))
        pose[:, 3:] = jaw_pose
        verts = flame(shape, exp, pose)

        vertices = verts[0].detach().cpu().numpy()
        # if visualize:
        #     import pyvista as pv
        #     import pyvistaqt as pvqt
        #
        target_mesh = pv.PolyData(vertices[0], np.hstack([np.ones(shape=(faces.shape[0],1), dtype=np.int32 )*3, faces]))
        pl = pvqt.BackgroundPlotter(auto_update=True)
        pl.add_mesh(target_mesh, opacity=0.5)
        final_mesh = target_mesh.copy(deep=True)
        pl.add_mesh(final_mesh)
        pl.show()


if __name__ == "__main__":
    main()
