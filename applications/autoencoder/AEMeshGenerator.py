import os, sys

mv_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", ".."))
if mv_path not in sys.path:
    sys.path += [mv_path]

from MeshVista.Mesh import VistaMesh
from MeshVista.MeshGenerator import MeshGeneratorBase
from MeshVista.Vis import SurfaceWithProperties
from MeshVista.Visualization import VisType
from psbody.mesh import Mesh
import yaml
import torch
import numpy as np
from models.Coma import Coma
from datasets.coma_dataset import ComaDataset
import glob
from train_autoencoder import meta_from_config, load_model


class AEMeshGenerator(MeshGeneratorBase):

    def __init__(self, path_to_model, device=None):
        self.path_to_model = path_to_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config_file = os.path.join(path_to_model, "config.yaml")
        with open(config_file, 'r') as f:
            self.config = yaml.load(f)

        dataset_train = ComaDataset(root_dir=self.config['InputOutput']['data_dir'],
                                    dtype='train',
                                    split=self.config['DataParameters']['split'],
                                    split_term= self.config['DataParameters']['split_term'],
                                    pre_transform=None)
        self.dataset_mean = dataset_train.mean.numpy()

        # just to make sure the topology is loaded consistently
        mesh = Mesh(filename=self.config["InputOutput"]["template_fname"])
        # self.template_mesh = VistaMesh(path=self.config["InputOutput"]["template_fname"])
        self.template_mesh = VistaMesh(mesh.v, mesh.f)
        self.current_mesh = self.template_mesh.copy()

        D_t, U_t, A_t, num_nodes = meta_from_config(self.config, self.device)

        if 'num_input_features' not in self.config['ModelParameters'].keys():
            self.config['ModelParameters']['num_input_features'] = 3

        #TODO: remove this hack
        # del self.config['ModelParameters']['num_conv_filters'][0]

        self.model = Coma(self.config['ModelParameters'], D_t, U_t, A_t, num_nodes)
        self.model.to(self.device)

        all_checkpoints = glob.glob(os.path.join(path_to_model, "checkpoints", "*.pt"))
        all_checkpoints.sort()
        if len(all_checkpoints) == 0:
            raise RuntimeError("No checkpoint found")
        latest_checkpoint = all_checkpoints[-1]
        self.model_name = load_model(self.model, latest_checkpoint, self.device)
        self.model.eval()

        self.current_mesh_visualization = None

        self.latent_code = np.zeros(self.model.z)
        self.current_mesh_params = self.get_defaults().copy()
        self.lower_bounds = self.latent_code - 5
        self.upper_bounds = self.latent_code + 5
        self.regenerate_mesh()

    def regenerate_mesh(self):
        return self.generate_mesh(**self.current_mesh_params)

    def generate_mesh(self, latent_code : np.ndarray = None):
        if latent_code is not None:
            self.latent_code = latent_code

        z = torch.Tensor(self.latent_code).reshape(1, -1)
        z = z.to(self.device)

        verts = self.model.decoder(z)

        v = verts.reshape(-1,3).detach().cpu().numpy() + self.dataset_mean
        self.current_mesh.set_vertices(v)

        return self.current_mesh

    def get_current_mesh(self):
        return self.current_mesh

    def get_current_mesh_params(self):
        params = {
            "latent_code": np.copy(self.latent_code)
        }
        return params

    def get_visualizations(self):
        visualizations = []
        if self.current_mesh_visualization is None:
            self.current_mesh_visualization \
                = SurfaceWithProperties("Generated mesh", VisType.SURFACE,
                                        self.get_parameters(), self.get_defaults(), self)
            # self.current_mesh_visualization.opacity = 0.25
            # self.current_mesh_visualization.show_edges = True

        visualizations += [self.current_mesh_visualization]
        return visualizations

    def get_parameters(self):
        return {
            "latent_code": [np.ndarray, self.lower_bounds, self.upper_bounds, np.zeros(self.model.z)+0.01]
        }

    def get_defaults(self):
        return {
            "latent_code": np.zeros(self.model.z)
        }

    def get_anchor_indices(self):
        raise NotImplementedError()

    def export_surface_mesh(self, filename):
        self.current_mesh.export_surface(filename)

    def export_volumetric_mesh(self, filename):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    def deserialize(self, state):
        raise NotImplementedError()

    def get_original_input_mesh(self):
        return self.template_mesh


if __name__ == "__main__":

    # generator = AEMeshGenerator("../../results/COMA/2020_05_04_15_31_53_Coma")
    generator = AEMeshGenerator("results/COMA/2020_05_05_15_11_19_Coma")

    print("Done")

