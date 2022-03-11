import numpy as np
import torch
from pytorch3d.io import load_ply, save_ply
from torch_geometric.data import Data
from gdl.utils.mesh_operations import get_vert_connectivity
import glob, os, sys
from tqdm import tqdm
from torch_geometric.data import Dataset, InMemoryDataset
from gdl.transforms.normalize import NormalizeGeometricData


class ComaDataset(InMemoryDataset):

    def __init__(self, root_dir,
                 dtype='train', split='sliced', split_term='sliced',
                 nVal = 100, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_term = split_term
        self.nVal = nVal
        self.transform = transform
        self.pre_transform = pre_transform
        self.data_file = glob.glob(self.root_dir + '/*/*/*.ply')
        self.data_file.sort()

        self.root_dir = root_dir # backwards compatibility hack, remove in the future
        self.root = root_dir # backwards compatibility hack, remove in the future

        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'val':
            data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        pre_transform_path = self.processed_paths[4]

        # backwards compatiblity hack
        if not os.path.exists(pre_transform_path) and os.path.exists(data_path):
            from gdl.transforms.normalize import NormalizeGeometricData
            norm_path = self.processed_paths[3]
            norm_dict = torch.load(norm_path)
            self.mean, self.std = norm_dict['mean'], norm_dict['std']
            pre_transform = NormalizeGeometricData(self.mean, self.std)
            torch.save(pre_transform, self.processed_paths[4])

        super(ComaDataset, self).__init__(root_dir, transform, pre_transform)

        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)

        pre_transform_path = self.processed_paths[4]
        if os.path.exists(pre_transform_path):
            pre_transform = torch.load(pre_transform_path)
        else:
            # backwards compatiblity hack, remove in the future
            from gdl.transforms.normalize import NormalizeGeometricData
            pre_transform = NormalizeGeometricData(self.mean, self.std)
            torch.save(self.pre_transform, self.processed_paths[4])

        if self.pre_transform is None:
            self.pre_transform = pre_transform
        else:
            if self.pre_transform.__repr__() != pre_transform.__repr__():
                print("[BIG WARNING] The specified pre_transform is not the same as the one "
                      "used to actually preprocess the dataset!!!")

        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    # def __getitem__(self, item):
    #     mesh_path = self.data_files[item]
    #     verts, faces = load_ply(mesh_path)
    #     sample = {
    #         'mesh_path': mesh_path,
    #         'vertices': verts,
    #         'faces': faces,
    #     }
    #     return sample

    # def __len__(self):
    #     return len(self.data_files)

    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt', 'pre_transform.pt']
        processed_files = [self.split_term+'_'+pf for pf in processed_files]
        return processed_files

    def process(self):
        from psbody.mesh import Mesh
        train_data, val_data, test_data = [], [], []
        train_vertices = []
        for idx, data_file in tqdm(enumerate(self.data_file)):
            mesh = Mesh(filename=data_file)
            mesh_verts = torch.Tensor(mesh.v)
            adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

            if self.split == 'sliced':
                if idx % 100 <= 10:
                    test_data.append(data)
                elif idx % 100 <= 20:
                    val_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

            elif self.split == 'expression':
                import os
                if os.name == 'nt':
                    split = data_file.split('\\')[-2]
                else:
                    split = data_file.split('/')[-2]
                if split == self.split_term:
                    test_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

            elif self.split == 'identity':
                import os
                if os.name == 'nt':
                    split = data_file.split('\\')[-3]
                else:
                    split = data_file.split('/')[-3]
                if split == self.split_term:
                    test_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)
            else:
                raise Exception('sliced, expression and identity are the only supported split terms')

        if self.split != 'sliced':
            val_data = test_data[-self.nVal:]
            test_data = test_data[:-self.nVal]

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_transform.mean is None:
                    self.pre_transform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_transform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])
        torch.save(self.pre_transform, self.processed_paths[4])


def prepare_sliced_dataset(path):
    ComaDataset(path, pre_transform=NormalizeGeometricData())


def prepare_expression_dataset(path):
    test_exps = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up', 'mouth_down',
                 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']
    for exp in test_exps:
        ComaDataset(path, split='expression', split_term=exp, pre_transform=NormalizeGeometricData())

def prepare_identity_dataset(path):
    test_ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170811_03274_TA',
                'FaceTalk_170904_00128_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170913_03279_TA',
                'FaceTalk_170728_03272_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170811_03275_TA',
                'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170915_00223_TA']

    for ids in test_ids:
        ComaDataset(path, split='identity', split_term=ids, pre_transform=NormalizeGeometricData())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')

    args = parser.parse_args()
    split = args.split
    data_dir = args.data_dir
    if split == 'sliced':
        prepare_sliced_dataset(data_dir)
    elif split == 'expression':
        prepare_expression_dataset(data_dir)
    elif split == 'identity':
        prepare_identity_dataset(data_dir)
    else:
        raise Exception("Only sliced, expression and identity split are supported")

#
# if __name__ == "__main__":
#     dataset = ComaDataset('../data/COMA')
#     sample = dataset[0]
#     sample = dataset[1]
#     sample = dataset[-1]
#     print(sample)
