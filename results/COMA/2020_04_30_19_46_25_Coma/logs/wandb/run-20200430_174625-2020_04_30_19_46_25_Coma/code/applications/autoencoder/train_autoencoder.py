import os
import argparse
import configparser
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from utils import mesh_operations
from models.Coma import Coma
from datasets.coma_dataset import ComaDataset
from utils.logging import WandbLogger
from transforms.normalize import NormalizeGeometricData
import yaml

try:
    from psbody.mesh import Mesh, MeshViewers
    psbody_available = True
except ImportError as e:
    psbody_available = False
    print("Importing MPI mesh package failed.")


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay


def read_config(fname):
    if not os.path.exists(fname):
        print('Config not found %s' % fname)
        return
    with open(fname, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_model(coma, optimizer, epoch, train_loss, val_loss, run_id, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    checkpoint['run_id'] = run_id
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def load_model(model, optimizer, device, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch_num']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    run_id = checkpoint['run_id']
    # To find if this is fixed in pytorch
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return run_id


def get_current_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


def main(args):
    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = config['InputOutput']['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    # template_mesh = load_obj(template_file_path)

    print('Loading Dataset')
    if args.data_dir:
        data_dir = args.data_dir
        config['InputOutput']['data_dir'] = data_dir
    else:
        data_dir = config['InputOutput']['data_dir']

    if hasattr(args, 'experiment_name') and args.experiment_name:
        experiment_name = args.experiment_name
    else:
        if 'experiment_name' in config.keys():
            experiment_name = config['ModelParameters']['experiment_name']
        else:
            experiment_name = config['ModelParameters']['model']

    experiment_name = get_current_time() + "_" + experiment_name
    config['InputOutput']['experiment_name'] = experiment_name

    output_dir = os.path.join(config['InputOutput']['output_dir'], experiment_name)
    batch_size = config['LearningParameters']['batch_size']
    workers_thread = config['DataParameters']['workers_thread']


    config['DataParameters']['split'] = args.split
    config['DataParameters']['split_term'] = args.split_term

    logger = WandbLogger("Coma", experiment_name, os.path.join(output_dir, 'logs'), id=experiment_name)

    normalize_transform = NormalizeGeometricData() # the normalization params (mean and std) will get loaded
    dataset_train = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(
        template_mesh, config['ModelParameters']['downsampling_factors'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading model')

    start_epoch = 1
    if config['ModelParameters']['model'] == 'Coma':
        model = Coma(dataset_train.num_features, config['ModelParameters'], D_t, U_t, A_t, num_nodes)
    # if config['ModelParameters']['model'] == 'spline':
    #     model = SplineComa(dataset_train.num_features, config, D_t, U_t, A_t, num_nodes)
    # elif config['ModelParameters']['model'] == 'edge':
    #     model = EdgeComa(dataset_train.num_features, config, D_t, U_t, A_t, num_nodes)
    else:
        raise ValueError("Unsupported model '%s'" % config['ModelParameters']['model'])
    total_epochs = config['LearningParameters']['epoch']


    lr = config['LearningParameters']['learning_rate']
    lr_decay = config['LearningParameters']['learning_rate_decay']
    weight_decay = config['LearningParameters']['weight_decay']
    opt = config['LearningParameters']['optimizer']
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    checkpoint_file = config['ModelParameters']['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        experiment_name = load_model(model, optimizer, device, checkpoint_file)
        config['InputOutput']['experiment_name'] = experiment_name
    model.to(device)


    # logger = WandbLogger("Coma", experiment_name, os.path.join(output_dir, 'logs'), id=experiment_name)
    logger.add_config(config)
    training_loop(model, optimizer, lr_scheduler, train_loader, test_loader, start_epoch, total_epochs, device, output_dir, config, logger)


def training_loop(model, optimizer, lr_scheduler,
                  train_loader, test_loader, start_epoch, total_epochs, device, output_dir, config, logger):
    print('Initializing parameters')
    template_file_path = config['InputOutput']['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    visual_out_dir = os.path.join(output_dir, 'visuals')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualize = config['InputOutput']['visualize']
    # output_dir = config['visual_output_dir']
    if visualize is True and not output_dir:
        print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
        # output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_flag = config['LearningParameters']['eval']
    val_losses, accs, durations = [], [], []

    logger.watch_model(model)

    if eval_flag:
        val_loss = evaluate(model, visual_out_dir, test_loader, template_mesh, device, visualize)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        lr_scheduler.step(epoch)

        logger.log_values(epoch, {'lr': lr_scheduler.get_lr()})

        train_loss = train_epoch(model, train_loader, len(train_loader.dataset), optimizer, device)

        logger.log_values(epoch, {'loss': train_loss})

        val_loss = evaluate(model, visual_out_dir, test_loader, template_mesh, device, visualize=visualize)

        logger.log_values(epoch, {'val_loss': val_loss})

        print('epoch ', epoch,' Train loss ', train_loss, ' Val loss ', val_loss)
        if val_loss < best_val_loss:
            save_model(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        # if opt == 'sgd':
        #     adjust_learning_rate(optimizer, lr_decay)



def train_epoch(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len_dataset


def evaluate(coma, epoch, output_dir, test_loader, template_mesh, device, logger=None, visualize=False):
    dataset = test_loader.dataset
    coma.eval()
    total_loss = 0
    if psbody_available:
        meshviewer = MeshViewers(shape=(1, 2))
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()

        if visualize and i % 100 == 0:
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()

            if psbody_available:
                result_mesh = Mesh(v=save_out, f=template_mesh.f)
                expected_mesh = Mesh(v=expected_out, f=template_mesh.f)

                result_fname = os.path.join(output_dir, 'eval_%.4d.obj' % i)
                expected_fname = os.path.join(output_dir, 'gt_%.4d.obj' % i)
                result_mesh.write_obj(result_fname)
                expected_out.write_obj(expected_fname)
                if logger is not None:
                    logger.log_3D_shape(
                        epoch=epoch,
                        models={
                          "result_%.4d" % i: result_fname,
                          "gt_%.4d" % i : expected_fname
                        })

                meshviewer[0][0].set_dynamic_meshes([result_mesh])
                meshviewer[0][1].set_dynamic_meshes([expected_mesh])
                image_fname = os.path.join(output_dir, 'comparison_%.4d.png')
                meshviewer[0][0].save_snapshot(image_fname, blocking=False)
                if logger is not None:
                    logger.log_image(
                        epoch=epoch,
                        models={
                          "comparison_%.4d" % i: image_fname,
                        })
    return total_loss/len(dataset)


if __name__ == '__main__':
    # logger = WandbLogger("Coma", 'a', os.path.join('a', 'logs'), id='aa')
    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.yaml')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
