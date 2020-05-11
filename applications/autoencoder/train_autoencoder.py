import os, sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')]
#os.environ['WANDB_MODE'] = 'dryrun'
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
from utils.loggers import WandbLogger
from transforms.normalize import NormalizeGeometricData
import yaml
from utils.loggers import TbXLogger
from utils.render import render, ComaMeshRenderer
from utils.image import concatenate_image_batch_to_wide_image
import skimage.io as skio
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
from tqdm import tqdm
from utils.image import torchFloatToNpUintImage

try:
    from psbody.mesh import Mesh, MeshViewers
    psbody_available = True
except ImportError as e:
    psbody_available = False
    print("Importing MPI mesh package failed.")


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    shape = scp_matrix.shape
    i = torch.LongTensor(indices)
    if values.dtype == np.int32 or values.dtype == np.int64:
        v = torch.LongTensor(values)
        sparse_tensor = torch.sparse.LongTensor(i, v, torch.Size(shape))
    else:
        v = torch.FloatTensor(values)
        sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    # sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    # sparse_tensor = torch.sparse.Tensor(i, v, torch.Size(shape), dtype=values.dype)
    return sparse_tensor


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay


def read_config(fname):
    fname = os.path.expanduser(fname)
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_fname = os.path.join(checkpoint_dir, 'checkpoint_%.4d.pt' % epoch)
    torch.save(checkpoint, checkpoint_fname)
    return checkpoint_fname


def save_preprocessing_transforms(file, pre_transform=None, transform=None, std=None, mean=None):
    transforms = {}
    transforms['pre_transform'] = pre_transform
    transforms['transform'] = transform
    transforms['std'] = std
    transforms['mean'] = mean
    # os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(transforms, file)


def load_model(model, checkpoint_file, device, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch_num']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    run_id = checkpoint['run_id']
    # To find if this is fixed in pytorch
    return run_id


def get_current_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


def meta_from_config(config, device=None):
    template_file_path = config['InputOutput']['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    M, A, D, U = mesh_operations.generate_transform_matrices(
        template_mesh, config['ModelParameters']['downsampling_factors'])

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    return D_t, U_t, A_t, num_nodes


def main(args):
    config = read_config(args.conf)

    print('Initializing:')

    print('Loading Dataset')
    if args.data_dir:
        data_dir = args.data_dir
        config['InputOutput']['data_dir'] = data_dir
    else:
        data_dir = config['InputOutput']['data_dir']

    config['DataParameters']['split'] = args.split
    config['DataParameters']['split_term'] = args.split_term

    if hasattr(args, 'name') and args.name:
        experiment_name = args.name
    else:
        if 'experiment_name' in config.keys():
            experiment_name = config['ModelParameters']['experiment_name']
        else:
            experiment_name = config['ModelParameters']['model']
    experiment_name += "_" + config['DataParameters']['split']

    experiment_name = get_current_time() + "_" + experiment_name
    config['InputOutput']['experiment_name'] = experiment_name

    output_dir = os.path.join(config['InputOutput']['output_dir'], experiment_name)
    batch_size = config['LearningParameters']['batch_size']
    workers_thread = config['DataParameters']['workers_thread']



    normalize_transform = NormalizeGeometricData() # the normalization params (mean and std) will get loaded
    dataset_train = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    # import pickle as pkl
    # # for batch in train_loader:
    # #     break
    # # with open("test_batch.pkl", "wb") as f:
    # #     pkl.dump(batch, f)
    # with open("test_batch.pkl", "rb") as f:
    #     batch = pkl.load(f)
    #     config['ModelParameters']['num_input_features'] = 3

    print("Loading template mesh")
    template_file_path = config['InputOutput']['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    # template_mesh = load_obj(template_file_path)

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(
        template_mesh, config['ModelParameters']['downsampling_factors'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading model')

    config['ModelParameters']['num_input_features'] = dataset_train.num_features

    start_epoch = 0
    if config['ModelParameters']['model'] == 'Coma':
        model = Coma(config['ModelParameters'], D_t, U_t, A_t, num_nodes)
    # if config['ModelParameters']['model'] == 'spline':
    #     model = SplineComa(dataset_train.num_features, config, D_t, U_t, A_t, num_nodes)
    # elif config['ModelParameters']['model'] == 'edge':
    #     model = EdgeComa(dataset_train.num_features, config, D_t, U_t, A_t, num_nodes)
    else:
        raise ValueError("Unsupported model '%s'" % config['ModelParameters']['model'])
    total_epochs = config['LearningParameters']['num_epochs']

    # batch = batch.to(device)
    # model.to(device)
    # out = model(batch)
    # print("Peace")
    # exit(0)

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
        experiment_name = load_model(model, checkpoint_file, device, optimizer)
        config['InputOutput']['experiment_name'] = experiment_name
    model.to(device)

    loss_label = config['LearningParameters']['loss_function']
    if loss_label == 'l1':
        loss_function = F.l1_loss
    elif loss_label == 'l2':
        loss_function = F.mse_loss
    else:
        raise ValueError("Invalid loss function: '%s'" % loss_label)

    wandb_logger = WandbLogger("Coma", experiment_name, os.path.join(output_dir, 'logs'), id=experiment_name)
    # wandb_logger = None
    # logger = TbXLogger("Coma", experiment_name, os.path.join(output_dir, 'logs'), id=experiment_name)
    logger = TbXLogger("Coma", experiment_name, output_dir, id=experiment_name, wandb_logger=wandb_logger)
    logger.add_config(config)

    save_preprocessing_transforms(os.path.join(output_dir, "pre_transforms.pt"),
                                  pre_transform=dataset_train.pre_transform,
                                  transform=dataset_train.transform,
                                  std=dataset_train.std,
                                  mean=dataset_train.mean
                                  )
    logger.save(os.path.join(output_dir, "pre_transforms.pt"))
    train_eval(model, optimizer, lr_scheduler, loss_function, train_loader, test_loader, start_epoch, total_epochs, device, output_dir, config, logger)
    print("Training finished")
    logger.sync()


def train_eval(model, optimizer, lr_scheduler, loss_function,
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

    renderer = ComaMeshRenderer('flat', device, image_size=1024, num_views=5)

    if eval_flag:
        val_loss = evaluate(model, None, visual_out_dir, test_loader, template_mesh, device, logger=logger,
                            visualize=visualize, render_images=renderer, log_meshes=True)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []
    best_checkpoint_fname = None
    last_checkpoint_fname = None

    for epoch in range(start_epoch, total_epochs):
        print("Training for epoch ", epoch)

        logger.log_values(epoch, {'lr': lr_scheduler.get_lr()})

        train_loss = train_epoch(model, epoch, train_loader, len(train_loader.dataset), optimizer, lr_scheduler, loss_function, device)

        logger.log_values(epoch, {'loss': train_loss})


        log_meshes = epoch == start_epoch or epoch == total_epochs-1 or epoch % 10 == 0

        val_loss = evaluate(model, epoch, visual_out_dir, test_loader, template_mesh, device,
                            logger=logger, visualize=visualize, render_images=renderer, log_meshes=log_meshes)

        logger.log_values(epoch, {'val_loss': val_loss})

        print('epoch ', epoch, ' Train loss ', train_loss, ' Val loss ', val_loss)
        if val_loss < best_val_loss:
            best_checkpoint_fname = save_model(model, optimizer, epoch, train_loss, val_loss, logger.get_experiment_id(), checkpoint_dir)
            best_val_loss = val_loss

        if epoch == total_epochs-1:
            last_checkpoint_fname = save_model(model, optimizer, epoch, train_loss, val_loss, logger.get_experiment_id(), checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

    logger.save(best_checkpoint_fname)
    if last_checkpoint_fname != best_checkpoint_fname:
        logger.save(last_checkpoint_fname)


def train_epoch(model, epoch, train_loader, len_dataset, optimizer, lr_scheduler, loss_function, device):
    model.train()
    total_loss = 0
    # i = 0
    for data in tqdm(train_loader):
        # if i % 10 == 0:
        #     break
        # i+=1
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, data.y)
        total_loss += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)

    return total_loss / len_dataset


def evaluate(coma, epoch, output_dir, test_loader, template_mesh, device, logger=None,
             visualize=False, render_images=None, log_meshes=False, visualization_frequency=200):
    dataset = test_loader.dataset
    coma.eval()
    total_loss = 0
    if psbody_available:
        meshviewer = MeshViewers(shape=(1, 2))
    print("Testing")
    for i, data in tqdm(enumerate(test_loader)):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()

        if visualize and i % visualization_frequency == 0:
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()

            if psbody_available:
                result_mesh = Mesh(v=save_out, f=template_mesh.f)
                expected_mesh = Mesh(v=expected_out, f=template_mesh.f)
                if epoch is not None:
                    visual_folder = os.path.join(output_dir, "%.5d" % epoch)
                else:
                    visual_folder = os.path.join(output_dir)
                os.makedirs(visual_folder, exist_ok=True)
                if log_meshes:
                    result_fname = os.path.join(visual_folder, 'eval_%.4d.obj' % i)
                    expected_fname = os.path.join(visual_folder, 'gt_%.4d.obj' % i)
                    result_mesh.write_obj(result_fname)
                    expected_mesh.write_obj(expected_fname)
                    if logger is not None:
                        logger.log_3D_shape(
                            epoch=epoch,
                            models={
                              "result_%.4d" % i: result_fname,
                              "gt_%.4d" % i: expected_fname
                            })

                meshviewer[0][0].set_dynamic_meshes([expected_mesh])
                meshviewer[0][1].set_dynamic_meshes([result_mesh])

                # image_result_flat = render_images.render(result_fname)
                # image_gt_flat = render_images.render(expected_fname)
                image_result_flat = render_images.render([result_mesh.v, result_mesh.f])
                image_gt_flat = render_images.render([expected_mesh.v, expected_mesh.f])

                image_result_flat = concatenate_image_batch_to_wide_image(image_result_flat)
                image_gt_flat = concatenate_image_batch_to_wide_image(image_gt_flat)

                comparison = torch.cat([image_gt_flat, image_result_flat], dim=0)
                comparison = torchFloatToNpUintImage(comparison)

                skio.imsave(os.path.join(visual_folder, "comparison_%.4d.png" % i),
                            img_as_ubyte(comparison))

                if logger is not None:
                    logger.log_image(
                        epoch=epoch,
                        images={"comparison_%.4d.png" % i : comparison})

                # comparison = np.split(comparison.cpu().numpy(), indices_or_sections=comparison.shape[0], axis=0)
                # comparison = {"comparison_%.4d_flat" % (i, j) :
                #                   rescale_intensity(np.squeeze(comparison[j]*255), in_range='uint8')
                #               for j in range(len(comparison))}


                # for name, im in comparison.items():
                #     skio.imsave(os.path.join(visual_folder, name + ".png"), img_as_ubyte(im))
                # if logger is not None:
                #     logger.log_image(
                #         epoch=epoch,
                #         images=comparison)


                # image_result_smooth = render(result_fname, device=device, renderer='smooth')
                # image_gt_smooth = render(expected_fname, device=device, renderer='smooth')
                #
                # comparison = torch.cat([image_gt_smooth, image_result_smooth], dim=2)
                # comparison = np.split(comparison.cpu().numpy(), indices_or_sections=comparison.shape[0], axis=0)
                # comparison = {"comparison_%.4d_smooth_view_%.2d" % (i, j) :
                #                   rescale_intensity(np.squeeze(comparison[j]*255), in_range='uint8')
                #               for j in range(len(comparison))}
                # for name, im in comparison.items():
                #     skio.imsave(os.path.join(visual_folder, name + ".png"), img_as_ubyte(im))
                # if logger is not None:
                #     logger.log_image(
                #         epoch=epoch,
                #         images=comparison)

                # image_fname = os.path.join(visual_folder, 'comparison_%.4d.png' % i)
                # meshviewer[0][0].save_snapshot(image_fname, blocking=True)
                # if logger is not None:
                #     logger.log_image(
                #         epoch=epoch,
                #         images={
                #           # "comparison_%.4d" % i: skio.imread(image_fname),
                #           "comparison_%.4d" % i: image_fname,
                #         })
    return total_loss/len(dataset)


if __name__ == '__main__':
    # logger = WandbLogger("Coma", 'a', os.path.join('a', 'logs'), id='aa')
    # print("yo")
    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')
    parser.add_argument('-n', '--name', help='Experiment name.')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.yaml')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
