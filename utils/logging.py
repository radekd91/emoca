import os
# os.system('wandb login ac022f3e88af42e979689d6a8f7617288a0ed93f')
# os.system('wandb login')

import wandb

class AbstractLogger(object):

    def add_config(self, cfg: dict):
        raise NotImplementedError()

    def watch_model(self, model):
        raise NotImplementedError()

    def log_values(self, epochm, values: dict):
        raise NotImplementedError()

    def log_image(self, epoch, images: dict):
        raise NotImplementedError()

    def log_3D_shape(self, epoch, models: dict):
        raise NotImplementedError()

    def get_experiment_id(self):
        raise NotImplementedError()


class WandbLogger(AbstractLogger):

    def __init__(self, project_name, run_name, output_foder, id=None):
        os.makedirs(output_foder)
        self.id = id
        self.wandb_run = wandb.init(project=project_name,
                   name=run_name,
                   sync_tensorboard=True,
                   dir=output_foder,
                   id=id
                   )

    def add_config(self, config: dict):
        wandb.config = config

    def watch_model(self, model):
        wandb.watch(model)

    def log_values(self, epoch: int, values: dict):
        # wandb.log(vals, step=epoch) # epoch into dict or not?
        vals = values.copy()
        # vals['epoch'] = epoch
        wandb.log(vals, step=epoch)
        # wandb.log(vals)

    def log_3D_shape(self, epoch: int, models: dict):
        shapes = {key: wandb.Object3D(value) for key, value in models.items() }
        wandb.log(shapes, step=epoch)

    def log_image(self, epoch, images: dict):
        ims = {key: wandb.Image(value) for key, value in images.items()}
        wandb.log(ims, step=epoch)

    def get_experiment_id(self):
        return self.id


import tensorboardX as tbx


class TbXLogger(AbstractLogger):

    def __init__(self, project_name, run_name, output_folder, id=None, wandb_logger=None):
        self.output_folder = output_folder
        # os.makedirs(os.path.join(output_folder, 'tbx', run_name))
        os.makedirs(os.path.join(output_folder), exist_ok=True)
        self.id = id
        self.run_name = run_name
        self.summary_writer = tbx.SummaryWriter(
            # os.path.join(output_folder, 'tbx', run_name),
            output_folder,
            # comment=run_name
        )
        self.wandb_logger = wandb_logger

    def add_config(self, config: dict):
        self.config = config
        import yaml
        with open(os.path.join(self.output_folder, "config.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        if self.wandb_logger is not None:
            self.wandb_logger.add_config(config)

    def watch_model(self, model):
        # self.summary_writer.add_graph(model=model)
        if self.wandb_logger is not None:
            self.wandb_logger.watch_model(model)

    def log_values(self, epoch: int, values: dict):
        # wandb.log(vals, step=epoch) # epoch into dict or not?
        # vals = values.copy()
        # vals['epoch'] = epoch
        # wandb.log(vals, step=epoch)
        # wandb.log(vals)
        for key, value in values.items():
            self.summary_writer.add_scalar("values/" + key, value, global_step=epoch)
            # self.summary_writer.add_scalar(self.run_name + "/" + key, value, global_step=epoch)
        # self.summary_writer.add_scalars(self.run_name, values, global_step=epoch)

    def log_3D_shape(self, epoch: int, models: dict):
        if self.wandb_logger is not None:
            self.wandb_logger.log_3D_shape(epoch, models)
        from pytorch3d.io import load_obj, load_ply
        # shapes = {key: wandb.Object3D(value) for key, value in models.items() }
        for key, value in models.items():
            if isinstance(value, str):
                fname, ext = os.path.splitext(value)
                if ext == '.ply':
                    vertices, faces = load_ply(value)
                    colors = None
                elif ext == '.obj':
                    vertices, face_data, _ = load_obj(value)
                    faces = face_data[0]
                    colors = None
                else:
                    raise ValueError("Unknown extension '%s'" % ext)
            elif isinstance(value, tuple) or isinstance(value, list):
                vertices = value[0]
                faces = value[1]
                colors = None
                if len(value) > 2:
                    colors = value[2]
            else:
                raise ValueError("Wrong shape representation")

            if len(vertices.shape) == 2:
                vertices = vertices.reshape((1, vertices.shape[0], vertices.shape[1]))

            if faces is not None and len(faces.shape) == 2:
                faces = faces.reshape((1, faces.shape[0], faces.shape[1]))

            if colors is not None and len(colors.shape) == 2:
                colors = colors.reshape((1, colors.shape[0], colors.shape[1]))

            self.summary_writer.add_mesh(#tag=self.run_name + "/" + key,
                                         # tag=key,
                                         tag="shapes/" + key,
                                         vertices=vertices, faces=faces, colors=colors,
                                         global_step=epoch)

    def log_image(self, epoch, images: dict):
        if self.wandb_logger is not None:
            self.wandb_logger.log_image(epoch, images)
        from skimage.io import imread
        for key, value in images.items():
            if isinstance(value, str):
                value = imread(value)
            self.summary_writer.add_image(
                                          # tag=self.run_name + "/" + key,
                                          # tag=key,
                                          tag= "images/" + key,
                                          img_tensor=value,
                                          global_step=epoch,
                                          dataformats='HWC')


    def get_experiment_id(self):
        return self.id


if __name__ == "__main__":
    logger = WandbLogger('test_project_name', 'test_run_name', 'test_out_folder', None)
    print("Logger initialized")