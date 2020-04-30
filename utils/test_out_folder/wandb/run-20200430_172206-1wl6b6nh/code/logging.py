import os
os.system('wandb login ac022f3e88af42e979689d6a8f7617288a0ed93f')
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


class WandbLogger(AbstractLogger):

    def __init__(self, project_name, run_name, output_foder, id=None):
        os.makedirs(output_foder)
        wandb.init(project=project_name,
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


if __name__ == "__main__":
    logger = WandbLogger('test_project_name', 'test_run_name', 'test_out_folder', None)
    print("Logger initialized")