# from test_and_finetune_deca import configure_and_finetune
import torch

from train_deca import configure_and_resume, prepare_data
from train_deca_modular import configure
from models.DECA import DecaModule
from tqdm.auto import tqdm


def main():
    # coarse_cfg_default = 'deca_finetune_coarse_cluster'
    coarse_cfg_default = 'deca_finetune_coarse'
    coarse_overrides = ['model/settings=coarse_train_expdeca',
                        # 'model.expression_backbone=deca_parallel',
                        # 'model.expression_backbone=deca_clone',
                        # 'model.expression_backbone=emonet_trainable',
                        'model.expression_backbone=emonet_static',
                        'model.exp_deca_global_pose=False',
                        'model.exp_deca_jaw_pose=True',
                        'model.useSeg=gt',
                        'model.resume_training=True', # careful!!!!!
                        'data/augmentations=default',
                        'data/datasets=coarse_data_desktop',
                        'learning.early_stopping.patience=10',
                        'model/paths=desktop',
                        'model/flame_tex=bfm_desktop',
                        ]


    detail_cfg_default = 'deca_finetune_detail'
    detail_overrides = ['model/settings=detail_train_expdeca',
                        # 'model.expression_backbone=deca_parallel',
                        # 'model.expression_backbone=deca_clone',
                        # 'model.expression_backbone=emonet_trainable',
                        'model.expression_backbone=emonet_static',
                        'model.exp_deca_global_pose=False',
                        'model.exp_deca_jaw_pose=True',
                        'model.resume_training=True', # careful!!!!!!
                        'data/augmentations=default',
                        'data/datasets=detail_data_desktop',
                        'model.train_coarse=true',
                        # 'model.train_coarse=false',
                        'model.useSeg=gt',
                        'learning.early_stopping.patience=20',
                        'model/paths=desktop',
                        'model/flame_tex=bfm_desktop',
                        ]

    # configure_and_finetune(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides)

    # run_path = ""

    # configure_and_resume(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides)


    # cfg = configure(coarse_cfg_default, coarse_overrides)
    cfg = configure(detail_cfg_default, detail_overrides)

    deca = DecaModule(cfg.model, cfg.learning, cfg.inout, "")
    deca.cuda()
    dm, _ = prepare_data(cfg)

    dm.setup()
    dl = dm.train_dataloader()

    for bi, batch in enumerate(tqdm(dl)):
        print(f"Batch index: {bi}")
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        deca.train()
        deca.training_step(batch, bi)

        deca.eval()
        deca.validation_step(batch, bi)

        if bi == 10:
            break



if __name__ == "__main__":
    main()
