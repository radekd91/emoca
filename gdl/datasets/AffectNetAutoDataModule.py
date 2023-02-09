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

from gdl.datasets.AffectNetDataModule import *
from pytorch_lightning import LightningDataModule

class AffectNetAutoDataModule(AffectNetDataModule):

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "automatic"
        kwargs["use_processed"] = False
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.df = self.df[self.df["expression"] != "Neutral"]
        print("Pass")
        self.training_set = self._new_training_set()
        self.training_set.use_processed = False
        self.validation_set = new_affectnet(self.dataset_type)(self.image_path, self.val_dataframe_path,
                                                               self.image_size, self.scale,
                                        None, ignore_invalid=self.ignore_invalid,
                                        ring_type=self.ring_type,
                                        ring_size=1,
                                        ext=self.processed_ext,
                                                               use_gt=self.use_gt,
                                        )
        self.validation_set.use_processed = False
        self.test_set = new_affectnet(self.dataset_type)(self.image_path, self.val_dataframe_path, self.image_size, self.scale,
                                    None, ignore_invalid= self.ignore_invalid,
                                  ring_type=self.ring_type,
                                  ring_size=1,
                                  ext=self.processed_ext,
                                                         use_gt=self.use_gt,
                                  )
        self.validation_set.use_processed = False

class AffectNetAutoTestDataModule(AffectNetAutoDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.use_gt = False

    def setup(self, stage=None):
        self.val_dataframe_path = Path(self.input_dir.parent / "Automatically_annotated_file_list" / "representative.csv" )
        self.validation_set = new_affectnet(self.dataset_type)(self.image_path, self.val_dataframe_path,
                                                               self.image_size, self.scale,
                                        None, ignore_invalid=self.ignore_invalid,
                                        ring_type=self.ring_type,
                                        ring_size=1,
                                        ext=self.processed_ext,
                                                               use_gt=self.use_gt,
                                        )
        self.validation_set.use_processed = False

        self.test_set = new_affectnet(self.dataset_type)(self.image_path, self.val_dataframe_path, self.image_size, self.scale,
                                    None, ignore_invalid= self.ignore_invalid,
                                  ring_type=self.ring_type,
                                  ring_size=1,
                                  ext=self.processed_ext,
                                                         use_gt=self.use_gt,
                                  )
        self.test_set.use_processed = False

if __name__ == '__main__':
    # df_dir = Path("/ps/project/EmotionalFacialAnimation/data/affectnet/Automatically_Annotated/Automatically_annotated_file_list")
    # df = pd.read_csv(df_dir / "automatically_annotated.csv")
    # # sample_representative_set(df, df_dir / "representative.csv")
    # sample_representative_set(df, df_dir / "representative2.csv", num_per_bin=1)
    dm = AffectNetAutoDataModule(
    # dm = AffectNetAutoTestDataModule(
        # dm = AffectNetDataModule(
        # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
        # "/ps/project_cifs/EmotionalFacialAnimation/data/affectnet/",
        "/ps/project/EmotionalFacialAnimation/data/affectnet/",
        # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
        # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
        "/is/cluster/work/rdanecek/data/affectnet/",
        # processed_subfolder="",
        processed_ext=".jpg",
        mode="auto",
        scale=1.25,
        image_size=224,
        # bb_center_shift_x=0,
        # bb_center_shift_y=-0.3,
        ignore_invalid=True,
        # ignore_invalid="like_emonet",
        # ring_type="gt_expression",
        # ring_type="gt_va",
        # ring_type="emonet_feature",
        # ring_size=4,
        # augmentation=augmenter,
        # use_clean_labels=True
        # dataset_type="AffectNetWithMGCNetPredictions",
        # dataset_type="AffectNetWithExpNetPredictions",
    )

    dm.prepare_data()
    dm.setup()

    dataset = dm.training_set
    # dataset = dm.validation_set
    # dataset = dm.test_set

    for i in range(len(dataset)):
        sample = dataset[i]
        if AffectNetExpressions(sample["affectnetexp"].item()) != AffectNetExpressions.Contempt:
            print(AffectNetExpressions(sample["affectnetexp"].item()))
            continue
        # print(AffectNetExpressions(sample["affectnetexp"].item()))
        print(sample["va"])
        dataset.visualize_sample(sample)

    print("Done")
