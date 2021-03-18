from train_deca import configure, prepare_data
import matplotlib.pyplot as plt

def main():
    pretrain_coarse_conf = "deca_train_coarse_pretrain"
    coarse_conf = "deca_train_coarse"
    detail_conf = "deca_train_detail"
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(pretrain_coarse_conf, [],
                                       coarse_conf, [],
                                       detail_conf, []
                                       )

    cfg_detail.data.scale_min = 1.4
    cfg_detail.data.scale_max = 1.8
    cfg_detail.data.trans_scale = 0.2
    cfg_detail.learning.train_K = 1

    dm, sequence_name = prepare_data(cfg_detail)
    dm.setup()

    dataset = dm.train_dataset
    for i in range(50):
        print(i)
        sample = dataset[i]
        image = sample["image"].numpy().squeeze().transpose([1,2,0])
        plt.figure()
        plt.imshow(image)
        plt.show()
    print("Done")




if __name__ == "__main__":
    main()

