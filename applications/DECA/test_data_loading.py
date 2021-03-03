from test_and_finetune_deca import configure, prepare_data


def main():
    coarse_conf = "deca_finetune_coarse_emonet"
    detail_conf = "deca_finetune_detail_emonet"
    cfg_coarse, cfg_detail = configure(coarse_conf, ['data/augmentations=default'],
                                       detail_conf, ['data/augmentations=default']
                                       )
    dm, sequence_name = prepare_data(cfg_coarse)
    dm.setup()

    dataset = dm.training_set
    # image_index = 0
    # sample = dataset[image_index]
    for i in range(50):
        sample = dataset[i]
        dataset.visualize_sample(sample)
    print("Done")


if __name__ == "__main__":
    main()

