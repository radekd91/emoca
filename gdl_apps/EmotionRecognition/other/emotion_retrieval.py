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


import os, sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule
from gdl.datasets.IO import load_emotion
from tqdm import auto
import pickle as pkl
import matplotlib.pyplot as plt
from skimage.io import imread, imsave


def path_to_cache(dm: FaceVideoDataModule):
    data_path = Path(dm.output_dir) / "cache" / "arrays" / "emotion"
    return data_path


def create_data_array(dm: FaceVideoDataModule, emotion_feature="emo_feat_2", force_cache_load=False):
    if not isinstance(emotion_feature, list):
        emotion_feature = [emotion_feature, ]
    cache_file = path_to_cache(dm) / f"{'_'.join(emotion_feature)}_meta.pkl"
    if not isinstance(emotion_feature, list):
        emotion_feature = [emotion_feature, ]
    if not cache_file.is_file():
        print(f"Meta data not cached in {cache_file}. Starting from scratch")
        if force_cache_load:
            raise RuntimeError(f"Cache file '{cache_file}' not found and force_cache_load is set to True")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        num_samples = 0
        feature_size = None
        sample_counts = []
        file_list = []
        for vi, video in enumerate(auto.tqdm(dm.video_list)):
            video_emotion_folder = dm._get_path_to_sequence_emotions(vi)
            samples = sorted(list(video_emotion_folder.glob("*.pkl")))
            file_list += [samples]
            if feature_size is None:
                feature_size = 0
                emotion_features, emotion_type = load_emotion(samples[0])
                feats = []
                for f in emotion_feature:
                    feats += [np.array([emotion_features[f]])]
                feat = np.concatenate(feats)
                feature_size = feat.size

            num_samples_video = len(samples)
            sample_counts += [num_samples_video]
            num_samples += num_samples_video

        feature_dtype = feat.dtype
        with open(cache_file, "wb") as f:
            pkl.dump(num_samples, f)
            pkl.dump(feature_size, f)
            pkl.dump(feat.dtype, f)
            pkl.dump(sample_counts, f)
            pkl.dump(file_list, f)
    else:
        print(f"Meta data cached in {cache_file}. Loading")
        with open(cache_file, "rb") as f:
            num_samples = pkl.load(f)
            feature_size = pkl.load(f)
            feature_dtype = pkl.load(f)
            sample_counts = pkl.load(f)
            file_list = pkl.load(f)
    print("Meta data loaded.")

    if num_samples == 0 or feature_size is None:
        raise RuntimeError(f"No emotion features found. num_samples={num_samples}")

    data_path = path_to_cache(dm)
    data_path.mkdir(exist_ok=True, parents=True)

    array_path = data_path / ('_'.join(emotion_feature) + ".memmap")

    if array_path.is_file():
        print(f"Opening array file {array_path}")
        data = np.memmap(array_path,
                         dtype=feature_dtype,
                         mode='r+',
                         shape=(num_samples, feature_size)
                         )
    else:
        print(f"Creating array file {array_path}")
        data = np.memmap(array_path,
                         dtype=feature_dtype,
                         mode='w+',
                         shape=(num_samples, feature_size)
                         )
    return data, sample_counts, file_list


def fill_data_array_single_sequence(dm, data, vid_id, emotion_feature, first_idx, end_idx):
    if not isinstance(emotion_feature, list):
        emotion_feature = [emotion_feature, ]

    video_emotion_folder = dm._get_path_to_sequence_emotions(vid_id)
    samples = sorted(list(video_emotion_folder.glob("*.pkl")))

    data_path = path_to_cache(dm)
    status_array_path = data_path / ("_".join(emotion_feature) + "_status.memmap")
    if status_array_path.is_file():
        status_array = np.memmap(status_array_path,
                                 dtype=np.bool,
                                 mode='r',
                                 shape=(dm.num_sequences,)
                                 )
        processed = status_array[vid_id]
        del status_array
        if processed:
            print(f"Sequence {vid_id} is already processed. Skipping")
            return
        print(f"Sequence {vid_id} is not processed. Processing ...")
    else:
        raise RuntimeError(f"Status array not found in {status_array_path}")

    for i, sample_path in enumerate(auto.tqdm(samples)):
        if first_idx + i == end_idx:
            raise RuntimeError("We were about to write to next video's slots. This should not happen."
                               f"Video = {video_emotion_folder}, first_idx={first_idx}, end_idx={end_idx}")
        emotion_features, emotion_type = load_emotion(sample_path)
        # feat = emotion_features[emotion_feature]
        feats = []
        for f in emotion_feature:
            feats += [emotion_features[f]]
        feat = np.concatenate(feats)
        data[first_idx + i, ...] = feat
    data.flush()

    status_array = np.memmap(status_array_path,
                             dtype=np.bool,
                             mode='r+',
                             shape=(dm.num_sequences,)
                             )
    status_array[vid_id] = True
    status_array.flush()
    del status_array
    # return data, samples


def fill_data_array(dm, data, sample_counts, emotion_feature, seq_id=None):
    if not isinstance(emotion_feature, list):
        emotion_feature = [emotion_feature, ]
    print(f"Loading emotion feataures '{emotion_feature}' into a large array.")
    # filename_list = []
    status_array_path = path_to_cache(dm) / ('_'.join(emotion_feature) + "_status.memmap")
    if not status_array_path.is_file():
        print(f"Status array file is not present. Creating: {status_array_path}")
        status_array = np.memmap(status_array_path,
                                 dtype=np.bool,
                                 mode='w+',
                                 shape=(dm.num_sequences,)
                                 )
        status_array[...] = False
        status_array.flush()
        del status_array
    else:
        print(f"Status array file found. Checking if everything is processed.")
        status_array = np.memmap(status_array_path,
                                 dtype=np.bool,
                                 mode='r',
                                 shape=(dm.num_sequences,)
                                 )
        all_processed = status_array.all()
        del status_array
        if all_processed:
            print("Every sequence already processed. The data array is ready")
            with open(path_to_cache(dm) / f"{'_'.join(emotion_feature)}_status.pkl", "wb") as f:
                pkl.dump("finished", f)
            return data
        else:
            print("Not every sequence already processed. Processing ...")

    if seq_id is None:
        for vi, video in enumerate(auto.tqdm(dm.video_list)):
            first_idx = sum(sample_counts[0:vi])
            last_idx = sum(sample_counts[0:(vi + 1)])
            print(f"Processing sequence {vi}")
            fill_data_array_single_sequence(dm, data, vi, emotion_feature, first_idx, last_idx)
        with open(path_to_cache(dm) / f"{'_'.join(emotion_feature)}_status.pkl", "wb") as f:
            pkl.dump("finished", f)
    else:
        first_idx = sum(sample_counts[0:seq_id])
        last_idx = sum(sample_counts[0:(seq_id + 1)])
        print(f"Processing sequence {seq_id}")
        fill_data_array_single_sequence(dm, data, seq_id, emotion_feature, first_idx, last_idx)
    return data  # , filename_list


def load_data_array(dm, emotion_feature, load_file_list=False):
    if not isinstance(emotion_feature, list):
        emotion_feature = [emotion_feature, ]

    data_path = path_to_cache(dm)
    with open(data_path / f"{'_'.join(emotion_feature)}_meta.pkl", "rb") as f:
        num_samples = pkl.load(f)
        feature_size = pkl.load(f)
        feature_dtype = pkl.load(f)
        sample_counts = pkl.load(f)
        if load_file_list:
            file_list = pkl.load(f)

    array_path = data_path / ('_'.join(emotion_feature) + ".memmap")
    data = np.memmap(array_path,
                     dtype=feature_dtype,
                     mode='r',
                     shape=(num_samples, feature_size)
                     )
    if not load_file_list:
        return data
    return data, file_list


def build_database(dm, data, emotion_feature, filename_list=None, sampling_rate=30, overwrite=False):
    filename_list_ = None
    if isinstance(filename_list, list) and isinstance(filename_list[0], list):  # nested list
        print("The list is nested. Unnesting...")
        filename_list_ = []
        for fl in filename_list:
            filename_list_ += fl
        print(" ... done")

    if sampling_rate > 1:
        data_ = data[::sampling_rate, ...]
        if filename_list_ is not None:
            filename_list_ = filename_list_[::sampling_rate]
    else:
        data_ = data
    cache_path = path_to_cache(dm)
    model_path = cache_path / ('_'.join(emotion_feature) + "_n_neighbors_model.pkl")
    res_path = cache_path / ('_'.join(emotion_feature) + "_n_neighbors.pkl")

    if not model_path.is_file() or overwrite:
        nbrs = NearestNeighbors(n_neighbors=30, algorithm='auto', n_jobs=-1).fit(data_)
        distances, indices = nbrs.kneighbors(data_, 100)

        with open(model_path, "wb") as f:
            pkl.dump(nbrs, f)
            pkl.dump(sampling_rate, f)
            pkl.dump(filename_list_, f)

        with open(res_path, "wb") as f:
            pkl.dump(distances, f)
            pkl.dump(indices, f)

    else:
        with open(model_path, "rb") as f:
            nbrs = pkl.load(f)
            sampling_rate = pkl.load(f)
            filename_list_ = pkl.load(f)

        with open(res_path, "rb") as f:
            distances = pkl.load(f)
            indices = pkl.load(f)
    return nbrs, sampling_rate, distances, indices, filename_list_


def analyze_sample(nbrs, distances, indices, filename_list):
    pass


def fix_path(path):
    return path.relative_to(path.parents[4])


def create_visualization(image_fnames, indices, captions, title, path_prefix, save_path=None, show=True):
    n = len(image_fnames)
    fig, axs = plt.subplots(1, n, figsize=(n, 2))
    plt.suptitle(title)

    for i in range(n):
        im = imread(path_prefix / image_fnames[i])
        if n > 1:
            axs[i].imshow(im)
            axs[i].set_title(captions[i], fontsize=4)
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        else:
            axs.imshow(im)
            axs.set_title(captions[i], fontsize=4)
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def analyze_model(dm, data, nn_model, sampling_rate, emotion_feature, distances, indices, filename_list):
    if not isinstance(emotion_feature, list):
        emotion_feature = [emotion_feature, ]

    N = distances.shape[0]
    M = distances.shape[1]
    # n_freq = N // 10
    nn_freq = M // 10
    save_path = Path(f"/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/"
                     f"emotion_retrieval/{'_'.join(emotion_feature)}/plots")
    save_path.mkdir(exist_ok=True, parents=True)
    for i_ in auto.tqdm(range(N)):
        i = i_ * sampling_rate
        # print(f" --- Sample {i} --- ")
        # print(f"Name: {filename_list[i]}")
        # print(f"Min distance: {distances[i].min()}")
        # print(f"Max distance: {distances[i].max()}")
        # print(f"Mean distance: {distances[i].mean()}")
        # print(f"Std distance: {distances[i].std()}")
        qry_image_fname = fix_path(Path(str(filename_list[i]).replace("emotions", "detections")))
        qry_image_fname = qry_image_fname.parent / (qry_image_fname.stem + ".png")

        # name = Path(image_fname.parents[3].name) / image_fname.parents[2].name / \
        #        image_fname.parents[1].name / image_fname.parents[0].name / image_fname.name

        name = str(qry_image_fname)
        # caption = f"Query name={name}"
        caption = f"Query"

        neighbor_indices = [-1]
        neighbor_images = [qry_image_fname]
        neighbor_captions = [caption]
        for j in range(nn_freq):
            j_ = nn_freq * j
            # print(f"Neighbor {j_}:")
            idx = indices[i, j_]
            dist = distances[i, j_]
            fname = filename_list[idx]
            # print(filename_list[idx])
            im_fname = fix_path(Path(str(fname).replace("emotions", "detections")))
            im_fname = im_fname.parent / (im_fname.stem + ".png")
            neighbor_indices += [j_]
            neighbor_images += [im_fname]
            # name = Path(im_fname.parents[3].name) / im_fname.parents[2].name / \
            #        im_fname.parents[1].name / im_fname.parents[0].name / im_fname.name
            name = str(im_fname)
            neighbor_captions += [  # f"name={name}\n"
                f"n={j_}\n"
                f"d={dist:.03f}\n"
                f"frame={im_fname.stem}\n"
                f"vid={im_fname.parent.name}\n"
                f"set={im_fname.parents[1].stem}\n"
                f"split={im_fname.parents[3].stem}\n"]
            # print(neighbor_captions[-1])

        file_path = save_path / f"{i:08d}_all.png"
        create_visualization(neighbor_images, neighbor_indices, neighbor_captions,
                             f"Neighbors for sample {i}", dm.output_dir,
                             save_path=file_path, show=False)

        neighbor_indices = [-1]
        neighbor_images = [qry_image_fname]
        neighbor_captions = [caption]
        used_videos = set([qry_image_fname.parent.name, ])
        for j in range(M):
            # print(f"Neighbor {j_}:")
            idx = indices[i, j]
            dist = distances[i, j]
            fname = filename_list[idx]
            # print(filename_list[idx])
            im_fname = fix_path(Path(str(fname).replace("emotions", "detections")))
            im_fname = im_fname.parent / (im_fname.stem + ".png")
            video_name = im_fname.parent.name

            if video_name in used_videos:
                continue
            used_videos.add(video_name)

            neighbor_indices += [j]
            neighbor_images += [im_fname]
            # name = Path(im_fname.parents[3].name) / im_fname.parents[2].name / \
            #        im_fname.parents[1].name / im_fname.parents[0].name / im_fname.name
            name = str(im_fname)
            neighbor_captions += [  # f"name={name}\n"
                f"n={j}\n"
                f"d={dist:.03f}\n"
                f"frame={im_fname.stem}\n"
                f"vid={video_name}\n"
                f"set={im_fname.parents[1].stem}\n"
                f"split={im_fname.parents[3].stem}\n"]
            if len(neighbor_indices) > 10:
                break
        file_path = save_path / f"{i:08d}_other.png"
        create_visualization(neighbor_images, neighbor_indices, neighbor_captions,
                             f"Neighbors for sample {i} from different videos", dm.output_dir,
                             save_path=file_path, show=False)


def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2021_Jan_19_20-25-10'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    emotion_feature = ["emo_feat_2"]
    # emotion_feature = ["valence", "arousal"]
    if not (path_to_cache(dm) / f"{'_'.join(emotion_feature)}_status.pkl").is_file():
        data, sample_counts, filename_list = create_data_array(dm, emotion_feature)
        # sys.exit(0)
        fill_data_array(dm, data, sample_counts, emotion_feature)
    # else:

    # overwrite_model = True
    overwrite_model = False
    if overwrite_model:
        data, filename_list = load_data_array(dm, emotion_feature, load_file_list=True)
    else:
        data, filename_list = load_data_array(dm, emotion_feature), None
    sampling_rate = 30
    nbrs, sampling_rate, distances, indices, filename_list_ = build_database(dm, data,
                                                                             emotion_feature,
                                                                             filename_list,
                                                                             sampling_rate=sampling_rate,
                                                                             overwrite=overwrite_model,
                                                                             )
    data_ = data[::sampling_rate, ...]
    analyze_model(dm, data_, nbrs, sampling_rate, emotion_feature, distances, indices, filename_list_)


if __name__ == "__main__":
    main()
