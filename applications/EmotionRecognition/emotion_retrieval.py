import os, sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datasets.FaceVideoDataset import FaceVideoDataModule, load_emotion
from tqdm import auto
import pickle as pkl


def path_to_cache(dm : FaceVideoDataModule):
    data_path = Path(dm.output_dir) / "cache" / "arrays" / "emotion"
    return data_path


def create_data_array(dm : FaceVideoDataModule, emotion_feature="emo_feat_2", force_cache_load=False):
    cache_file = path_to_cache(dm) / f"{emotion_feature}_meta.pkl"
    if not cache_file.is_file():
        print(f"Meta data not cached in {cache_file}. Starting from scratch")
        if force_cache_load:
            raise RuntimeError(f"Cache file '{cache_file}' not found and force_cache_load is set to True")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        num_samples = 0
        feature_size = None
        sample_counts = []
        file_list = []
        for vi, video in enumerate( auto.tqdm(dm.video_list)):
            video_emotion_folder = dm._get_path_to_sequence_emotions(vi)
            samples = sorted(list(video_emotion_folder.glob("*.pkl")))
            file_list += [samples]
            if feature_size is None:
                emotion_features, emotion_type = load_emotion(samples[0])
                feat = emotion_features[emotion_feature]
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

    array_path = data_path / (emotion_feature + ".memmap")

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
    video_emotion_folder = dm._get_path_to_sequence_emotions(vid_id)
    samples = sorted(list(video_emotion_folder.glob("*.pkl")))

    data_path = path_to_cache(dm)
    status_array_path = data_path / (emotion_feature + "_status.memmap")
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
        feat = emotion_features[emotion_feature]
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
    print(f"Loading emotion feataures '{emotion_feature}' into a large array.")
    # filename_list = []
    status_array_path = path_to_cache(dm) / (emotion_feature + "_status.memmap")
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
            return data
        else:
            print("Not every sequence already processed. Processing ...")

    if seq_id is None:
        for vi, video in enumerate(auto.tqdm(dm.video_list)):
            first_idx = sum(sample_counts[0:vi])
            last_idx = sum(sample_counts[0:(vi+1)])
            print(f"Processing sequence {vi}")
            fill_data_array_single_sequence(dm, data, vi, emotion_feature, first_idx, last_idx)
        with open(path_to_cache(dm) / f"{emotion_feature}_status.pkl", "wb") as f:
            pkl.dump("finished", f)
    else:
        first_idx = sum(sample_counts[0:seq_id])
        last_idx = sum(sample_counts[0:(seq_id + 1)])
        print(f"Processing sequence {seq_id}")
        fill_data_array_single_sequence(dm, data, seq_id, emotion_feature, first_idx, last_idx)
    return data#, filename_list


def load_data_array(dm, emotion_feature):
    data_path = path_to_cache(dm)
    with open(data_path / f"{emotion_feature}_meta.pkl", "rb") as f:
        num_samples = pkl.load(f)
        feature_size = pkl.load(f)
        feature_dtype = pkl.load(f)
        sample_counts = pkl.load(f)
        file_list = pkl.load(f)

    array_path = data_path / (emotion_feature + ".memmap")
    data = np.memmap(array_path,
                     dtype=feature_dtype,
                     mode='r',
                     shape=(num_samples, feature_size)
                     )
    return data, file_list


def build_database(data, filename_list):
    nbrs = NearestNeighbors(n_neighbors=30, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)



def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    # subfolder = 'processed_2020_Dec_21_00-30-03'
    subfolder = 'processed_2021_Jan_19_20-25-10'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    emotion_feature = "emo_feat_2"
    if not (path_to_cache(dm) / f"{emotion_feature}_status.pkl").is_file():
        data, sample_counts, filename_list = create_data_array(dm, emotion_feature)
        # sys.exit(0)
        fill_data_array(dm, data, sample_counts, emotion_feature)
    # else:
    data, filename_list = load_data_array(dm, emotion_feature)
    build_database(data, filename_list)



if __name__ == "__main__":
    main()
