from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

import glob, os, sys
from pathlib import Path
import pyvista as pv
# from utils.mesh import load_mesh
# from scipy.io import wavfile
# import resampy
import numpy as np
import torch
import torchaudio
from enum import Enum
from typing import Optional, Union, List
import pickle as pkl
from collections import OrderedDict
from tqdm import tqdm


class SoundAlignment(Enum):
    START_AT = 1
    ENDS_AT = 2
    MID_AT = 3


class Emotion(Enum):
    ANGRY = 0
    DISGUSTED = 1
    EXCITED = 2
    FEARFUL = 3
    FRUSTRATED = 4
    HAPPY = 5
    NEUTRAL = 6
    SAD = 7
    SURPRISED = 8

    @staticmethod
    def fromString(s : str):
        sub = s[:3].lower()
        if sub == 'ang':
            return Emotion.ANGRY
        if sub == 'dis':
            return Emotion.DISGUSTED
        if sub == 'exc':
            return Emotion.EXCITED
        if sub == 'fea':
            return Emotion.FEARFUL
        if sub == 'fru':
            return Emotion.FRUSTRATED
        if sub == 'hap':
            return Emotion.HAPPY
        if sub == 'neu':
            return Emotion.NEUTRAL
        if sub == 'sad':
            return Emotion.SAD
        if sub == 'sur':
            return Emotion.SURPRISED
        raise ValueError("Invalid emotion string: %s" % s)


def sentenceID(s : str):
    # the filenames are named starting from 1, so make it 0 based
    return int(s[-2:])-1



class EmoSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 # output_dir,
                 root_dir,
                 output_dir,
                 processed_subfolder = None,
                 # root_mesh_dir,
                 # root_audio_dir=None,
                 mesh_fps=60,
                 sound_target_samplerate=22020,
                 sound_alignment=SoundAlignment.MID_AT,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 train_pattern="sentences?(0[0-9]|1[0-1])",
                 validation_pattern="sentences?(1[2-4])",
                 test_pattern="sentences?15",
                 dims=None
                 ):
        self.root_dir = root_dir
        # self.root_mesh_dir = root_mesh_dir
        # self.root_audio_dir = root_audio_dir
        self.root_mesh_dir = os.path.join(self.root_dir, "EmotionalSpeech_alignments_new", "seq_align")
        self.root_audio_dir = os.path.join(self.root_dir, "EmotionalSpeech_data", "audio")

        self.train_pattern = train_pattern
        self.validation_pattern = validation_pattern
        self.test_pattern = test_pattern

        self.mesh_fps = mesh_fps

        self.sound_alignment = sound_alignment
        self.sound_target_samplerate = sound_target_samplerate

        assert self.sound_target_samplerate % self.mesh_fps == 0

        if processed_subfolder is None:
            import datetime
            date = datetime.datetime.now()
            processed_folder = os.path.join(output_dir, "processed_%s" % date.strftime("%Y_%b_%d_%H-%M-%S"))
        else:
            processed_folder = os.path.join(output_dir, processed_subfolder)
        self.output_dir = processed_folder


        # To be initializaed
        self.all_mesh_paths = None
        self.all_audio_paths = None
        self.subjects2sequences = None
        self.identity_name2idx = None

        self.vertex_array = None
        self.raw_audio_array = None
        self.emotion_array = None
        self.sentence_array = None
        self.identity_array = None
        self.sequence_array = None

        #flame arrays
        self.flame_expression_params = 100
        self.fitted_vertex_array = None
        self.expr_array = None
        self.pose_array = None
        self.neck_array = None
        self.eye_array = None
        self.translation_array = None

        self.temporal_window = 16
        self.temporal_stride = 1
        self.ds_alphabet = 29

        super().__init__(train_transforms, val_transforms, test_transforms)


    @property
    def verts_array_path(self):
        return os.path.join(self.output_dir, "verts.memmap")

    @property
    def raw_audio_array_path(self):
        return os.path.join(self.output_dir, "raw_audio.memmap")

    @property
    def ds_array_path(self):
        return os.path.join(self.output_dir, "ds.memmap")

    @property
    def emotion_array_path(self):
        return os.path.join(self.output_dir, "emotion.pkl")

    @property
    def identity_array_path(self):
        return os.path.join(self.output_dir, "identity.pkl")

    @property
    def sentence_array_path(self):
        return os.path.join(self.output_dir, "sentence.pkl")

    @property
    def sequence_array_path(self):
        return os.path.join(self.output_dir, "sequence.pkl")

    @property
    def fitted_vertex_array_path(self):
        return os.path.join(self.output_dir, "verts_fitted.memmap")

    @property
    def expr_array_path(self):
        return os.path.join(self.output_dir, "expression.memmap")

    @property
    def pose_array_path(self):
        return os.path.join(self.output_dir, "pose.memmap")

    @property
    def neck_array_path(self):
        return os.path.join(self.output_dir, "neck.memmap")

    @property
    def eye_array_path(self):
        return os.path.join(self.output_dir, "eye.memmap")

    @property
    def translation_array_path(self):
        return os.path.join(self.output_dir, "translation.memmap")

    @property
    def sequence_length_array_path(self):
        return os.path.join(self.output_dir, "sequence_length.pkl")

    @property
    def templates_path(self):
        return os.path.join(self.output_dir, "templates.pkl")

    @property
    def metadata_path(self):
        return os.path.join(self.output_dir, "metadata.pkl")


    @property
    def personalized_template_paths(self):
        return [ os.path.join(self.root_dir, "EmotionalSpeech_alignments_new", "personalization", "personalized_template", subject + ".ply")
                 for subject in self.subjects2sequences.keys() ]

    @property
    def num_audio_samples_per_scan(self):
        return int(self.sound_target_samplerate / self.mesh_fps)

    @property
    def num_samples(self):
        return len(self.all_mesh_paths)

    @property
    def num_verts(self):
        return self.subjects_templates[0].number_of_points

    @property
    def version(self):
        return 1

    def _index_from_mesh_path(self, meshpath):
        return int(meshpath.stem[-6:])

    def _load_templates(self):
        self.subjects_templates = [pv.read(template_path) for template_path in self.personalized_template_paths]

    def prepare_data(self, *args, **kwargs):
        outdir = Path(self.output_dir)

        # is dataset already processed?
        if outdir.is_dir():
            print("The dataset is already processed. Loading")
            self._loadMeta()
            self._load_templates()
            self._load_arrays()
            print("Dataset loaded")
            return

        self._gather_data()

        self._load_templates()
        # create data arrays
        self.vertex_array = np.memmap(self.verts_array_path, dtype=np.float32, mode='r+',
                                      shape=(self.num_samples,3*self.num_verts))
        self.raw_audio_array = np.memmap(self.raw_audio_array_path, dtype=np.float32, mode='r+', shape=(self.num_samples, self.num_audio_samples_per_scan))

        self.emotion_array = np.zeros(dtype=np.int32, shape=(self.num_samples, 1))
        self.sentence_array = np.zeros(dtype=np.int32, shape=(self.num_samples, 1))
        self.identity_array = np.zeros(dtype=np.int32, shape=(self.num_samples, 1))
        self.sequence_array = np.zeros(dtype=np.int32, shape=(self.num_samples, 1))
        self.sequence_length_array = np.zeros(dtype=np.int32, shape=(self.num_samples, 1))

        # populate data arrays
        self._process_data()

        with open(self.emotion_array_path, "wb") as f:
            pkl.dump(self.emotion_array, f)
        with open(self.sentence_array_path, "wb") as f:
            pkl.dump(self.sentence_array, f)
        with open(self.identity_array_path, "wb") as f:
            pkl.dump(self.identity_array, f)
        with open(self.sequence_array_path, "wb") as f:
            pkl.dump(self.sequence_array, f)
        with open(self.sequence_length_array_path, "wb") as f:
            pkl.dump(self.sequence_length_array, f)
        # with open(self.templates_path, "wb") as f:
        #     pkl.dump(self.subjects_templates, f)

        self._saveMeta()

        # close data arrays
        self._cleanup_memmaps()

        # self._load_arrays()

    def _saveMeta(self):
        with open(self.metadata_path, "wb") as f:
            pkl.dump(self.version, f)
            pkl.dump(self.all_mesh_paths, f)
            pkl.dump(self.all_audio_paths, f)
            pkl.dump(self.subjects2sequences, f)
            pkl.dump(self.identity_name2idx, f)
            pkl.dump(self.sound_alignment, f)

    def _loadMeta(self):
        with open(self.metadata_path, "rb") as f:
            version = pkl.load(f)
            self.all_mesh_paths = pkl.load(f)
            self.all_audio_paths = pkl.load(f)
            self.subjects2sequences = pkl.load(f)
            self.identity_name2idx = pkl.load(f)
            self.sound_alignment = pkl.load(f)

    def _gather_data(self, exist_ok=False):
        print("Processing dataset")
        Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)
        root_mesh_path = Path(self.root_mesh_dir)
        # root_audio_path = Path(self.root_audio_dir)

        pattern = root_mesh_path / "*"
        subjects = sorted([os.path.basename(dir) for dir in glob.glob(pattern.as_posix()) if os.path.isdir(dir)])

        self.identity_name2idx = OrderedDict(zip(subjects, range(len(subjects))))
        self.subjects2sequences = OrderedDict()
        self.all_mesh_paths = []

        print("Discovering data")
        for subject in subjects:
            print("Found subject: '%s'" % subject)
            subject_path = root_mesh_path / subject
            sequences = sorted([dir.name for dir in subject_path.iterdir() if dir.is_dir()])
            # sequences = sorted([os.path.basename(dir) for dir in glob.glob(subject_path.as_posix()) if os.path.isdir(dir)])
            seq2paths = OrderedDict()
            for sequence in tqdm(sequences):
                audio_file = Path(self.root_audio_dir) / subject / "scanner" / (sequence + ".wav")


                if not audio_file.is_file():
                    # skip this sequence, the file is missing
                    print("'%s' is missing. Skipping" % audio_file)
                    continue

                mesh_paths = sorted(list((subject_path / sequence).glob("*.ply")))

                # if len(mesh_paths) == int(mesh_paths[-1].stem[-6:]):
                #     print("Missing a mesh file in sequence '%s/%s'. Skipping" % (subject_path, sequence))
                #     continue

                if len(mesh_paths) == 0:
                    print("Missing all mesh files in sequence '%s/%s'. Skipping" % (subject_path, sequence))
                    continue

                relative_mesh_paths = [path.relative_to(self.root_mesh_dir) for path in mesh_paths]
                seq2paths[sequence] = relative_mesh_paths
                self.all_mesh_paths += relative_mesh_paths

                # if self.root_audio_dir is not None:
                #     audio_file = root_audio_path / subject / "scanner" / (sequence + ".wav")
                #     # sample_rate, audio_data = wavfile.read(audio_file)
                #     audio_data, sample_rate = torchaudio.load(audio_file)
                #     if sample_rate != self.sound_target_samplerate:
                #         # audio_data_resampled = resampy.resample(audio_data.astype(np.float64), sample_rate, self.sound_target_samplerate)
                #         audio_data_resampled = torchaudio.transforms.Resample(sample_rate, self.sound_target_samplerate)(audio_data[0, :].view(1, -1))
                #         num_sound_samples = audio_data_resampled.shape[1]
                #         samples_per_scan = self.num_audio_samples_per_scan
                #
                #         num_meshes_in_sequence = len(mesh_paths)
                #         assert ((num_meshes_in_sequence)*samples_per_scan >= num_sound_samples and
                #                 (num_meshes_in_sequence-1)*samples_per_scan <= num_sound_samples)
                #
                #         audio_data_aligned = torch.zeros((1, samples_per_scan*num_meshes_in_sequence), dtype=audio_data_resampled.dtype)
                #         if self.sound_alignment == SoundAlignment.START_AT:
                #             # padd zeros to the end
                #             start_at = 0
                #         elif self.sound_alignment == SoundAlignment.ENDS_AT:
                #             # pad zeros to the beginning
                #             start_at = self.mesh_fps
                #         elif self.sound_alignment == SoundAlignment.MID_AT:
                #             start_at = int(self.mesh_fps / 2)
                #             assert self.mesh_fps % 2 == 0
                #         else:
                #             raise ValueError("Invalid sound alignment '%s' " % str(self.sound_alignment))
                #         audio_data_aligned[:, start_at:start_at+audio_data_resampled.shape[1]] = audio_data_resampled[:,...]

            self.subjects2sequences[subject] = seq2paths

    def _process_data(self):
        mesh_idx = 0
        sequence_idx = 0
        sequence_lengths = []

        self.all_audio_paths = []
        print("Processing discovered data")
        with tqdm(total=self.num_samples) as pbar:

            for subject_name, sequences in self.subjects2sequences.items():
                for seq_name, meshes in sequences.items():
                    print("Starting processing sequence: '%s' of subject '%s'" % (seq_name, subject_name))

                    sentence_number = sentenceID(seq_name)

                    # num_meshes_in_sequence = len(meshes)
                    num_meshes_in_sequence = self._index_from_mesh_path(meshes[-1])

                    audio_subpath = Path(subject_name) / "scanner" / (seq_name + ".wav")
                    audio_file = Path(self.root_audio_dir) / audio_subpath
                    self.all_audio_paths += [audio_subpath]
                    # sample_rate, audio_data = wavfile.read(audio_file)
                    audio_data, sample_rate = torchaudio.load(audio_file)

                    if sample_rate != self.sound_target_samplerate:
                        # audio_data_resampled = resampy.resample(audio_data.astype(np.float64), sample_rate, self.sound_target_samplerate)
                        audio_data_resampled = torchaudio.transforms.Resample(sample_rate, self.sound_target_samplerate)(
                            audio_data[0, :].view(1, -1))
                    else:
                        audio_data_resampled = audio_data

                    num_sound_samples = audio_data_resampled.shape[1]
                    samples_per_scan = self.num_audio_samples_per_scan

                    # if not (num_meshes_in_sequence * self.num_audio_samples_per_scan >= num_sound_samples and
                    #         (num_meshes_in_sequence - 1) * self.num_audio_samples_per_scan <= num_sound_samples):
                    #     print("Num sound samples: %d" % num_sound_samples)
                    #     print("Expected range: %d - %d" %
                    #           (num_meshes_in_sequence * self.num_audio_samples_per_scan, (num_meshes_in_sequence-1) * self.num_audio_samples_per_scan)
                    #           )
                    # assert ((num_meshes_in_sequence) * self.num_audio_samples_per_scan >= num_sound_samples and
                    #         (num_meshes_in_sequence - 1) * self.num_audio_samples_per_scan <= num_sound_samples)

                    aligned_array_size = samples_per_scan * num_meshes_in_sequence

                    audio_data_aligned = torch.zeros((1, aligned_array_size),
                                                     dtype=audio_data_resampled.dtype)

                    if self.sound_alignment == SoundAlignment.START_AT:
                        # padd zeros to the end
                        start_at = 0
                    elif self.sound_alignment == SoundAlignment.ENDS_AT:
                        # pad zeros to the beginning
                        start_at = self.mesh_fps
                    elif self.sound_alignment == SoundAlignment.MID_AT:
                        start_at = int(self.mesh_fps / 2)
                        assert self.mesh_fps % 2 == 0
                    else:
                        raise ValueError("Invalid sound alignment '%s' " % str(self.sound_alignment))
                    length = min(audio_data_resampled.shape[1], audio_data_aligned.shape[1] - start_at)
                    audio_data_aligned[:, start_at:(start_at + length)] = audio_data_resampled[:, :length]

                    print("Starting processing sequence ID %d: '%s' of subject '%s'" % (sequence_idx, seq_name, subject_name))
                    old_index = -1
                    for i, mesh_name in enumerate(meshes):
                        mesh_path = Path(self.root_mesh_dir) / mesh_name
                        mesh = pv.read(mesh_path)
                        index = self._index_from_mesh_path(mesh_path) - 1

                        if index != old_index+1:
                            print("Skipping frames between %d and %d due to missing frames" % (old_index, index))

                        self.vertex_array[mesh_idx, :] = np.reshape(mesh.points, newshape=(1, -1))
                        # self.raw_audio_array[mesh_idx, :] = audio_data_aligned[0, i * self.num_audio_samples_per_scan:(i + 1) * self.num_audio_samples_per_scan].numpy()
                        self.raw_audio_array[mesh_idx, :] = audio_data_aligned[0, index * self.num_audio_samples_per_scan:(index + 1) * self.num_audio_samples_per_scan].numpy()

                        self.emotion_array[mesh_idx, 0] = Emotion.fromString(seq_name).value
                        self.identity_array[mesh_idx, 0] = self.identity_name2idx[subject_name]
                        self.sentence_array[mesh_idx, 0] = sentence_number
                        self.sequence_array[mesh_idx, 0] = sequence_idx

                        old_index = index

                        mesh_idx += 1
                        pbar.update()

                    print("Done processing sequence ID %d: '%s' of subject '%s'" % (sequence_idx, seq_name, subject_name))
                    sequence_lengths += [len(meshes)]
                    sequence_idx += 1

        self.sequence_length_array = np.array(sequence_lengths, dtype=np.int32)

        self._raw_audio_to_deepspeech()

        self._fit_flame()

    def _fit_flame(self, visualize=False, specify_indentity_indices=None):
        from applications.FLAME.fit import load_FLAME, fit_FLAME_to_registered

        self.fitted_vertex_array = np.memmap(self.fitted_vertex_array_path, dtype=np.float32, mode='r+',
                                        shape=(self.num_samples, 3 * self.num_verts))
        self.expr_array = np.memmap(self.expr_array_path, dtype=np.float32, mode='r+',
                                    shape=(self.num_samples, self.flame_expression_params))

        self.pose_array = np.memmap(self.pose_array_path, dtype=np.float32, mode='r+',
                               shape=(self.num_samples, 6))

        self.neck_array = np.memmap(self.neck_array_path, dtype=np.float32, mode='r+',
                               shape=(self.num_samples, 3))

        self.eye_array = np.memmap(self.eye_array_path, dtype=np.float32, mode='r+',
                              shape=(self.num_samples, 6))

        self.translation_array = np.memmap(self.translation_array_path, dtype=np.float32, mode='r+',
                                      shape=(self.num_samples, 3))

        if specify_indentity_indices is None:
            specify_indentity_indices = list(range(len(self.subjects_templates)))

        # for id, mesh in enumerate(self.subjects_templates):
        for id in specify_indentity_indices:
            # verts = torch.from_numpy(mesh.points)
            mesh = self.subjects_templates[id]

            print("Beginning to process mesh %d" % id)
            frames = np.where(self.identity_array == id)[0]
            # frames = frames[:10]

            flame = load_FLAME('neutral', expression_params=self.flame_expression_params, v_template=mesh.points)

            verts = self.vertex_array[frames, ...].reshape(frames.size, -1, 3)
            target_verts = np.split(verts, verts.shape[0], 0)

            fitted_verts, shape, expr, pose, neck, eye, trans = fit_FLAME_to_registered(flame, target_verts, fit_shape=False, visualize=visualize)

            self.fitted_vertex_array[frames, ...] = np.reshape(fitted_verts, newshape=(frames.size, -1))
            self.expr_array[frames, ...] = expr
            self.pose_array[frames, ...] = pose
            self.neck_array[frames, ...] = neck
            self.eye_array[frames, ...] = eye
            self.translation_array[frames, ...] = trans

            print("Finished processing mesh %d" % id)

        print("FLAME fitting finished")


    def _raw_audio_to_deepspeech(self, audio_scaler=32500):
        from utils.DeepSpeechConverter import DeepSpeechConverter
        ah = DeepSpeechConverter('/home/rdanecek/Workspace/Repos/voca/ds_graph/output_graph.pb')
        self.ds_array = np.memmap(self.ds_array_path, dtype='float32', mode='r+',
                                         shape=(self.num_samples, self.temporal_window, self.ds_alphabet))
        for si in tqdm(range(self.sequence_length_array.size)):
            idxs = np.where(self.sequence_array == si)[0]
            audio_data_ds = self.raw_audio_array[idxs, ...].reshape(-1)
            #32500 - was caused by torchaudio scaling
            self.ds_array[idxs, ...] = ah.convert_to_deepspeech(audio_data_ds * audio_scaler, self.sound_target_samplerate,
                                                                self.temporal_window, self.temporal_stride)


    def _cleanup_memmaps(self):
        if self.vertex_array is not None:
            del self.vertex_array
            self.vertex_array = None
        if self.raw_audio_array is not None:
            del self.raw_audio_array
            self.raw_audio_array = None
        if self.ds_array is not None:
            del self.ds_array
            self.ds_array = None

    def __del__(self):
        self._cleanup_memmaps()

    def _load_arrays(self):
        # load data arrays in read mode

        self.vertex_array = np.memmap(self.verts_array_path, dtype='float32', mode='r', shape=(self.num_samples, self.num_verts*3))
        self.raw_audio_array = np.memmap(self.raw_audio_array_path, dtype='float32', mode='r', shape=(self.num_samples, self.num_audio_samples_per_scan))
        self.ds_array = np.memmap(self.ds_array_path, dtype='float32', mode='r', shape=(self.num_samples, self.temporal_window, self.ds_alphabet))

        with open(self.emotion_array_path, "rb") as f:
            self.emotion_array = pkl.load(f)

        with open(self.sentence_array_path, "rb") as f:
            self.sentence_array = pkl.load(f)

        with open(self.sequence_array_path, "rb") as f:
            self.sequence_array = pkl.load(f)

        with open(self.identity_array_path, "rb") as f:
            self.identity_array = pkl.load(f)

        with open(self.sequence_length_array_path, "rb") as f:
            self.sequence_length_array = pkl.load(f)

        # with open(self.templates_path, "rb") as f:
        #     self.subjects_templates = pkl.load(f)

        #load FLAME arrays
        #TODO: fix the try catch
        try:
            self.fitted_vertex_array = np.memmap(self.fitted_vertex_array_path, dtype=np.float32, mode='r',
                                            shape=(self.num_samples, 3 * self.num_verts))
        except Exception:
            pass
        try:
            self.expr_array = np.memmap(self.expr_array_path, dtype=np.float32, mode='r',
                                        shape=(self.num_samples, self.flame_expression_params))
        except Exception:
            pass
        try:
            self.pose_array = np.memmap(self.pose_array_path, dtype=np.float32, mode='r',
                                   shape=(self.num_samples, 6))
        except Exception:
            pass

        try:
            self.neck_array = np.memmap(self.neck_array_path, dtype=np.float32, mode='r',
                               shape=(self.num_samples, 3))
        except Exception:
            pass
        try:
            self.eye_array = np.memmap(self.eye_array_path, dtype=np.float32, mode='r',
                              shape=(self.num_samples, 6))
        except Exception:
            pass

        try:
            self.translation_array = np.memmap(self.translation_array_path, dtype=np.float32, mode='r',
                                      shape=(self.num_samples, 3))
        except Exception:
            pass

    def setup(self, stage: Optional[str] = None):
        # is dataset already processed?
        if not Path(self.output_dir).is_dir():
            raise RuntimeError("The folder with the processed not not found")

        print("Loading the dataset")
        self._loadMeta()
        self._load_templates()
        self._load_arrays()
        print("Dataset loaded")

        import re
        self.training_indices = np.array([i for i, name in enumerate(self.all_mesh_paths) if re.search(self.train_pattern, str(name)) ], dtype=np.int32)
        self.val_indices = np.array([i for i, name in enumerate(self.all_mesh_paths) if re.search(self.validation_pattern, str(name)) ], dtype=np.int32)
        self.test_indices = np.array([i for i, name in enumerate(self.all_mesh_paths) if re.search(self.test_pattern, str(name)) ], dtype=np.int32)

        if self.training_indices.size == 0:
            raise ValueError("The training set is empty.")
        if self.val_indices.size == 0:
            raise ValueError("The validation set is empty.")
        if self.test_indices.size == 0:
            raise ValueError("The test set is empty.")
        self.raw_audio_array = np.memmap(self.raw_audio_array_path, dtype='float32', mode='r',
                                         shape=(self.num_samples, self.num_audio_samples_per_scan))
        if len(set(self.training_indices.tolist()).intersection(set(self.val_indices.tolist()))) != 0 or \
            len(set(self.training_indices.tolist()).intersection(set(self.test_indices.tolist()))) != 0 or \
            len(set(self.val_indices.tolist()).intersection(set(self.test_indices.tolist()))) != 0 :
            raise ValueError("The training, validation and test set are not disjoint!")


        if self.training_indices.size + self.val_indices.size + self.test_indices.size != self.num_samples:
            forgotten_samples = sorted(list(set(range(self.num_samples)).difference(set(self.training_indices.tolist() + self.val_indices.tolist() + self.test_indices.tolist()))))
            forgotten_example = self.all_mesh_paths[forgotten_samples[0]]
            raise ValueError("Train, test and vaiidation samples do not include all the samples in the dataset. "
                             "Some samples got forgotten. For instance '%s'" % forgotten_example)

        print("The training set contains %d/%d samples, %.1f%% of the entire dataset " % (self.training_indices.size, self.num_samples, 100*self.training_indices.size/self.num_samples))
        print("The validation set contains %d/%d samples, %.1f%% of the entire dataset " % (self.val_indices.size, self.num_samples, 100*self.val_indices.size/self.num_samples))
        print("The test set contains %d/%d samples, %.1f%% of the entire dataset " % (self.test_indices.size, self.num_samples, 100*self.test_indices.size/self.num_samples))

        self.dataset_train = EmoSpeechDataset(self, self.training_indices)
        self.dataset_val = EmoSpeechDataset(self, self.val_indices)
        self.dataset_test = EmoSpeechDataset(self, self.test_indices)


    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=64, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [DataLoader(self.dataset_val, batch_size=64), ]

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [DataLoader(self.dataset_test, batch_size=64), ]

    def create_dataset_video(self, filename=None, use_flame_fits=False, render_into_notebook=False):
        import pyvistaqt as pvqt
        import cv2

        if filename is None:
            if use_flame_fits:
                filename = os.path.join(self.output_dir, "video_flame.mp4")
            else:
                filename = os.path.join(self.output_dir, "video.mp4")

        mesh = pv.read(self.personalized_template_paths[0])

        # camera = [(0.01016587953526246, -0.10710759494716658, 0.7154280129463031),
        #         (-0.00125928595662117, -0.032396093010902405, -0.03774198144674301),
        #         (-0.012844951984790388, 0.9950148277606915, 0.09889640916064582)]
        # camera = [(0.017643774223258905, -0.10476257652816602, 0.7149231439996583),
        #          (-0.03735078070331948, -0.002708379456682772, -0.03309953157283428),
        #          (-0.024679802806785948, 0.9902746786078623, 0.13691956851200532)]
        camera = [(0.01406612981243684, -0.0032565289381143343, 0.7221936777551122),
                 (-0.04353638534524899, 0.007080320524320914, -0.03249333231780988),
                 (-0.03270108662524808, 0.9993341416403784, 0.01618370431688236)]
        if render_into_notebook:
            pl = pv.Plotter(notebook=True)
        else:
            pl = pvqt.BackgroundPlotter()
        pl.set_background([0,0,0])
        pl.camera_position = camera
        actor = pl.add_mesh(mesh)

        if render_into_notebook:
            pl.show(use_ipyvtk=True)
        from time import sleep
        textActor = pl.add_text("")

        height, width, layers = pl.image.shape
        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.mesh_fps, (width, height))

        for i in tqdm(range(self.num_samples)):
            if use_flame_fits:
                vertices = np.reshape(self.vertex_array[i,...], newshape=(-1,3))
            else:
                vertices = np.reshape(self.vertex_array[i, ...], newshape=(-1, 3))
            mesh.points[...] = vertices
            textActor.SetText(2, str(self.all_mesh_paths[i].parent.parent))
            textActor.SetText(3, str(self.all_mesh_paths[i].stem))
            pl.render()
            im = pl.image
            video.write(im)
            # import matplotlib.pyplot as plt
            # plt.imshow(im)
            # sleep(0.05)

        video.release()

    def create_dataset_audio(self, filename=None):
        if filename is None:
            filename = os.path.join(self.output_dir, "video.mp4")
        audio_filename = os.path.splitext(str(filename))[0] + ".wav"
        audio_tensor = torch.Tensor(self.raw_audio_array).view(1,-1)
        torchaudio.save(audio_filename, audio_tensor.clamp(-1,1), self.sound_target_samplerate)


    def combine_video_audio(self, filaneme=None, video=None, audio=None):
        if filaneme is None:
             filename = os.path.join(self.output_dir, "video_with_sound.mp4")
        if video is None:
            video = os.path.join(self.output_dir, "video.mp4")
        if audio is None:
            audio = os.path.join(self.output_dir, "video.wav")
        import ffmpeg
        video = ffmpeg.input(video)
        audio = ffmpeg.input(audio)
        out = ffmpeg.output(video, audio, filename, vcodec='copy', acodec='aac', strict='experimental')
        out.run()

        # import moviepy.editor as mpe
        # my_clip = mpe.VideoFileClip(video)
        # audio_background = mpe.AudioFileClip(audio, fps=self.sound_target_samplerate)
        # final_clip = my_clip.set_audio(audio_background)
        # final_clip.write_videofile(filename,fps=self.mesh_fps)


class EmoSpeechDataset(Dataset):

    def __init__(self, dm : EmoSpeechDataModule, indices):
        self.vertex_array = dm.vertex_array
        self.audio_array = dm.raw_audio_array
        self.subjects_templates = dm.subjects_templates

        self.emotion_array = dm.emotion_array
        self.sentence_array = dm.sentence_array
        self.identity_array = dm.identity_array
        self.sequence_array = dm.sequence_array

        self.mesh_paths = dm.all_mesh_paths

        self.indices = indices

    def __getitem__(self, index):
        i = self.indices[index]
        sample = {
            "mesh_path": [str(self.mesh_paths[i]),],
            "vertices" : torch.from_numpy(self.vertex_array[i,...]),
            "faces": torch.from_numpy(self.subjects_templates[0].faces),
            "emotion": torch.from_numpy(self.emotion_array[i]),
            "identity": torch.from_numpy(self.identity_array[i]),
            "sentence": torch.from_numpy(self.emotion_array[i]),
            "sequence": torch.from_numpy(self.sequence_array[i])
        }
        return sample

    def __len__(self):
        return self.indices.size



def main():
    root_dir = "/home/rdanecek/Workspace/mount/project/emotionalspeech/EmotionalSpeech/"
    processed_dir = "/home/rdanecek/Workspace/mount/scratch/rdanecek/EmotionalSpeech/"
    subfolder = "processed_2020_Dec_09_00-30-18"

    # sample_rate, audio_data = wavfile.read(audio_file)
    # audio_data, sample_rate = torchaudio.load(Path(root_dir) / "EmotionalSpeech_data/audio/EmotionalSpeech_171213_50034_TA/scanner/ang_sentence01.wav")
    # torchaudio.save("test.wav", audio_data, sample_rate)
    # audio_data_resampled = torchaudio.transforms.Resample(sample_rate, 22020)(
    #     audio_data[0, :].view(1, -1))
    # torchaudio.save("test_resampled.wav", audio_data_resampled, 22020)

    dm = EmoSpeechDataModule(root_dir, processed_dir, subfolder)
    dm.prepare_data()
    # dm._raw_audio_to_deepspeech()
    # dm.setup()
    # dm.create_dataset_video()
    # dm.create_dataset_audio()
    # dm.combine_video_audio()
    # # sample = dm[0]

    train_dl = dm.train_dataloader()
    for batch in train_dl:
        print(batch)
        break

    print("Peace out")


def main2():
    root_dir = "/home/rdanecek/Workspace/mount/project/emotionalspeech/EmotionalSpeech/"
    processed_dir = "/home/rdanecek/Workspace/mount/scratch/rdanecek/EmotionalSpeech/"
    subfolder = "processed_2020_Dec_09_00-30-18"

    dm = EmoSpeechDataModule(root_dir, processed_dir, subfolder)
    dm.prepare_data()
    dm.setup()
    #
    idxs = np.where(dm.sequence_array == 0)[0]
    audio_data_ds = dm.raw_audio_array[idxs, ...].reshape(1,-1)[:,30:]

    from scipy.io import wavfile

    audio_file = '/home/rdanecek/Workspace/mount/project/emotionalspeech/EmotionalSpeech/EmotionalSpeech_data/audio/EmotionalSpeech_171213_50034_TA/scanner/ang_sentence01.wav'
    # audio_file = '/home/rdanecek/Workspace/Repos/voca/audio/test_sentence.wav'

    sample_rate_wav, audio_data_wav = wavfile.read(audio_file)
    audio_data_torch, sample_rate_torch = torchaudio.load(audio_file)

    target_sample_rate = 22020

    import resampy
    audio_data_resampled_wav = resampy.resample(audio_data_wav.astype(np.float64), sample_rate_wav, target_sample_rate)
    audio_data_resampled_torch = torchaudio.transforms.Resample(sample_rate_torch, target_sample_rate)(audio_data_torch[0, :].view(1, -1))

    with open('/home/rdanecek/Workspace/Repos/voca/processed_test_audio.pkl', 'rb') as f:
        processed_original_audio = pkl.load(f)

    from utils.DeepSpeechConverter import DeepSpeechConverter
    # ah = DeepSpeechConverter('/home/rdanecek/Workspace/Repos/voca/ds_graph/output_graph.pb')
    ah = DeepSpeechConverter()
    seq1_ds_wav = ah.convert_to_deepspeech(audio_data_resampled_wav, target_sample_rate, 16, 1)
    seq1_ds_torch = ah.convert_to_deepspeech(audio_data_resampled_torch.numpy()[0], target_sample_rate, 16, 1)
    seq1_ds_torch_scaled = ah.convert_to_deepspeech(audio_data_resampled_torch.numpy()[0]*32500, target_sample_rate, 16, 1)
    seq1_ds_scaled = ah.convert_to_deepspeech(audio_data_ds[0,30:audio_data_resampled_torch.shape[1]]*32500, target_sample_rate, 16, 1)

    pass



    pass
    # import deepspeech
    # model_file_path = '/home/rdanecek/Workspace/Data/deepspeech/deepspeech-0.9.2-models.pbmm'
    # print(os.path.exists(model_file_path))
    # model = deepspeech.Model(model_file_path)
    # # model.enableExternalScorer(scorer_file_path)
    # beam_width = 500
    # model.setBeamWidth(beam_width)
    # print(model.sampleRate())
    # model.stt()

def main3():
    root_dir = "/home/rdanecek/Workspace/mount/project/emotionalspeech/EmotionalSpeech/"
    processed_dir = "/home/rdanecek/Workspace/mount/scratch/rdanecek/EmotionalSpeech/"
    subfolder = "processed_2020_Dec_09_00-30-18"
    dm = EmoSpeechDataModule(root_dir, processed_dir, subfolder)
    dm.prepare_data()
    # dm._fit_flame()
    # print(dm.fitted_vertex_array[76334])
    print(dm.fitted_vertex_array[116798])
    pass


if __name__ == "__main__":
    # main()
    # main2()
    main3()

