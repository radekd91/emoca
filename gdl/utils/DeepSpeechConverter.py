import tensorflow as tf
from python_speech_features import mfcc
import copy
import numpy as np
import resampy


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps,
                                             input_timestamps,
                                             features[:, feat])
    return output_features


def audioToInputVector(audio, fs, numcep, numcontext):
    # Get mfcc coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep)

    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]

    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2 * numcontext + 1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    # Return results
    return train_inputs


class DeepSpeechConverter(object):

    def __init__(self, graph_fname=None, fps=60):

        if graph_fname is None:
            import os
            graph_fname = os.path.join(os.path.dirname(__file__), "..", "trained_models", "DeepSpeech", "output_graph.pb")

        # Load graph and place_holders
        # with tf.gfile.GFile(self.config['deepspeech_graph_fname'], "rb") as f:
        with tf.compat.v1.gfile.GFile(graph_fname, "rb") as f:
            self.graph_def = tf.compat.v1.GraphDef()
            self.graph_def.ParseFromString(f.read())

        self.graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.import_graph_def(self.graph_def, name="deepspeech")
        self.input_tensor = self.graph.get_tensor_by_name('input_node:0')
        self.seq_length = self.graph.get_tensor_by_name('input_lengths:0')
        self.layer_6 = self.graph.get_tensor_by_name('logits:0')

        self.n_input = 26
        self.n_context = 9
        self.output_fps = fps
        self.deep_speech_sample_rate = 50

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def convert_to_deepspeech(self, audio_sample, sample_rate, audio_window_size, audio_window_stride):
        # processed_audio = copy.deepcopy(audio)
        # with tf.compat.v1.Session(graph=self.graph) as sess:
        # for subj in audio.keys():
        #     for seq in audio[subj].keys():
        #         print('process audio: %s - %s' % (subj, seq))

        # audio_sample = audio[subj][seq]['audio']
        # sample_rate = audio[subj][seq]['sample_rate']
        resampled_audio = resampy.resample(audio_sample.astype(float), sample_rate, 16000)
        input_vector = audioToInputVector(resampled_audio.astype('int16'), 16000, self.n_input, self.n_context)

        network_output = self.sess.run(self.layer_6, feed_dict={self.input_tensor: input_vector[np.newaxis, ...],
                                                      self.seq_length: [input_vector.shape[0]]})

        # Resample network output from 50 fps to 60 fps
        audio_len_s = float(audio_sample.shape[0]) / sample_rate
        # audio_len_s = float(audio_sample.shape[1]) / sample_rate
        num_frames = int(round(audio_len_s * 60))
        network_output = interpolate_features(network_output[:, 0], 50, 60,
                                              output_len=num_frames)

        # Make windows
        zero_pad = np.zeros((int(audio_window_size / 2), network_output.shape[1]))
        network_output = np.concatenate((zero_pad, network_output, zero_pad), axis=0)
        windows = []
        for window_index in range(0, network_output.shape[0] - audio_window_size, audio_window_stride):
            windows.append(network_output[window_index:window_index + audio_window_size])

        # processed_audio[subj][seq]['audio'] = np.array(windows)
        result = np.array(windows)
        return result