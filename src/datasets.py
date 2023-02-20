import os
import json
import glob
import math
import random
import numpy as np
import itertools
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader


class Datasets():

    def __init__(self, config):
        self.config = config

        self.meta_folder = os.path.join(config.root_path, "meta")
        if self.config.dataset == "thumos14":
            self.datasets_folder = os.path.join(config.dataset_root_path, "THUMOS14")
            self.target_path = os.path.join(self.meta_folder, "thumos14.json")
            self.class_label_path = os.path.join(self.meta_folder, "thumos14_classes.txt")
        elif self.config.dataset == "activitynet":
            self.datasets_folder = os.path.join(config.dataset_root_path, "ActivityNet/v1.3")
            self.target_path = os.path.join(self.meta_folder, "activity_net.v1.3.min.json")
            self.class_label_path = os.path.join(self.meta_folder, "activitynet_classes.txt")

        self.frames_folder = os.path.join(self.datasets_folder, "frames")
        if self.config.dataset == "thumos14":
            self.features_folder = os.path.join(self.datasets_folder,
                                                "Kinetics_Data_{}_0301".format(self.config.dataset),
                                                "features")
        else:
            self.features_folder = os.path.join(self.datasets_folder,
                                                "Finetuned_Data_{}_0427".format(self.config.dataset),
                                                "features")

        self.training_data_folder = os.path.join(self.datasets_folder, "training")
        self.validation_data_folder = os.path.join(self.datasets_folder, "validation")
        self.testing_data_folder = os.path.join(self.datasets_folder, "testing")

        self.category_dic = self._getCategoryDic()
        self.label_dic = self._getLabelDic()
        self.number_of_classes = len(self.category_dic)

        with open(self.target_path, "r") as fp:
            self.meta_dic = json.load(fp)

    def _getCategoryDic(self):
        categories = {}
        with open(self.class_label_path, "r") as fp:
            while True:
                line = fp.readline()
                splits = line[:-1].split()
                if len(splits) < 2:
                    break
                category = splits[0]
                class_number = int(splits[1])
                categories[category] = class_number

        return categories

    def _getLabelDic(self):
        labels = dict()
        with open(self.class_label_path, "r") as fp:
            while True:
                line = fp.readline()
                splits = line[:-1].split()
                if len(splits) < 2:
                    break
                category = splits[0]
                class_number = splits[1]
                labels[class_number] = category

        return labels

    def getDataset(self, mode, dataset_type=None):
        if mode == "train":
            train = self.Train(self)
            validation = self.Validation(self)
            return train, validation
        elif mode == "test":
            test = self.Test(self)
            return test
        elif mode == "make":
            make = self.Make(self, dataset_type)
            return make

    class Train(Dataset):

        def __init__(self, datasets):
            self.datasets = datasets

            print("Converting Json Train Data to Tensor Data ...")
            json_file_path = os.path.join(self.datasets.meta_folder,
                                          "{}_train_data.json".format(self.datasets.config.dataset))

            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as fp:
                    tf_data = json.load(fp)
            else:
                print("There is no json file. Make the json file")

                videos = glob.glob(os.path.join(self.datasets.training_data_folder, "*"))
                tf_data = list()
                for v_i, video in enumerate(videos):
                    video_fps = self.datasets.video_fps

                    identity = video.split("/")[-1].split(".")[-2]
                    frames = len(glob.glob(os.path.join(self.datasets.frames_folder, identity, "images", "*")))

                    annotations = self.datasets.meta_dic["database"][identity]["annotations"]

                    if not frames:
                        continue

                    segments_string = ""
                    for annotation in annotations:
                        target = self.datasets.category_dic[annotation["label"].replace(" ", "_")]

                        segment = annotation["segment"]

                        start_index = max(1, int(math.floor(segment[0] * video_fps)))
                        end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                        if end_index - start_index + 1 >= self.datasets.config.feature_frame_step_size:
                            segments_string += "{} {} {} ".format(target, start_index, end_index)

                    if len(segments_string):
                        segments_string = segments_string[:-1]

                        video = "{} {} {}".format(identity, frames, segments_string)
                        tf_datum = video
                        tf_data.append(tf_datum)

                    print("VIDEO {}: {:05d}/{:05d} Done".format(identity, v_i + 1, len(videos)))

                with open(json_file_path, "w") as fp:
                    json.dump(tf_data, fp, indent=4, sort_keys=True)

            self.tf_data = tf_data
            self.data_count = len(self.tf_data)
            print("Making Train Dataset Object ... {} Instances".format(len(tf_data)))
            batch_size = self.datasets.config.batch_size * self.datasets.config.num_gpus
            self.dataloader = DataLoader(self, batch_size=batch_size, shuffle=True,
                                         num_workers=self.datasets.config.num_workers, drop_last=True,
                                         pin_memory=False, prefetch_factor=self.datasets.config.prefetch_factor)

        def __len__(self):
            return len(self.tf_data)

        def __getitem__(self, index):

            splits = self.tf_data[index].split(" ")

            identity = splits[0]
            frame_length = int(splits[1])
            segment_strings = splits[2:]

            background_segments = list()
            foreground_segments = list()
            start_boundary_segments = list()
            end_boundary_segments = list()

            previous_end_index = 0
            for segment_index in range(len(segment_strings) // 3):
                target = int(segment_strings[segment_index * 3])
                start_index = int(segment_strings[segment_index * 3 + 1])
                end_index = int(segment_strings[segment_index * 3 + 2])

                # boundary segments
                # note that we add boundary segments as the possible starting frame indices.
                # we can sample a training sequence by beginning
                # from one of the frame indices specified in an interval

                # start boundary segment
                start_boundary_start_index = \
                    max(start_index - self.datasets.config.temporal_width + 1, 1)

                start_boundary_end_index = start_index - 1

                if start_boundary_end_index - start_boundary_start_index >= 0:
                    start_boundary_segments.append([start_boundary_start_index, start_boundary_end_index,
                                                    ["start", start_index, target]])

                # end boundary segment
                end_boundary_start_index = \
                    max(max(end_index - self.datasets.config.temporal_width + 2, 1),
                        start_index)

                end_boundary_end_index = end_index

                if end_boundary_end_index - end_boundary_start_index >= 0:
                    end_boundary_segments.append([end_boundary_start_index, end_boundary_end_index,
                                                  ["end", end_index, target]])

                # background segment
                background_start_index = previous_end_index + 1
                background_end_index = start_index - 1

                if background_end_index - background_start_index + 1 >= 1:
                    background_segments.append([background_start_index, background_end_index])

                # foreground segment
                if end_index - start_index + 1 >= 1:
                    foreground_segments.append([start_index, end_index, target])

                # if the foreground segment is the last one, add the final background segment
                if segment_index == len(segment_strings) / 3 - 1:
                    background_start_index = min(end_index + 1, frame_length)
                    background_end_index = frame_length

                    if background_end_index - background_start_index + 1 >= 1:
                        background_segments.append([background_start_index, background_end_index])

                previous_end_index = end_index

            entire_framewise_targets = np.zeros(dtype=np.int64, shape=(frame_length))
            foreground_segments.sort()
            shuffled_segments = list(foreground_segments)
            random.shuffle(shuffled_segments)
            for segment in shuffled_segments:
                start_index, end_index, target = segment
                entire_framewise_targets[start_index - 1:end_index - 1 + 1] = target

            if self.datasets.config.use_random_crop:
                random_crop_index = random.choice(range(1, self.datasets.config.crop_length + 1))
                random_flip_index = random.choice(range(2))
                feature_path = \
                    os.path.join(self.datasets.features_folder,
                                 identity,
                                 "{}_C{}_F{}_features.npy".format(
                                     identity,
                                     random_crop_index,
                                     random_flip_index))
            else:
                random_crop_index = self.datasets.config.crop_length // 2 + 1
                random_flip_index = 0
                feature_path = \
                    os.path.join(self.datasets.features_folder,
                                 identity,
                                 "{}_C{}_F{}_features.npy".format(
                                     identity,
                                     random_crop_index,
                                     random_flip_index))
            kinetics_features = np.load(feature_path)
            feature_length = len(kinetics_features)

            scaled_segments = np.asarray(shuffled_segments, dtype=np.float32)
            scaled_segments[:, :2] = (scaled_segments[:, :2] - 1.0) / (frame_length - 1) * (feature_length - 1)
            scaled_segments = scaled_segments.round().astype(np.int32)
            scaled_feature_targets = np.zeros(dtype=np.float32,
                                              shape=(feature_length, self.datasets.number_of_classes))
            scaled_feature_targets[..., 0] = 1.0
            for segment in scaled_segments:
                start_index, end_index, target = segment
                scaled_feature_targets[start_index:end_index + 1, 0] = 0.0
                scaled_feature_targets[start_index:end_index + 1, target] = 1.0

            target_segment = random.choice(scaled_segments.tolist())
            target_start_index = target_segment[0]
            target_end_index = target_segment[1]
            # target_length = target_end_index - target_start_index + 1

            # # pad features and targets to indicate the boundary of the video
            # scaled_feature_targets = np.array(scaled_feature_targets, dtype=np.float32)
            # kinetics_features = np.concatenate([kinetics_features, np.zeros_like(kinetics_features[0:1])], axis=0)
            # dummy_onehot = np.zeros(dtype=np.float32, shape=(self.datasets.config.number_of_classes))
            # dummy_onehot[0] = 1.0
            # scaled_feature_targets = \
            #     np.concatenate([scaled_feature_targets, np.expand_dims(dummy_onehot, axis=0)], axis=0)
            # feature_length += 1

            if self.datasets.config.dataset == "activitynet":
                # sampled_length = feature_length
                sampled_length = random.choice(range(max(feature_length // 4, 2), feature_length + 1))
                # start_range = range(1)
            else:
                sampled_length = self.datasets.config.feature_width
                # start_range = range(target_start_index,
                #                     max(target_end_index - target_length // 4, target_start_index) + 1, 1)

            # start_range = range(target_start_index,
            #                     max(target_end_index - target_length // 4, target_start_index) + 1, 1)
            # start_range = range(target_start_index,
            #                     max(target_end_index - target_length, target_start_index) + 1, 1)
            start_range = range(target_start_index - sampled_length + 1, target_end_index + 1, 1)
            # start_range = range(feature_length)

            sampled_start_index = random.choice(start_range)
            sampled_end_index = sampled_start_index + sampled_length
            sampled_indices = np.arange(sampled_start_index, sampled_end_index, 1, dtype=np.int64)

            features = kinetics_features[sampled_indices % feature_length]
            action_targets = scaled_feature_targets[sampled_indices % feature_length]

            target_slices = list()
            this_targets = action_targets
            this_target_sequence = np.argmax(this_targets, axis=-1)
            run = {"class_number": this_target_sequence[0], "indices": []}
            slices, expect, index = [run], None, 0
            for index, target in enumerate(this_target_sequence):
                if (target == expect) or (expect is None):
                    run["indices"].append(index)
                else:
                    run = {"class_number": target, "indices": [index]}
                    slices.append(run)
                expect = target

            for slice in slices:
                if slice["class_number"] >= 1:
                    target_slices.append([slice["indices"][0], slice["indices"][-1], slice["class_number"]])

            target_slices = np.asarray(target_slices, dtype=np.float32)
            target_slices[:, :2] = target_slices[:, :2] / (sampled_length - 1) * \
                                   (self.datasets.config.feature_width - 1)
            target_slices = target_slices.round().astype(np.int32)

            if self.datasets.config.dataset == "activitynet":
                feature_length = len(features)
                features = interp1d(np.arange(feature_length), features, axis=0, kind="linear")
                features = features(
                    np.linspace(0, feature_length - 1,
                                self.datasets.config.feature_width, dtype=np.float32))
                action_targets = np.zeros(dtype=np.float32,
                                          shape=(self.datasets.config.feature_width,
                                                 self.datasets.number_of_classes))
                action_targets[..., 0] = 1.0
                for segment in target_slices:
                    start_index, end_index, target = segment
                    action_targets[start_index:end_index + 1, 0] = 0.0
                    action_targets[start_index:end_index + 1, target] = 1.0

            features = np.array(features, dtype=np.float32)

            copy_paste = random.random() < self.datasets.config.copypaste_prob
            if copy_paste:
                while True:
                    source_splits = random.choice(self.tf_data).split(" ")
                    source_identity = source_splits[0]
                    source_frame_length = int(source_splits[1])
                    source_segment_strings = source_splits[2:]

                    if self.datasets.config.use_random_crop:
                        random_crop_index = random.choice(range(1, self.datasets.config.crop_length + 1))
                        random_flip_index = random.choice(range(2))
                        feature_path = \
                            os.path.join(self.datasets.features_folder,
                                         source_identity,
                                         "{}_C{}_F{}_features.npy".format(
                                             source_identity,
                                             random_crop_index,
                                             random_flip_index))
                    else:
                        random_crop_index = self.datasets.config.crop_length // 2 + 1
                        random_flip_index = 0
                        feature_path = \
                            os.path.join(self.datasets.features_folder,
                                         source_identity,
                                         "{}_C{}_F{}_features.npy".format(
                                             source_identity,
                                             random_crop_index,
                                             random_flip_index))
                    source_kinetics_features = np.load(feature_path)

                    source_segments = list()
                    for segment_index in range(len(source_segment_strings) // 3):
                        target = int(source_segment_strings[segment_index * 3])
                        start_index = int(source_segment_strings[segment_index * 3 + 1])
                        end_index = int(source_segment_strings[segment_index * 3 + 2])

                        # foreground segment
                        if end_index - start_index + 1 >= self.datasets.config.feature_frame_step_size * 2:
                            source_segments.append([start_index, end_index, target])

                    if len(source_segments):
                        break

                num_source = np.random.randint(1, len(source_segments) + 1)
                sampled_segments = random.sample(source_segments, num_source)
                sampled_segments = np.array(sampled_segments)
                sampled_segments[:, :2] = \
                    np.array(np.floor((sampled_segments[:, :2] - 1) /
                                      (self.datasets.config.feature_frame_step_size)), dtype=np.int32)
                sampled_segments = sampled_segments.tolist()
                for segment in sampled_segments:
                    start_feature_index = segment[0]
                    end_feature_index = segment[1]
                    class_index = segment[2]

                    sampled_features = source_kinetics_features[start_feature_index:end_feature_index + 1]
                    sampled_length = np.random.randint(2, self.datasets.config.feature_width // 4)

                    feature_length = len(sampled_features)
                    sampled_features = interp1d(np.arange(feature_length), sampled_features, axis=0, kind="linear")
                    kinetics_features = sampled_features(
                        np.linspace(0, feature_length - 1, sampled_length, dtype=np.float32))

                    sampled_index = np.random.randint(0, self.datasets.config.feature_width - sampled_length + 1)
                    features[sampled_index:sampled_index + sampled_length] = kinetics_features
                    this_target_vectors = np.zeros(dtype=np.float32,
                                                   shape=(sampled_length, self.datasets.number_of_classes))
                    this_target_vectors[..., class_index] = 1.0
                    action_targets[sampled_index:sampled_index + sampled_length] = this_target_vectors

                target_slices = list()
                this_targets = action_targets
                this_target_sequence = np.argmax(this_targets, axis=-1)
                run = {"class_number": this_target_sequence[0], "indices": []}
                slices, expect, index = [run], None, 0
                for index, target in enumerate(this_target_sequence):
                    if (target == expect) or (expect is None):
                        run["indices"].append(index)
                    else:
                        run = {"class_number": target, "indices": [index]}
                        slices.append(run)
                    expect = target

                for slice in slices:
                    if slice["class_number"] >= 1:
                        target_slices.append([slice["indices"][0], slice["indices"][-1], slice["class_number"]])

                target_slices = np.asarray(target_slices, dtype=np.float32)
                target_slices[:, :2] = target_slices[:, :2] / (sampled_length - 1) * \
                                       (self.datasets.config.feature_width - 1)
                target_slices = target_slices.round().astype(np.int32)

            W, C = action_targets.shape

            target_slices = np.asarray(target_slices, dtype=np.float32)
            target_slices[:, :2] = target_slices[:, :2] / (self.datasets.config.feature_width - 1)
            target_slices[:, :2] = np.clip(target_slices[:, :2], 0.0, 1.0)
            detection_targets = np.zeros(dtype=np.float32, shape=(W, 4))
            if len(target_slices):
                foreground_segments = target_slices
                detection_targets[:len(foreground_segments), 0] = 1.0
                if self.datasets.config.use_classification:
                    detection_targets[:len(foreground_segments), 1] = foreground_segments[:, -1] - 1
                else:
                    detection_targets[:len(foreground_segments), 1] = 0
                detection_targets[:len(foreground_segments), 2:] = \
                    np.stack(((foreground_segments[:, 0] + foreground_segments[:, 1]) / 2,
                              (foreground_segments[:, 1] - foreground_segments[:, 0])), axis=-1)

            features = torch.from_numpy(features).permute(1, 0)
            targets = torch.from_numpy(detection_targets)

            return features, targets, identity, frame_length

    class Validation(Dataset):

        def __init__(self, datasets):
            self.datasets = datasets

            print("Converting Json Validation Data to Tensor Data ...")
            json_file_path = os.path.join(self.datasets.meta_folder,
                                          "{}_validation_data.json".format(self.datasets.config.dataset))

            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as fp:
                    tf_data = json.load(fp)
            else:
                print("There is no json file. Make the json file")
                videos = glob.glob(os.path.join(self.datasets.validation_data_folder, "*"))
                tf_data = list()
                for v_i, video in enumerate(videos):
                    video_fps = self.datasets.video_fps

                    identity = video.split("/")[-1].split(".")[-2]
                    frames = len(
                        glob.glob(os.path.join(self.datasets.frames_folder, identity, "images", "*")))

                    annotations = self.datasets.meta_dic["database"][identity]["annotations"]

                    if not frames:
                        continue

                    segments_string = ""
                    for annotation in annotations:
                        target = self.datasets.category_dic[annotation["label"].replace(" ", "_")]
                        segment = annotation["segment"]

                        start_index = max(1, int(math.floor(segment[0] * video_fps)))
                        end_index = min(int(math.ceil(segment[1] * video_fps)), frames)

                        if end_index - start_index + 1 >= self.datasets.config.feature_frame_step_size:
                            segments_string += "{} {} {} ".format(target, start_index, end_index)

                    if len(segments_string):
                        segments_string = segments_string[:-1]

                        video = "{} {} {}".format(identity, frames, segments_string)
                        tf_datum = video
                        tf_data.append(tf_datum)

                    print("VIDEO {}: {:05d}/{:05d} Done".format(identity, v_i + 1, len(videos)))

                    with open(json_file_path, "w") as fp:
                        json.dump(tf_data, fp, indent=4, sort_keys=True)

            self.frame_lengths = dict()
            for datum in tf_data:
                splits = datum.split()
                identity = splits[0]
                frame_length = int(splits[1])
                self.frame_lengths[identity] = frame_length

            self.tf_data = tf_data
            self.data_count = len(self.tf_data)
            print("Making Tensorflow Validation Dataset Object ... {} Instances".format(len(tf_data)))
            batch_size = self.datasets.config.batch_size * self.datasets.config.num_gpus
            self.dataloader = DataLoader(self, batch_size=batch_size, shuffle=True,
                                         num_workers=self.datasets.config.num_workers, drop_last=False,
                                         pin_memory=False, prefetch_factor=self.datasets.config.prefetch_factor)

        def __len__(self):
            return len(self.tf_data)

        def __getitem__(self, index):
            splits = self.tf_data[index].split(" ")

            identity = splits[0]
            frame_length = int(splits[1])
            segment_strings = splits[2:]

            background_segments = list()
            foreground_segments = list()
            start_boundary_segments = list()
            end_boundary_segments = list()

            previous_end_index = 0
            for segment_index in range(len(segment_strings) // 3):
                target = int(segment_strings[segment_index * 3])
                start_index = int(segment_strings[segment_index * 3 + 1])
                end_index = int(segment_strings[segment_index * 3 + 2])

                # boundary segments
                # note that we add boundary segments as the possible starting frame indices.
                # we can sample a training sequence by beginning
                # from one of the frame indices specified in an interval

                # start boundary segment
                start_boundary_start_index = \
                    max(start_index - self.datasets.config.temporal_width + 1, 1)

                start_boundary_end_index = start_index - 1

                if start_boundary_end_index - start_boundary_start_index >= 0:
                    start_boundary_segments.append([start_boundary_start_index, start_boundary_end_index,
                                                    ["start", start_index, target]])

                # end boundary segment
                end_boundary_start_index = \
                    max(max(end_index - self.datasets.config.temporal_width + 2, 1),
                        start_index)

                end_boundary_end_index = end_index

                if end_boundary_end_index - end_boundary_start_index >= 0:
                    end_boundary_segments.append([end_boundary_start_index, end_boundary_end_index,
                                                  ["end", end_index, target]])

                # background segment
                background_start_index = previous_end_index + 1
                background_end_index = start_index - 1

                if background_end_index - background_start_index + 1 >= 1:
                    background_segments.append([background_start_index, background_end_index])

                # foreground segment
                if end_index - start_index + 1 >= 1:
                    foreground_segments.append([start_index, end_index, target])

                # if the foreground segment is the last one, add the final background segment
                if segment_index == len(segment_strings) / 3 - 1:
                    background_start_index = min(end_index + 1, frame_length)
                    background_end_index = frame_length

                    if background_end_index - background_start_index + 1 >= 1:
                        background_segments.append([background_start_index, background_end_index])

                previous_end_index = end_index

            entire_targets = np.zeros(dtype=np.int64, shape=(frame_length))
            foreground_segments.sort()
            shuffled_segments = list(foreground_segments)
            random.shuffle(shuffled_segments)
            for segment in shuffled_segments:
                start_index, end_index, target = segment
                entire_targets[start_index - 1:end_index - 1 + 1] = target

            kinetics_features = np.load(os.path.join(self.datasets.features_folder,
                                                     identity, "{}_features.npy".format(identity)))

            feature_length = len(kinetics_features)

            target_length = feature_length if self.datasets.config.dataset == "thumos14" \
                else self.datasets.config.feature_width

            scaled_segments = np.asarray(shuffled_segments, dtype=np.float32)
            scaled_segments[:, :2] = (scaled_segments[:, :2] - 1.0) / (frame_length - 1) * (target_length - 1)
            scaled_segments = scaled_segments.round().astype(np.int32)
            action_targets = np.zeros(dtype=np.float32, shape=(target_length, self.datasets.number_of_classes))
            action_targets[..., 0] = 1.0
            for segment in scaled_segments:
                start_index, end_index, target = segment
                action_targets[start_index:end_index + 1, 0] = 0.0
                action_targets[start_index:end_index + 1, target] = 1.0

            features = kinetics_features

            if self.datasets.config.dataset == "activitynet":
                feature_length = len(features)
                features = interp1d(np.arange(feature_length), features, axis=0, kind="linear")
                features = features(
                    np.linspace(0, feature_length - 1,
                                self.datasets.config.feature_width, dtype=np.float32))
                # action_targets = interp1d(np.arange(feature_length), action_targets, axis=0, kind="linear")
                # action_targets = action_targets(
                #     np.linspace(0, feature_length - 1,
                #                 self.datasets.config.feature_width,
                #                 dtype=np.float32))
                # action_targets /= np.sum(action_targets, axis=-1, keepdims=True)

            features = np.array(features, dtype=np.float32)

            target_slices = list()
            this_targets = action_targets
            this_target_sequence = np.argmax(this_targets, axis=-1)
            run = {"class_number": this_target_sequence[0], "indices": []}
            slices, expect, index = [run], None, 0
            for index, target in enumerate(this_target_sequence):
                if (target == expect) or (expect is None):
                    run["indices"].append(index)
                else:
                    run = {"class_number": target, "indices": [index]}
                    slices.append(run)
                expect = target

            for slice in slices:
                if slice["class_number"] >= 1:
                    target_slices.append([slice["indices"][0],
                                          slice["indices"][-1],
                                          slice["class_number"]])

            target_slices = np.asarray(target_slices, dtype=np.float32)
            target_slices[:, :2] = target_slices[:, :2] / (target_length - 1)
            target_slices[:, :2] = np.clip(target_slices[:, :2], 0.0, 1.0)

            W, C = action_targets.shape

            detection_targets = np.zeros(dtype=np.float32, shape=(W, 4))
            if len(target_slices):
                foreground_segments = target_slices
                detection_targets[:len(foreground_segments), 0] = 1.0
                if self.datasets.config.use_classification:
                    detection_targets[:len(foreground_segments), 1] = foreground_segments[:, -1] - 1
                else:
                    detection_targets[:len(foreground_segments), 1] = 0
                detection_targets[:len(foreground_segments), 2:] = \
                    np.stack(((foreground_segments[:, 0] + foreground_segments[:, 1]) / 2,
                              (foreground_segments[:, 1] - foreground_segments[:, 0])), axis=-1)

            features = torch.from_numpy(features).permute(1, 0)
            targets = torch.from_numpy(detection_targets)

            return features, targets, identity, frame_length
