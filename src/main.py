import os
import sys
import numpy as np
import glob
import json
import random
import math
import time
import gc
import pandas as pd
import argparse
import torch
from tensorboardX import SummaryWriter
np.seterr(all="ignore")
from shutil import rmtree
from evaluation.eval_detection import ANETdetection
from easydict import EasyDict
from models.DABDETR import build
from datasets import Datasets
from util.nms import batched_nms
from util import segment_ops


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = config.model_name
    now = time.localtime()
    train_date = "{:02d}{:02d}".format(now.tm_mon, now.tm_mday)

    model, criterion = build(config)
    datasets = Datasets(config)

    train_data, validation_data = datasets.getDataset("train")

    save_ckpt_file_folder = os.path.join(config.root_path, "networks", "weights",
                                         "save", "{}_{}_{}".format(model_name, config.dataset, train_date))
    summary_folder = os.path.join(config.root_path, "networks", "summaries",
                                  "{}_{}_{}".format(model_name, config.dataset, train_date))

    if config.postfix is not None:
        save_ckpt_file_folder += "_{}".format(config.postfix)
        summary_folder += "_{}".format(config.postfix)

    train_summary_file_path = os.path.join(summary_folder, "train_summary")
    validation_summary_file_path = os.path.join(summary_folder, "validation_summary")

    if config.dataset == "thumos14":
        boundaries = [1000, 1100]
    else:
        boundaries = [80, 100]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, boundaries, gamma=0.1)

    model = model.cuda()

    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(device_id) for device_id in range(config.num_gpus)])
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

    best_validation = float("-inf")
    previous_best_epoch = None
    if config.dataset == "activitynet":
        with open(os.path.join(datasets.datasets_folder,
                               "ANET2017-CUHK-GTAD", "cuhk_val_simp_share.json"), "r") as fp:
            all_anet2017_cuhk = json.load(fp)

    rmtree(summary_folder, ignore_errors=True)
    rmtree(save_ckpt_file_folder, ignore_errors=True)
    try:
        os.mkdir(save_ckpt_file_folder)
    except OSError:
        pass
    train_summary_writer = SummaryWriter(logdir=train_summary_file_path)
    validation_summary_writer = SummaryWriter(logdir=validation_summary_file_path)
    batch_iteration = 0
    for epoch in range(1, config.epochs + 1, 1):
        model.train()
        epoch_losses = dict()
        epoch_time = 0.0
        epoch_training_time = 0.0
        epoch_batch_iteration = 0
        epoch_preprocessing_time = 0.0

        if (epoch) % config.validation_term == 0 or epoch == 1:
            model.eval()
            if config.dataset == "thumos14":
                validation_batch_index = 0
                with open(datasets.target_path, "r") as json_fp:
                    first_ground_truth_json = json.loads(json_fp.read())
                ground_truth_json = dict(first_ground_truth_json)
                ground_truth_json["database"] = dict()
                detection_prediction_json = dict({"version": "VERSION 1.3", "results": {},
                                                  "external_data":
                                                      {"used": True,
                                                       "details": "CNN is pre-trained on Kinetics-400"}})

                identities = sorted([datum.split()[0] for datum in validation_data.tf_data])
                for video_idx, identity in enumerate(identities):
                    feature_path = os.path.join(datasets.features_folder,
                                                identity, "{}_features.npy".format(identity))
                    features = np.load(feature_path)
                    feature_length = len(features)

                    feature_width = config.feature_width
                    testing_step = config.testing_step

                    all_action_predictions = list()
                    all_proposal_predictions = list()
                    all_start_predictions = list()
                    all_end_predictions = list()
                    all_confidence_predictions = list()
                    all_classification_predictions = list()
                    loop_index = 0
                    for start_idx in range(0, len(features), testing_step):
                        this_features = features[start_idx:start_idx + feature_width]

                        if len(this_features) < feature_width:
                            this_features = \
                                np.concatenate([this_features,
                                                np.tile(np.expand_dims(np.zeros_like(this_features[0]), axis=0),
                                                        (config.feature_width - len(this_features), 1))],
                                               axis=0)

                        fetches = [model_validation.action_predictions, model_validation.proposal_predictions]

                        if config.use_boundary:
                            fetches.append(model_validation.start_predictions)
                            fetches.append(model_validation.end_predictions)
                        if config.use_confidence:
                            fetches.append(model_validation.confidence_predictions)
                        if config.use_classification:
                            fetches.append(model_validation.classification_predictions)

                        results = session.run(fetches, feed_dict={model_validation.features: [this_features]})

                        action_predictions = results[0]
                        proposal_predictions = results[1]
                        all_action_predictions.append(np.squeeze(action_predictions, axis=0))
                        all_proposal_predictions.append(np.squeeze(proposal_predictions, axis=0))
                        r_i = 2
                        if config.use_boundary:
                            start_predictions = results[r_i]
                            r_i += 1
                            end_predictions = results[r_i]
                            r_i += 1
                            all_start_predictions.append(np.squeeze(start_predictions, axis=0))
                            all_end_predictions.append(np.squeeze(end_predictions, axis=0))
                        if config.use_confidence:
                            confidence_predictions = results[r_i]
                            r_i += 1
                            all_confidence_predictions.append(np.squeeze(confidence_predictions, axis=0))
                        if config.use_classification:
                            classification_predictions = results[r_i]
                            all_classification_predictions.append(np.squeeze(classification_predictions, axis=0))

                        if (config.use_boundary) and (loop_index >= 1 and testing_step < feature_width):
                            avg_start_predictions = \
                                (all_start_predictions[loop_index - 1][:, testing_step:] +
                                 all_start_predictions[loop_index][:, :testing_step]) / 2.0
                            all_start_predictions[loop_index - 1][:, testing_step:] = avg_start_predictions
                            all_start_predictions[loop_index][:, :testing_step] = avg_start_predictions
                            avg_end_predictions = \
                                (all_end_predictions[loop_index - 1][:, testing_step:] +
                                 all_end_predictions[loop_index][:, :testing_step]) / 2.0
                            all_end_predictions[loop_index - 1][:, testing_step:] = avg_end_predictions
                            all_end_predictions[loop_index][:, :testing_step] = avg_end_predictions
                        loop_index += 1

                    '''
                    Localization
                    '''
                    frame_length = len(glob.glob(os.path.join(datasets.frames_folder, identity, "images", "*")))

                    class_indices = list()
                    start_indices = list()
                    end_indices = list()
                    scores = list()
                    proposals = list()
                    num_loops = len(all_action_predictions)
                    if config.use_classification:
                        video_level_classification = np.mean(all_classification_predictions, axis=(0, 1, 2))
                    for loop_index in range(num_loops):
                        num_levels = len(all_action_predictions[loop_index])
                        if config.use_classification:
                            classification_predictions = all_classification_predictions[loop_index]
                        for scale_index in range(num_levels):
                            actionness = all_action_predictions[loop_index][scale_index]
                            proposal_predictions = all_proposal_predictions[loop_index][scale_index]
                            if config.use_boundary:
                                startness = all_start_predictions[loop_index][scale_index]
                                endness = all_end_predictions[loop_index][scale_index]
                            if config.use_confidence:
                                confidence_predictions = all_confidence_predictions[loop_index][scale_index]

                            W = H = actionness.shape[0]

                            this_start_indices = list()
                            this_end_indices = list()
                            proposal_scores = list()

                            padded_actionness = np.pad(actionness, (1, 1), "constant")
                            p_a = \
                                np.where((actionness >= padded_actionness[:-2]) *
                                         (actionness >= padded_actionness[2:]) +
                                         actionness >= 0.5 *
                                         np.max(np.array(all_action_predictions)[:, scale_index]))[0]

                            p_s = proposal_predictions[..., 0]
                            p_e = proposal_predictions[..., 1]

                            if config.use_relative_regression:
                                p_s = np.round(p_a - p_s[p_a] * float(H - 1)).astype(np.int64)
                                p_e = np.round(p_a + p_e[p_a] * float(W - 1)).astype(np.int64)
                            else:
                                p_s = np.round(p_s[p_a] * float(H - 1)).astype(np.int64)
                                p_e = np.round(p_e[p_a] * float(W - 1)).astype(np.int64)

                            p_s = np.clip(p_s, 0, H - 1)
                            p_e = np.clip(p_e, 0, W - 1)
                            this_proposal_scores = actionness[p_a]
                            if config.use_boundary:
                                this_proposal_scores *= startness[p_s] * endness[p_e]
                            this_start_indices.append(p_s)
                            this_end_indices.append(p_e)
                            proposal_scores.append(this_proposal_scores)

                            this_start_indices = np.concatenate(this_start_indices, axis=0)
                            this_end_indices = np.concatenate(this_end_indices, axis=0)
                            proposal_scores = np.concatenate(proposal_scores, axis=0)

                            valid_flags = this_end_indices >= this_start_indices
                            this_start_indices = this_start_indices[valid_flags]
                            this_end_indices = this_end_indices[valid_flags]
                            proposal_scores = proposal_scores[valid_flags]

                            if config.use_confidence:
                                confidence_scores = confidence_predictions[this_start_indices, this_end_indices]
                            if config.use_classification:
                                proposal_level_classification = classification_predictions[
                                    this_start_indices, this_end_indices]

                            if config.use_classification:
                                classification = (proposal_level_classification +
                                                  video_level_classification) / 2.0
                            else:
                                pass
                            this_class_indices = np.argmax(classification, axis=-1)
                            classification_scores = classification[
                                np.arange(len(this_class_indices)), this_class_indices]
                            this_class_indices += 1

                            this_start_indices += loop_index * testing_step
                            this_start_indices = \
                                np.round((this_start_indices) / float(feature_length - 1) * frame_length)
                            this_start_indices = np.maximum(np.minimum(this_start_indices, frame_length), 1)
                            this_end_indices += loop_index * testing_step
                            this_end_indices = \
                                np.round((this_end_indices) / float(feature_length - 1) * frame_length)
                            this_end_indices = np.maximum(np.minimum(this_end_indices, frame_length),
                                                          this_start_indices)

                            this_scores = proposal_scores
                            if config.use_confidence:
                                this_scores *= confidence_scores
                            if config.use_classification:
                                this_scores *= classification_scores

                            valid_flags = this_end_indices - this_start_indices + 1 >= config.feature_frame_step_size

                            this_class_indices = this_class_indices[valid_flags]
                            this_start_indices = this_start_indices[valid_flags]
                            this_end_indices = this_end_indices[valid_flags]
                            this_scores = this_scores[valid_flags]

                            class_indices.append(this_class_indices)
                            start_indices.append(this_start_indices)
                            end_indices.append(this_end_indices)
                            scores.append(this_scores)
                            proposals.append(np.stack([this_class_indices, this_start_indices,
                                                       this_end_indices, this_scores], axis=-1))

                    class_indices = np.concatenate(class_indices, axis=0)
                    start_indices = np.concatenate(start_indices, axis=0)
                    end_indices = np.concatenate(end_indices, axis=0)
                    scores = np.concatenate(scores, axis=0)

                    video_prediction_slices = \
                        pd.DataFrame(data={"class_index": class_indices,
                                           "start_index": start_indices,
                                           "end_index": end_indices,
                                           "score": scores})

                    video_prediction_slices = video_prediction_slices.groupby("class_index")

                    nmsed_detection_slices = list()
                    for class_index, slices in video_prediction_slices:
                        slices = slices.values
                        slices = nms(slices, threshold=config.nms_threshold)
                        nmsed_detection_slices += slices.tolist()

                    nmsed_detection_slices.sort(reverse=True, key=lambda x: x[-1])
                    nmsed_detection_slices = nmsed_detection_slices[:200]

                    detection_prediction_json["results"][identity] = list()
                    for prediction_slice in nmsed_detection_slices:
                        score = prediction_slice[-1]
                        prediction_class = int(prediction_slice[0])
                        label = datasets.label_dic[str(prediction_class)].replace("_", " ")
                        frame_intervals = [prediction_slice[1], prediction_slice[2]]
                        time_intervals = [float(frame_intervals[0]) / config.video_fps,
                                          float(frame_intervals[1]) / config.dataset.video_fps]
                        # time_intervals = [
                        #     float(frame_intervals[0]) / frame_length * dataset.meta_dic["database"][identity][
                        #         "duration"],
                        #     float(frame_intervals[1]) / frame_length * dataset.meta_dic["database"][identity][
                        #         "duration"]]

                        detection_prediction_json["results"][identity].append(
                            {"label": label, "score": score, "segment": time_intervals})

                        if config.dataset == "thumos14" and label == "CliffDiving":
                            detection_prediction_json["results"][identity].append(
                                {"label": "Diving", "score": score, "segment": time_intervals})

                    for annotation in first_ground_truth_json["database"][identity]["annotations"]:
                        if identity in ground_truth_json["database"]:
                            ground_truth_json["database"][identity]["annotations"].append(annotation)
                        else:
                            ground_truth_json["database"][identity] = dict(
                                first_ground_truth_json["database"][identity])
                            ground_truth_json["database"][identity]["annotations"] = [annotation]

                    gc.collect()
                    print_string = \
                        "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                            "Validation", epoch, config.epochs, validation_batch_index + 1, len(identities), 0.0)
                    progress_step = validation_batch_index + 1
                    progress_length = len(identities)
                    print_string += \
                        " |{}{}|".format(
                            "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                            " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                    validation_batch_index += 1

                print()

                try:
                    anet_detection = ANETdetection(ground_truth_json,
                                                   detection_prediction_json,
                                                   subset="validation", check_status=False,
                                                   tiou_thresholds=np.arange(0.3, 0.8, 0.1))

                    interpolated_mAP = anet_detection.evaluate()
                    overall_interpolated_mAP = interpolated_mAP.mean()
                    validation_mAP = overall_interpolated_mAP
                except:
                    validation_mAP = 0.0

                validation_summary_feed_dict = dict()
                validation_summary_feed_dict[mAP_summary_ph] = validation_mAP

                validation_summary = \
                    session.run(validation_summaries, feed_dict=validation_summary_feed_dict)
                validation_summary_writer.add_summary(validation_summary, epoch)

                validation_quality = validation_mAP

                if epoch % config.ckpt_save_term == 0:
                    if previous_best_epoch and previous_best_epoch != epoch - config.ckpt_save_term:
                        weight_files = glob.glob(
                            os.path.join(save_ckpt_file_folder,
                                         "weights.ckpt-{}.*".format(epoch - config.ckpt_save_term)))
                        for file in weight_files:
                            try:
                                os.remove(file)
                            except OSError:
                                pass

                    saver.save(session, os.path.join(save_ckpt_file_folder, "weights.ckpt"), global_step=epoch)

                if validation_quality >= best_validation:
                    best_validation = validation_quality
                    if previous_best_epoch:
                        weight_files = glob.glob(os.path.join(save_ckpt_file_folder,
                                                              "weights.ckpt-{}.*".format(
                                                                  previous_best_epoch)))
                        for file in weight_files:
                            try:
                                os.remove(file)
                            except OSError:
                                pass

                    if epoch % config.ckpt_save_term != 0:
                        saver.save(session, os.path.join(save_ckpt_file_folder, "weights.ckpt"),
                                   global_step=epoch)
                    previous_best_epoch = epoch

                print("Validation Results ...")
                print("Validation Localization mAP {:.5f}".format(validation_mAP))
                print("=" * 90)
            else:
                valid_num = 0.0
                validation_mAP = 0.0
                validation_batch_index = 0
                validation_losses = dict()

                loop_rounds = \
                    int(math.ceil(float(validation_data.data_count) / float(config.batch_size * config.num_gpus)))

                with open(datasets.target_path, "r") as json_fp:
                    ground_truth_json = json.loads(json_fp.read())
                detection_prediction_json = dict({"version": "VERSION 1.3", "results": {},
                                                      "external_data":
                                                          {"used": True,
                                                           "details": "CNN is pre-trained on Kinetics-400"}})

                for features, targets, identities, frame_lengths in validation_data.dataloader:
                    features = features.cuda()
                    target_dict = list()
                    for b_i in range(len(targets)):
                        batch_dict = dict()
                        batch_dict["labels"] = list()
                        batch_dict["segments"] = list()
                        for t_i, t in enumerate(targets[b_i]):
                            if t[0] <= 0.0:
                                break
                            batch_dict["labels"].append(t[1])
                            batch_dict["segments"].append(t[2:])
                        if len(batch_dict["labels"]):
                            batch_dict["labels"] = torch.stack(batch_dict["labels"], dim=0).long().cuda()
                            batch_dict["segments"] = torch.stack(batch_dict["segments"], dim=0).float().cuda()
                        target_dict.append(batch_dict)

                    predictions = model(features)
                    pred_logits = predictions["pred_logits"].sigmoid().detach().cpu().numpy()
                    pred_segments = segment_ops.segment_cw_to_t1t2(predictions["pred_segments"])
                    pred_segments = pred_segments.detach().cpu().numpy()
                    loss_dict = criterion(predictions, target_dict)

                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                    loss = losses.item()
                    for loss_name, loss_value in loss_dict.items():
                        if loss_name not in validation_losses.keys():
                            validation_losses[loss_name] = loss_value.item()
                        else:
                            validation_losses[loss_name] += loss_value.item()
                    valid_num += len(identities)

                    frame_lengths = frame_lengths.numpy()
                    for n_i in range(len(identities)):
                        identity = identities[n_i]
                        frame_length = frame_lengths[n_i]
                        cuhk_classification_scores = np.array(all_anet2017_cuhk["results"][identity])

                        this_pred_logits = pred_logits[n_i]
                        this_pred_segments = pred_segments[n_i]

                        p_s = this_pred_segments[..., 0]
                        p_e = this_pred_segments[..., 1]
                        scores = np.max(this_pred_logits, axis=-1)

                        valid_flags = p_e >= p_s
                        p_s = p_s[valid_flags]
                        p_e = p_e[valid_flags]
                        scores = scores[valid_flags]

                        classification = this_pred_logits + np.expand_dims(cuhk_classification_scores, axis=0)
                        class_indices = np.argmax(classification, axis=-1)

                        start_indices = np.round(p_s * frame_length)
                        start_indices = np.clip(start_indices, 1, frame_length)
                        end_indices = np.round(p_e * frame_length)
                        end_indices = np.clip(end_indices, 1, frame_length)

                        valid_flags = end_indices - start_indices + 1 >= config.feature_frame_step_size

                        class_indices = class_indices[valid_flags]
                        start_indices = start_indices[valid_flags]
                        end_indices = end_indices[valid_flags]
                        scores = scores[valid_flags]

                        video_prediction_slices = \
                            pd.DataFrame(data={"class_index": class_indices,
                                               "start_index": start_indices,
                                               "end_index": end_indices,
                                               "score": scores})

                        if not config.use_soft_nms:
                            if config.use_classification:
                                nmsed_detection_slices = list()
                                video_prediction_slices = video_prediction_slices.groupby("class_index")
                                for class_index, slices in video_prediction_slices:
                                    slices = slices.values
                                    slices = nms(slices, threshold=config.nms_threshold)
                                    nmsed_detection_slices += slices.tolist()
                            else:
                                slices = video_prediction_slices.values
                                slices = nms(slices, threshold=config.nms_threshold)
                                nmsed_detection_slices = slices.tolist()

                            nmsed_detection_slices.sort(reverse=True, key=lambda x: x[-1])
                            nmsed_detection_slices = nmsed_detection_slices[:100]
                        else:
                            scores = torch.from_numpy(scores).float()
                            labels = torch.from_numpy(class_indices)
                            boxes = torch.from_numpy(np.stack((start_indices, end_indices), axis=-1)).float()
                            boxes, scores, labels = batched_nms(
                                boxes.contiguous(), scores.contiguous(), labels.contiguous(),
                                config.iou_threshold,
                                config.min_score,
                                config.max_seg_num,
                                use_soft_nms=True,
                                multiclass=config.multiclass_nms,
                                sigma=config.nms_sigma,
                                voting_thresh=config.voting_thresh)
                            boxes = torch.where(boxes.isnan(), torch.zeros_like(boxes), boxes).numpy()
                            labels = torch.where(labels.isnan(), torch.zeros_like(labels), labels).numpy()
                            scores = torch.where(scores.isnan(), torch.zeros_like(scores), scores).numpy()
                            nmsed_detection_slices = np.concatenate((labels[..., None], boxes, scores[..., None]),
                                                                    axis=-1)

                        detection_prediction_json["results"][identity] = list()
                        for prediction_slice in nmsed_detection_slices:
                            score = prediction_slice[-1]
                            prediction_class = int(prediction_slice[0])
                            label = datasets.label_dic[str(prediction_class)].replace("_", " ")
                            frame_intervals = [prediction_slice[1], prediction_slice[2]]
                            # time_intervals = [float(frame_intervals[0]) / config.video_fps,
                            #                   float(frame_intervals[1]) / config.video_fps]
                            time_intervals = [
                                float(frame_intervals[0] - 1) / (frame_length - 1) *
                                datasets.meta_dic["database"][identity]["duration"],
                                float(frame_intervals[1] - 1) / (frame_length - 1) *
                                datasets.meta_dic["database"][identity]["duration"]]

                            detection_prediction_json["results"][identity].append(
                                {"label": label, "score": score, "segment": time_intervals})

                            if config.dataset == "thumos14" and label == "CliffDiving":
                                detection_prediction_json["results"][identity].append(
                                    {"label": "Diving", "score": score, "segment": time_intervals})

                    print_string = \
                        "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                            "Validation", epoch, config.epochs, validation_batch_index + 1, loop_rounds, loss)
                    progress_step = validation_batch_index + 1
                    progress_length = loop_rounds
                    print_string += \
                        " |{}{}|".format(
                            "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                            " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                    validation_batch_index += 1

                print()

                if config.dataset == "activitynet":
                    try:
                        anet_detection = ANETdetection(ground_truth_json,
                                                       detection_prediction_json,
                                                       subset="validation", check_status=True,
                                                       tiou_thresholds=np.linspace(0.5, 0.95, 10))

                        interpolated_mAP = anet_detection.evaluate()
                        overall_interpolated_mAP = interpolated_mAP.mean()
                        validation_mAP = overall_interpolated_mAP
                    except:
                        validation_mAP = 0.0

                for loss_name, loss_value in validation_losses.items():
                    validation_summary_writer.add_scalar(loss_name, loss_value / float(validation_batch_index),
                                                         epoch)
                validation_summary_writer.add_scalar("mAP", validation_mAP, epoch)

                validation_quality = validation_mAP

                if validation_quality >= best_validation:
                    best_validation = validation_quality
                    if previous_best_epoch and previous_best_epoch != epoch - config.ckpt_save_term:
                        weight_files = glob.glob(os.path.join(save_ckpt_file_folder,
                                                              "weights-{}.pt".format(previous_best_epoch)))
                        for file in weight_files:
                            try:
                                os.remove(file)
                            except OSError:
                                pass

                    if epoch % config.ckpt_save_term != 0:
                        torch.save(model.state_dict(),
                                   os.path.join(save_ckpt_file_folder, "weights-{}.pt".format(epoch)))
                    previous_best_epoch = epoch

                print("Validation Results ...")
                print("Validation Localization mAP {:.5f}".format(validation_mAP))
                print("=" * 90)

        batch_length = \
            int(math.floor(float(train_data.data_count) / float(config.batch_size * config.num_gpus)))
        for features, targets, identities, frame_lengths in train_data.dataloader:
            iteration_start_time = time.time()
            preprocessing_start_time = time.time()

            features = features.cuda()
            target_dict = list()
            for b_i in range(len(targets)):
                batch_dict = dict()
                batch_dict["labels"] = list()
                batch_dict["segments"] = list()
                for t_i, t in enumerate(targets[b_i]):
                    if t[0] <= 0.0:
                        break
                    batch_dict["labels"].append(t[1])
                    batch_dict["segments"].append(t[2:])
                if len(batch_dict["labels"]):
                    batch_dict["labels"] = torch.stack(batch_dict["labels"], dim=0).long().cuda()
                    batch_dict["segments"] = torch.stack(batch_dict["segments"], dim=0).float().cuda()
                target_dict.append(batch_dict)

            epoch_preprocessing_time += time.time() - preprocessing_start_time
            train_step_start_time = time.time()
            outputs = model(features)
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()
            epoch_training_time += time.time() - train_step_start_time

            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_losses.keys():
                    epoch_losses[loss_name] = loss_value.item()
                else:
                    epoch_losses[loss_name] += loss_value.item()

            print_string = \
                "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                    "Training",
                    epoch,
                    config.epochs,
                    epoch_batch_iteration + 1,
                    batch_length,
                    losses.item())
            progress_step = epoch_batch_iteration + 1
            progress_length = batch_length
            print_string += \
                " |{}{}|".format(
                    "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                    " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
            sys.stdout.write("\r" + print_string)
            sys.stdout.flush()

            epoch_batch_iteration += 1
            batch_iteration += 1

            epoch_time += time.time() - iteration_start_time

        scheduler.step(epoch)
        print()
        epoch_training_time /= float(epoch_batch_iteration)
        epoch_preprocessing_time /= float(epoch_batch_iteration)
        for loss_name, loss_value in epoch_losses.items():
            train_summary_writer.add_scalar(loss_name, loss_value / float(epoch_batch_iteration), epoch)

        if epoch % config.ckpt_save_term == 0:
            torch.save(model.state_dict(), os.path.join(save_ckpt_file_folder, "weights-{}.pt".format(epoch)))

        print("=" * 90)
        print("Epoch {:05d} Done ... Current Batch Iterations {:07d}".format(epoch, batch_iteration))
        print("Epoch {:05d} Takes {:03d} Batch Iterations".format(epoch, epoch_batch_iteration))
        print("Epoch {:05d} Takes {:.2f} Hours".format(epoch, epoch_time / 3600.0))
        print("Epoch {:05d} Average One Loop Time {:.2f} Seconds".format(epoch,
                                                                         epoch_time / float(epoch_batch_iteration)))
        print("Epoch {:05d} Average One Preprocessing Time {:.2f} Seconds".format(epoch, epoch_preprocessing_time))
        print("Epoch {:05d} Average One Train Step Time {:.2f} Seconds".format(epoch, epoch_training_time))
        print("=" * 90)

        if (epoch) % config.validation_term == 0 or epoch == 1:
            model.eval()
            if config.dataset == "thumos14":
                validation_batch_index = 0
                with open(datasets.target_path, "r") as json_fp:
                    first_ground_truth_json = json.loads(json_fp.read())
                ground_truth_json = dict(first_ground_truth_json)
                ground_truth_json["database"] = dict()
                detection_prediction_json = dict({"version": "VERSION 1.3", "results": {},
                                                  "external_data":
                                                      {"used": True,
                                                       "details": "CNN is pre-trained on Kinetics-400"}})

                identities = sorted([datum.split()[0] for datum in validation_data.tf_data])
                for video_idx, identity in enumerate(identities):
                    feature_path = os.path.join(datasets.features_folder,
                                                identity, "{}_features.npy".format(identity))
                    features = np.load(feature_path)
                    feature_length = len(features)

                    feature_width = config.feature_width
                    testing_step = config.testing_step

                    all_action_predictions = list()
                    all_proposal_predictions = list()
                    all_start_predictions = list()
                    all_end_predictions = list()
                    all_confidence_predictions = list()
                    all_classification_predictions = list()
                    loop_index = 0
                    for start_idx in range(0, len(features), testing_step):
                        this_features = features[start_idx:start_idx + feature_width]

                        if len(this_features) < feature_width:
                            this_features = \
                                np.concatenate([this_features,
                                                np.tile(np.expand_dims(np.zeros_like(this_features[0]), axis=0),
                                                        (config.feature_width - len(this_features), 1))],
                                               axis=0)

                        fetches = [model_validation.action_predictions, model_validation.proposal_predictions]

                        if config.use_boundary:
                            fetches.append(model_validation.start_predictions)
                            fetches.append(model_validation.end_predictions)
                        if config.use_confidence:
                            fetches.append(model_validation.confidence_predictions)
                        if config.use_classification:
                            fetches.append(model_validation.classification_predictions)

                        results = session.run(fetches, feed_dict={model_validation.features: [this_features]})

                        action_predictions = results[0]
                        proposal_predictions = results[1]
                        all_action_predictions.append(np.squeeze(action_predictions, axis=0))
                        all_proposal_predictions.append(np.squeeze(proposal_predictions, axis=0))
                        r_i = 2
                        if config.use_boundary:
                            start_predictions = results[r_i]
                            r_i += 1
                            end_predictions = results[r_i]
                            r_i += 1
                            all_start_predictions.append(np.squeeze(start_predictions, axis=0))
                            all_end_predictions.append(np.squeeze(end_predictions, axis=0))
                        if config.use_confidence:
                            confidence_predictions = results[r_i]
                            r_i += 1
                            all_confidence_predictions.append(np.squeeze(confidence_predictions, axis=0))
                        if config.use_classification:
                            classification_predictions = results[r_i]
                            all_classification_predictions.append(np.squeeze(classification_predictions, axis=0))

                        if (config.use_boundary) and (loop_index >= 1 and testing_step < feature_width):
                            avg_start_predictions = \
                                (all_start_predictions[loop_index - 1][:, testing_step:] +
                                 all_start_predictions[loop_index][:, :testing_step]) / 2.0
                            all_start_predictions[loop_index - 1][:, testing_step:] = avg_start_predictions
                            all_start_predictions[loop_index][:, :testing_step] = avg_start_predictions
                            avg_end_predictions = \
                                (all_end_predictions[loop_index - 1][:, testing_step:] +
                                 all_end_predictions[loop_index][:, :testing_step]) / 2.0
                            all_end_predictions[loop_index - 1][:, testing_step:] = avg_end_predictions
                            all_end_predictions[loop_index][:, :testing_step] = avg_end_predictions
                        loop_index += 1

                    '''
                    Localization
                    '''
                    frame_length = len(glob.glob(os.path.join(datasets.frames_folder, identity, "images", "*")))

                    class_indices = list()
                    start_indices = list()
                    end_indices = list()
                    scores = list()
                    proposals = list()
                    num_loops = len(all_action_predictions)
                    if config.use_classification:
                        video_level_classification = np.mean(all_classification_predictions, axis=(0, 1, 2))
                    for loop_index in range(num_loops):
                        num_levels = len(all_action_predictions[loop_index])
                        if config.use_classification:
                            classification_predictions = all_classification_predictions[loop_index]
                        for scale_index in range(num_levels):
                            actionness = all_action_predictions[loop_index][scale_index]
                            proposal_predictions = all_proposal_predictions[loop_index][scale_index]
                            if config.use_boundary:
                                startness = all_start_predictions[loop_index][scale_index]
                                endness = all_end_predictions[loop_index][scale_index]
                            if config.use_confidence:
                                confidence_predictions = all_confidence_predictions[loop_index][scale_index]

                            W = H = actionness.shape[0]

                            this_start_indices = list()
                            this_end_indices = list()
                            proposal_scores = list()

                            padded_actionness = np.pad(actionness, (1, 1), "constant")
                            p_a = \
                                np.where((actionness >= padded_actionness[:-2]) *
                                         (actionness >= padded_actionness[2:]) +
                                         actionness >= 0.5 *
                                         np.max(np.array(all_action_predictions)[:, scale_index]))[0]

                            p_s = proposal_predictions[..., 0]
                            p_e = proposal_predictions[..., 1]

                            if config.use_relative_regression:
                                p_s = np.round(p_a - p_s[p_a] * float(H - 1)).astype(np.int64)
                                p_e = np.round(p_a + p_e[p_a] * float(W - 1)).astype(np.int64)
                            else:
                                p_s = np.round(p_s[p_a] * float(H - 1)).astype(np.int64)
                                p_e = np.round(p_e[p_a] * float(W - 1)).astype(np.int64)

                            p_s = np.clip(p_s, 0, H - 1)
                            p_e = np.clip(p_e, 0, W - 1)
                            this_proposal_scores = actionness[p_a]
                            if config.use_boundary:
                                this_proposal_scores *= startness[p_s] * endness[p_e]
                            this_start_indices.append(p_s)
                            this_end_indices.append(p_e)
                            proposal_scores.append(this_proposal_scores)

                            this_start_indices = np.concatenate(this_start_indices, axis=0)
                            this_end_indices = np.concatenate(this_end_indices, axis=0)
                            proposal_scores = np.concatenate(proposal_scores, axis=0)

                            valid_flags = this_end_indices >= this_start_indices
                            this_start_indices = this_start_indices[valid_flags]
                            this_end_indices = this_end_indices[valid_flags]
                            proposal_scores = proposal_scores[valid_flags]

                            if config.use_confidence:
                                confidence_scores = confidence_predictions[this_start_indices, this_end_indices]
                            if config.use_classification:
                                proposal_level_classification = classification_predictions[
                                    this_start_indices, this_end_indices]

                            if config.use_classification:
                                classification = (proposal_level_classification +
                                                  video_level_classification) / 2.0
                            else:
                                pass
                            this_class_indices = np.argmax(classification, axis=-1)
                            classification_scores = classification[
                                np.arange(len(this_class_indices)), this_class_indices]
                            this_class_indices += 1

                            this_start_indices += loop_index * testing_step
                            this_start_indices = \
                                np.round((this_start_indices) / float(feature_length - 1) * frame_length)
                            this_start_indices = np.maximum(np.minimum(this_start_indices, frame_length), 1)
                            this_end_indices += loop_index * testing_step
                            this_end_indices = \
                                np.round((this_end_indices) / float(feature_length - 1) * frame_length)
                            this_end_indices = np.maximum(np.minimum(this_end_indices, frame_length),
                                                          this_start_indices)

                            this_scores = proposal_scores
                            if config.use_confidence:
                                this_scores *= confidence_scores
                            if config.use_classification:
                                this_scores *= classification_scores

                            valid_flags = this_end_indices - this_start_indices + 1 >= config.feature_frame_step_size

                            this_class_indices = this_class_indices[valid_flags]
                            this_start_indices = this_start_indices[valid_flags]
                            this_end_indices = this_end_indices[valid_flags]
                            this_scores = this_scores[valid_flags]

                            class_indices.append(this_class_indices)
                            start_indices.append(this_start_indices)
                            end_indices.append(this_end_indices)
                            scores.append(this_scores)
                            proposals.append(np.stack([this_class_indices, this_start_indices,
                                                       this_end_indices, this_scores], axis=-1))

                    class_indices = np.concatenate(class_indices, axis=0)
                    start_indices = np.concatenate(start_indices, axis=0)
                    end_indices = np.concatenate(end_indices, axis=0)
                    scores = np.concatenate(scores, axis=0)

                    video_prediction_slices = \
                        pd.DataFrame(data={"class_index": class_indices,
                                           "start_index": start_indices,
                                           "end_index": end_indices,
                                           "score": scores})

                    video_prediction_slices = video_prediction_slices.groupby("class_index")

                    nmsed_detection_slices = list()
                    for class_index, slices in video_prediction_slices:
                        slices = slices.values
                        slices = nms(slices, threshold=config.nms_threshold)
                        nmsed_detection_slices += slices.tolist()

                    nmsed_detection_slices.sort(reverse=True, key=lambda x: x[-1])
                    nmsed_detection_slices = nmsed_detection_slices[:200]

                    detection_prediction_json["results"][identity] = list()
                    for prediction_slice in nmsed_detection_slices:
                        score = prediction_slice[-1]
                        prediction_class = int(prediction_slice[0])
                        label = datasets.label_dic[str(prediction_class)].replace("_", " ")
                        frame_intervals = [prediction_slice[1], prediction_slice[2]]
                        time_intervals = [float(frame_intervals[0]) / config.video_fps,
                                          float(frame_intervals[1]) / config.dataset.video_fps]
                        # time_intervals = [
                        #     float(frame_intervals[0]) / frame_length * dataset.meta_dic["database"][identity][
                        #         "duration"],
                        #     float(frame_intervals[1]) / frame_length * dataset.meta_dic["database"][identity][
                        #         "duration"]]

                        detection_prediction_json["results"][identity].append(
                            {"label": label, "score": score, "segment": time_intervals})

                        if config.dataset == "thumos14" and label == "CliffDiving":
                            detection_prediction_json["results"][identity].append(
                                {"label": "Diving", "score": score, "segment": time_intervals})

                    for annotation in first_ground_truth_json["database"][identity]["annotations"]:
                        if identity in ground_truth_json["database"]:
                            ground_truth_json["database"][identity]["annotations"].append(annotation)
                        else:
                            ground_truth_json["database"][identity] = dict(
                                first_ground_truth_json["database"][identity])
                            ground_truth_json["database"][identity]["annotations"] = [annotation]

                    gc.collect()
                    print_string = \
                        "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                            "Validation", epoch, config.epochs, validation_batch_index + 1, len(identities), 0.0)
                    progress_step = validation_batch_index + 1
                    progress_length = len(identities)
                    print_string += \
                        " |{}{}|".format(
                            "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                            " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                    validation_batch_index += 1

                print()

                try:
                    anet_detection = ANETdetection(ground_truth_json,
                                                   detection_prediction_json,
                                                   subset="validation", check_status=False,
                                                   tiou_thresholds=np.arange(0.3, 0.8, 0.1))

                    interpolated_mAP = anet_detection.evaluate()
                    overall_interpolated_mAP = interpolated_mAP.mean()
                    validation_mAP = overall_interpolated_mAP
                except:
                    validation_mAP = 0.0

                validation_summary_feed_dict = dict()
                validation_summary_feed_dict[mAP_summary_ph] = validation_mAP

                validation_summary = \
                    session.run(validation_summaries, feed_dict=validation_summary_feed_dict)
                validation_summary_writer.add_summary(validation_summary, epoch)

                validation_quality = validation_mAP

                if epoch % config.ckpt_save_term == 0:
                    if previous_best_epoch and previous_best_epoch != epoch - config.ckpt_save_term:
                        weight_files = glob.glob(
                            os.path.join(save_ckpt_file_folder,
                                         "weights.ckpt-{}.*".format(epoch - config.ckpt_save_term)))
                        for file in weight_files:
                            try:
                                os.remove(file)
                            except OSError:
                                pass

                    saver.save(session, os.path.join(save_ckpt_file_folder, "weights.ckpt"), global_step=epoch)

                if validation_quality >= best_validation:
                    best_validation = validation_quality
                    if previous_best_epoch:
                        weight_files = glob.glob(os.path.join(save_ckpt_file_folder,
                                                              "weights.ckpt-{}.*".format(
                                                                  previous_best_epoch)))
                        for file in weight_files:
                            try:
                                os.remove(file)
                            except OSError:
                                pass

                    if epoch % config.ckpt_save_term != 0:
                        saver.save(session, os.path.join(save_ckpt_file_folder, "weights.ckpt"),
                                   global_step=epoch)
                    previous_best_epoch = epoch

                print("Validation Results ...")
                print("Validation Localization mAP {:.5f}".format(validation_mAP))
                print("=" * 90)
            else:
                valid_num = 0.0
                validation_mAP = 0.0
                validation_batch_index = 0
                validation_losses = dict()

                loop_rounds = \
                    int(math.ceil(float(validation_data.data_count) / float(config.batch_size * config.num_gpus)))

                with open(datasets.target_path, "r") as json_fp:
                    ground_truth_json = json.loads(json_fp.read())
                detection_prediction_json = dict({"version": "VERSION 1.3", "results": {},
                                                      "external_data":
                                                          {"used": True,
                                                           "details": "CNN is pre-trained on Kinetics-400"}})

                for features, targets, identities, frame_lengths in validation_data.dataloader:
                    features = features.cuda()
                    target_dict = list()
                    for b_i in range(len(targets)):
                        batch_dict = dict()
                        batch_dict["labels"] = list()
                        batch_dict["segments"] = list()
                        for t_i, t in enumerate(targets[b_i]):
                            if t[0] <= 0.0:
                                break
                            batch_dict["labels"].append(t[1])
                            batch_dict["segments"].append(t[2:])
                        if len(batch_dict["labels"]):
                            batch_dict["labels"] = torch.stack(batch_dict["labels"], dim=0).long().cuda()
                            batch_dict["segments"] = torch.stack(batch_dict["segments"], dim=0).float().cuda()
                        target_dict.append(batch_dict)

                    predictions = model(features)
                    pred_boxes = segment_ops.segment_cw_to_t1t2(predictions["pred_boxes"])
                    pred_boxes = pred_boxes.detach().cpu().numpy()
                    pred_logits = predictions["pred_logits"].sigmoid().detach().cpu().numpy()
                    loss_dict = criterion(predictions, target_dict)

                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                    loss = losses.item()
                    for loss_name, loss_value in loss_dict.items():
                        if loss_name not in validation_losses.keys():
                            validation_losses[loss_name] = loss_value.item()
                        else:
                            validation_losses[loss_name] += loss_value.item()
                    valid_num += len(identities)
                    for n_i in range(len(identities)):
                        identity = identities[n_i]
                        frame_length = frame_lengths[n_i]
                        cuhk_classification_scores = np.array(all_anet2017_cuhk["results"][identity])

                        this_pred_boxes = pred_boxes[n_i]
                        this_pred_logits = pred_logits[n_i]

                        p_s = this_pred_boxes[..., 0]
                        p_e = this_pred_boxes[..., 1]
                        scores = np.max(this_pred_logits, axis=-1)

                        valid_flags = p_e >= p_s
                        p_s = p_s[valid_flags]
                        p_e = p_e[valid_flags]
                        scores = scores[valid_flags]

                        classification = this_pred_logits + np.expand_dims(cuhk_classification_scores, axis=0)
                        class_indices = np.argmax(classification, axis=-1)

                        start_indices = np.round(p_s * frame_length)
                        start_indices = np.clip(start_indices, 1, frame_length)
                        end_indices = np.round(p_e * frame_length)
                        end_indices = np.clip(end_indices, 1, frame_length)

                        valid_flags = end_indices - start_indices + 1 >= config.feature_frame_step_size

                        class_indices = class_indices[valid_flags]
                        start_indices = start_indices[valid_flags]
                        end_indices = end_indices[valid_flags]
                        scores = scores[valid_flags]

                        video_prediction_slices = \
                            pd.DataFrame(data={"class_index": class_indices,
                                               "start_index": start_indices,
                                               "end_index": end_indices,
                                               "score": scores})

                        if not config.use_soft_nms:
                            if config.use_classification:
                                nmsed_detection_slices = list()
                                video_prediction_slices = video_prediction_slices.groupby("class_index")
                                for class_index, slices in video_prediction_slices:
                                    slices = slices.values
                                    slices = nms(slices, threshold=config.nms_threshold)
                                    nmsed_detection_slices += slices.tolist()
                            else:
                                slices = video_prediction_slices.values
                                slices = nms(slices, threshold=config.nms_threshold)
                                nmsed_detection_slices = slices.tolist()

                            nmsed_detection_slices.sort(reverse=True, key=lambda x: x[-1])
                            nmsed_detection_slices = nmsed_detection_slices[:100]
                        else:
                            scores = torch.from_numpy(scores).float()
                            labels = torch.from_numpy(class_indices)
                            boxes = torch.from_numpy(np.stack((start_indices, end_indices), axis=-1)).float()
                            boxes, scores, labels = batched_nms(
                                boxes.contiguous(), scores.contiguous(), labels.contiguous(),
                                config.iou_threshold,
                                config.min_score,
                                config.max_seg_num,
                                use_soft_nms=True,
                                multiclass=config.multiclass_nms,
                                sigma=config.nms_sigma,
                                voting_thresh=config.voting_thresh)
                            boxes = torch.where(boxes.isnan(), torch.zeros_like(boxes), boxes).numpy()
                            labels = torch.where(labels.isnan(), torch.zeros_like(labels), labels).numpy()
                            scores = torch.where(scores.isnan(), torch.zeros_like(scores), scores).numpy()
                            nmsed_detection_slices = np.concatenate((labels[..., None], boxes, scores[..., None]),
                                                                    axis=-1)

                        detection_prediction_json["results"][identity] = list()
                        for prediction_slice in nmsed_detection_slices:
                            score = prediction_slice[-1]
                            prediction_class = int(prediction_slice[0])
                            label = datasets.label_dic[str(prediction_class)].replace("_", " ")
                            frame_intervals = [prediction_slice[1], prediction_slice[2]]
                            # time_intervals = [float(frame_intervals[0]) / config.video_fps,
                            #                   float(frame_intervals[1]) / config.video_fps]
                            time_intervals = [
                                float(frame_intervals[0] - 1) / (frame_length - 1) *
                                datasets.meta_dic["database"][identity]["duration"],
                                float(frame_intervals[1] - 1) / (frame_length - 1) *
                                datasets.meta_dic["database"][identity]["duration"]]

                            detection_prediction_json["results"][identity].append(
                                {"label": label, "score": score, "segment": time_intervals})

                            if config.dataset == "thumos14" and label == "CliffDiving":
                                detection_prediction_json["results"][identity].append(
                                    {"label": "Diving", "score": score, "segment": time_intervals})

                    print_string = \
                        "|{:10s}|Epoch {:3d}/{:3d}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                            "Validation", epoch, config.epochs, validation_batch_index + 1, loop_rounds, loss)
                    progress_step = validation_batch_index + 1
                    progress_length = loop_rounds
                    print_string += \
                        " |{}{}|".format(
                            "=" * int(round(37.0 * float(progress_step) / float(progress_length))),
                            " " * (37 - int(round(37.0 * float(progress_step) / float(progress_length)))))
                    sys.stdout.write("\r" + print_string)
                    sys.stdout.flush()

                    validation_batch_index += 1

                print()

                if config.dataset == "activitynet":
                    try:
                        anet_detection = ANETdetection(ground_truth_json,
                                                       detection_prediction_json,
                                                       subset="validation", check_status=True,
                                                       tiou_thresholds=np.linspace(0.5, 0.95, 10))

                        interpolated_mAP = anet_detection.evaluate()
                        overall_interpolated_mAP = interpolated_mAP.mean()
                        validation_mAP = overall_interpolated_mAP
                    except:
                        validation_mAP = 0.0

                for loss_name, loss_value in validation_losses.items():
                    validation_summary_writer.add_scalar(loss_name, loss_value / float(validation_batch_index),
                                                         epoch)
                validation_summary_writer.add_scalar("mAP", validation_mAP, epoch)

                validation_quality = validation_mAP

                if validation_quality >= best_validation:
                    best_validation = validation_quality
                    if previous_best_epoch and previous_best_epoch != epoch - config.ckpt_save_term:
                        weight_files = glob.glob(os.path.join(save_ckpt_file_folder,
                                                              "weights-{}.pt".format(previous_best_epoch)))
                        for file in weight_files:
                            try:
                                os.remove(file)
                            except OSError:
                                pass

                    if epoch % config.ckpt_save_term != 0:
                        torch.save(model.state_dict(),
                                   os.path.join(save_ckpt_file_folder, "weights-{}.pt".format(epoch)))
                    previous_best_epoch = epoch

                print("Validation Results ...")
                print("Validation Localization mAP {:.5f}".format(validation_mAP))
                print("=" * 90)


def nms(proposals, threshold=0.65):
    proposals = np.copy(sorted(proposals, key=lambda x: x[-1]))

    if threshold <= 0.0:
        return proposals

    # if there are no boxes, return an empty list
    if len(proposals) == 0:
        return np.array([])

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = proposals[:, 1]
    x2 = proposals[:, 2]

    # compute the area of the bounding boxes and grab the indexes to sort
    # area = (x2 - x1 + 1)

    idxs = np.arange(len(proposals))

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        union_x1 = np.minimum(x1[i], x1[idxs[:last]])
        union_x2 = np.maximum(x2[i], x2[idxs[:last]])
        intersection_x1 = np.maximum(x1[i], x1[idxs[:last]])
        intersection_x2 = np.minimum(x2[i], x2[idxs[:last]])

        # compute the width and height of the bounding box
        union_w = np.maximum(1, union_x2 - union_x1 + 1)
        intersection_w = np.maximum(0, intersection_x2 - intersection_x1 + 1)

        # compute the ratio of overlap
        overlap = intersection_w / union_w

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs,
                         np.concatenate(([last],
                                         np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked
    return proposals[pick]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--num_gpus", type=int, default=1)
    argparser.add_argument("--dataset", type=str, default=["thumos14", "activitynet"][1])
    argparser.add_argument("--postfix", type=str, default=None)

    args = argparser.parse_args()

    config = \
        {
            # base
            "num_gpus": args.num_gpus,
            "dataset": args.dataset,
            "postfix": args.postfix,

            # train
            "seed": 2023,
            "epochs": 1200 if args.dataset == "thumos14" else 120,
            "lr": 2.0e-4,
            "validation_term": 100 if args.dataset == "thumos14" else 10,
            "ckpt_save_term": 50 if args.dataset == "thumos14" else 5,
            "display_term": 1,
            "batch_size": 16 // args.num_gpus if args.dataset == "thumos14" else 16 // args.num_gpus,
            "num_workers": 48,
            "weight_decay": 1.0e-4,
            "clip_norm": 0.1,

            # test
            "nms_threshold": 0.65,
            "testing_step": 256,
            "use_soft_nms": False,
            "multiclass_nms": True,
            "max_seg_num": 100,
            "min_score": 0.001,
            "nms_sigma": 0.75,
            "voting_thresh": 0.9, # [0.75, 0.90]
            "iou_threshold": 0.1,

            # dataset
            "root_path": os.path.abspath(".."),
            "dataset_root_path": "/mnt/hdd0",
            "number_of_classes": 20 + 1 if args.dataset == "thumos14" else 200 + 1,
            "use_random_crop": True,
            "crop_length": 25 if args.dataset == "thumos14" else 9,
            "feature_frame_step_size": 8,
            "video_fps": 25.0,
            "temporal_width": 64,
            "feature_width": 512 if args.dataset == "thumos14" else 256,
            "dformat": "NDHWC",

            # model
            "model_name": "SelfDETR",
            "position_embedding": "sine",
            "hidden_dim": 256,
            "num_queries": 40,
            "dropout": 0.1,
            "nheads": 8,
            "dim_feedforward": 2048,
            "enc_layers": 2,
            "dec_layers": 4,
            "aux_loss": True,
            "seg_refine": True,
            "use_classification": True,
            "act_reg": False,
            "use_KK": True,
            "use_QQ": True,
            "cls_loss_coef": 2,
            "seg_loss_coef": 5,
            "iou_loss_coef": 2,
            "act_loss_coef": 4,
            "KK_loss_coef": 5,
            "QQ_loss_coef": 5,
            "set_cost_class": 6,
            "set_cost_seg": 5,
            "set_cost_iou": 2,
            "focal_alpha": 0.25,
        }

    config = EasyDict(config)

    train(config=config)
