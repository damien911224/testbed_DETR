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
import matplotlib
import seaborn as sn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    torch.set_num_threads(config.num_workers)

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

    if config.use_wandb:
        wandb.init(project=config.model_name, config=dict(config), tags=[config.dataset, config.postfix])
        # wandb.run.name = "{}_{}_{}{}".format(model_name, config.dataset, train_date,
        #                                      "_{}".format(config.postfix) if config.postfix is not None else "")
        # wandb.run.save()

    if config.postfix is not None:
        save_ckpt_file_folder += "_{}".format(config.postfix)
        summary_folder += "_{}".format(config.postfix)

    train_summary_file_path = os.path.join(summary_folder, "train_summary")
    validation_summary_file_path = os.path.join(summary_folder, "validation_summary")

    boundaries = (round(config.epochs * 0.8), round(config.epochs * 0.9))
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
    if config.dataset == "activitynet":
        with open(os.path.join(datasets.meta_folder, "activity_net_1_3_new.json"), "r") as fp:
            new_GT_json = json.load(fp)
    for epoch in range(1, config.epochs + 1, 1):
        model.train()
        epoch_losses = dict()
        epoch_time = 0.0
        epoch_training_time = 0.0
        epoch_batch_iteration = 0
        epoch_preprocessing_time = 0.0

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
            # losses = list()
            # for k in loss_dict.keys():
            #     if k in weight_dict:
            #         if "QQ"  in k or "KK" in k:
            #             if epoch < 30:
            #                 continue
            #         losses.append(loss_dict[k] * weight_dict[k])
            # losses = torch.stack(losses).sum()
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

        if config.use_wandb:
            wandb_log_dict = dict({"train": dict()})

        for loss_name, loss_value in epoch_losses.items():
            train_summary_writer.add_scalar(loss_name, loss_value / float(epoch_batch_iteration), epoch)
            if config.use_wandb:
                wandb_log_dict["train"][loss_name] = loss_value / float(epoch_batch_iteration)

        if config.use_wandb:
            wandb.log(data=wandb_log_dict, step=epoch)

        if epoch % config.ckpt_save_term == 0:
            torch.save(model.state_dict(), os.path.join(save_ckpt_file_folder, "weights-{}.pt".format(epoch)))

        train_summary_writer.flush()

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
            with torch.no_grad():
                if config.dataset == "thumos14":
                    validation_batch_index = 0
                    with open(datasets.target_path, "r") as json_fp:
                        ground_truth_json = json.loads(json_fp.read())
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

                        loop_index = 0
                        all_pred_logits = list()
                        all_pred_segments = list()
                        for start_idx in range(0, len(features), testing_step):
                            this_features = features[start_idx:start_idx + feature_width]

                            if len(this_features) < feature_width:
                                this_features = \
                                    np.concatenate([this_features,
                                                    np.tile(np.expand_dims(np.zeros_like(this_features[0]), axis=0),
                                                            (config.feature_width - len(this_features), 1))],
                                                   axis=0)

                            this_features = torch.from_numpy(this_features).transpose(0, 1).cuda()

                            predictions = model(this_features.unsqueeze(0))
                            pred_logits = predictions["pred_logits"].squeeze(0).sigmoid().detach().cpu()
                            pred_segments = predictions["pred_segments"].squeeze(0).detach().cpu()
                            pred_segments = segment_ops.segment_cw_to_t1t2(pred_segments)
                            all_pred_logits.append(pred_logits)
                            all_pred_segments.append(pred_segments)

                            loop_index += 1

                        '''
                        Localization
                        '''
                        frame_length = validation_data.frame_lengths[identity]

                        all_class_indices = list()
                        all_start_indices = list()
                        all_end_indices = list()
                        all_scores = list()
                        num_loops = len(all_pred_logits)
                        for loop_index in range(num_loops):
                            this_pred_logits = all_pred_logits[loop_index]
                            this_pred_segments = all_pred_segments[loop_index]

                            Q, C = this_pred_logits.shape
                            # sort across different instances, pick top 100 at most
                            topk_values, topk_indexes = torch.topk(this_pred_logits.flatten(), min(200, Q * C), dim=0)
                            scores = topk_values.numpy()
                            topk_segments = topk_indexes // C
                            class_indices = (topk_indexes % C).numpy() + 1
                            # bs, nq, 2; bs, num, 2
                            segments = torch.gather(this_pred_segments, 0, topk_segments.unsqueeze(-1).repeat(1, 2))
                            p_s = segments[..., 0].numpy()
                            p_e = segments[..., 1].numpy()

                            # this_pred_logits = this_pred_logits.numpy()
                            # this_pred_segments = this_pred_segments.numpy()

                            # p_s = this_pred_segments[..., 0]
                            # p_e = this_pred_segments[..., 1]
                            # scores = np.max(this_pred_logits, axis=-1)

                            # class_indices = np.argmax(this_pred_logits, axis=-1)
                            # class_indices += 1

                            valid_flags = p_e >= p_s
                            p_s = p_s[valid_flags]
                            p_e = p_e[valid_flags]
                            scores = scores[valid_flags]
                            class_indices = class_indices[valid_flags]

                            start_indices = loop_index * testing_step + p_s * (feature_width - 1)
                            start_indices = np.clip(start_indices, 0, feature_length - 1)
                            start_indices = (start_indices / (feature_length - 1)) * (frame_length - 1) + 1
                            end_indices = loop_index * testing_step + p_e * (feature_width - 1)
                            end_indices = np.clip(end_indices, 0, feature_length - 1)
                            end_indices = (end_indices / (feature_length - 1)) * (frame_length - 1) + 1

                            valid_flags = end_indices - start_indices + 1 >= config.feature_frame_step_size

                            class_indices = class_indices[valid_flags]
                            start_indices = start_indices[valid_flags]
                            end_indices = end_indices[valid_flags]
                            scores = scores[valid_flags]

                            all_class_indices.append(class_indices)
                            all_start_indices.append(start_indices)
                            all_end_indices.append(end_indices)
                            all_scores.append(scores)

                        all_class_indices = np.concatenate(all_class_indices, axis=0)
                        all_start_indices = np.concatenate(all_start_indices, axis=0)
                        all_end_indices = np.concatenate(all_end_indices, axis=0)
                        all_scores = np.concatenate(all_scores, axis=0)

                        video_prediction_slices = \
                            pd.DataFrame(data={"class_index": all_class_indices,
                                               "start_index": all_start_indices,
                                               "end_index": all_end_indices,
                                               "score": all_scores})

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
                            #                   float(frame_intervals[1]) / config.dataset.video_fps]
                            time_intervals = [float(frame_intervals[0] - 1) / (frame_length - 1) *
                                              datasets.meta_dic["database"][identity]["duration"],
                                              float(frame_intervals[1] - 1) / (frame_length - 1) *
                                              datasets.meta_dic["database"][identity]["duration"]]

                            detection_prediction_json["results"][identity].append(
                                {"label": label, "score": score, "segment": time_intervals})

                            if config.dataset == "thumos14" and label == "CliffDiving":
                                detection_prediction_json["results"][identity].append(
                                    {"label": "Diving", "score": score, "segment": time_intervals})

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
                else:
                    valid_num = 0.0
                    validation_mAP = 0.0
                    validation_batch_index = 0
                    validation_losses = dict()

                    loop_rounds = \
                        int(math.ceil(float(validation_data.data_count) / float(config.batch_size * config.num_gpus)))

                    with open(datasets.target_path, "r") as json_fp:
                        first_ground_truth_json = json.loads(json_fp.read())
                    ground_truth_json = dict(first_ground_truth_json)
                    ground_truth_json["database"] = dict()
                    detection_prediction_json = dict({"version": "VERSION 1.3", "results": {},
                                                      "external_data":
                                                          {"used": True,
                                                           "details": "CNN is pre-trained on Kinetics-400"}})

                    KK_images = list()
                    QQ_images = list()
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
                        pred_segments = predictions["pred_segments"].detach().cpu()
                        pred_segments = segment_ops.segment_cw_to_t1t2(pred_segments).numpy()
                        if config.act_reg:
                            pred_actionness = predictions["pred_actionness"].detach().cpu().numpy()
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
                            if identity not in new_GT_json["database"]:
                                continue
                            frame_length = frame_lengths[n_i]
                            cuhk_classification_scores = np.array(all_anet2017_cuhk["results"][identity])

                            this_pred_logits = pred_logits[n_i]
                            this_pred_segments = pred_segments[n_i]

                            p_s = this_pred_segments[..., 0]
                            p_e = this_pred_segments[..., 1]
                            scores = np.max(this_pred_logits, axis=-1)

                            if config.act_reg:
                                scores *= pred_actionness[n_i].squeeze(-1)

                            valid_flags = p_e >= p_s
                            p_s = p_s[valid_flags]
                            p_e = p_e[valid_flags]
                            scores = scores[valid_flags]

                            classification = this_pred_logits + np.expand_dims(cuhk_classification_scores, axis=0)
                            class_indices = np.argmax(classification, axis=-1)
                            class_indices += 1

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

                            for annotation in first_ground_truth_json["database"][identity]["annotations"]:
                                if identity in ground_truth_json["database"]:
                                    ground_truth_json["database"][identity]["annotations"].append(annotation)
                                else:
                                    ground_truth_json["database"][identity] = \
                                        dict(first_ground_truth_json["database"][identity])
                                    ground_truth_json["database"][identity]["annotations"] = [annotation]

                        if validation_batch_index < 1:
                            for n_i in range(len(identities))[:3]:
                                QK = predictions["C_weights"][-1][n_i].detach().cpu()
                                tgt_KK = torch.sqrt(torch.matmul(QK.transpose(0, 1), QK))
                                tgt_KK = (tgt_KK / torch.sum(tgt_KK, dim=-1, keepdim=True)).numpy()
                                tgt_QQ = torch.sqrt(torch.matmul(QK, QK.transpose(0, 1)))
                                tgt_QQ = (tgt_QQ / torch.sum(tgt_QQ, dim=-1, keepdim=True)).numpy()

                                this_targets = target_dict[n_i]["segments"].detach().cpu().numpy()

                                src_KK = predictions["K_weights"][:, n_i].detach().cpu().numpy()
                                src_KK_mean = np.mean(src_KK, axis=0)

                                L, H, W = src_KK.shape
                                KK_box = np.zeros(dtype=np.float32, shape=(1 + H // 40, W))
                                for box in this_targets:
                                    s_i = round((box[0] - box[1] / 2) * (W - 1))
                                    e_i = round((box[0] + box[1] / 2) * (W - 1))
                                    KK_box[1:, s_i:e_i + 1] = 1.0
                                snes, plot_axis, num_plots = list(), 0, L + 2 + 1
                                H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * (H // 40)
                                W_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                                fig, axs = plt.subplots(1, num_plots,
                                                        figsize=(10 * num_plots, 10),
                                                        gridspec_kw={'width_ratios': [1] * (num_plots - 1) + [0.08]})
                                maps = [tgt_KK, src_KK_mean] + [src_KK[l_i] for l_i in range(L)]
                                titles = ["Tgt", "Src Mean"] + ["Src_L{:02d}".format(l_i + 1) for l_i in range(L)]
                                for map, title in zip(maps, titles):
                                    map -= np.min(map)
                                    map /= np.max(map)
                                    map = np.concatenate((map, KK_box), axis=0)
                                    df = pd.DataFrame(map, H_labels, W_labels)
                                    this_sn = sn.heatmap(df, cmap="YlGnBu", cbar=plot_axis >= num_plots - 1,
                                                         ax=axs[plot_axis], cbar_ax=axs[-1])
                                    this_sn.set_xlabel("")
                                    this_sn.set_ylabel("")
                                    snes.append(this_sn)
                                    axs[plot_axis].set_title(title)
                                    plot_axis += 1

                                for snx in snes:
                                    tl = snx.get_xticklabels()
                                    snx.set_xticklabels(tl, rotation=90)
                                    tly = snx.get_yticklabels()
                                    snx.set_yticklabels(tly, rotation=0)

                                fig.canvas.draw()
                                vis_array = np.array(fig.canvas.renderer._renderer)
                                plt.close(fig)
                                KK_images.append(vis_array)

                                src_QQ = predictions["Q_weights"][:, n_i].detach().cpu().numpy()
                                src_QQ_mean = np.mean(src_QQ, axis=0)

                                L, H, W = src_QQ.shape
                                QQ_box = np.zeros(dtype=np.float32, shape=(1 + H // 40, W))
                                for box in this_targets:
                                    s_i = round((box[0] - box[1] / 2) * (W - 1))
                                    e_i = round((box[0] + box[1] / 2) * (W - 1))
                                    QQ_box[1:, s_i:e_i + 1] = 1.0
                                snes, plot_axis, num_plots = list(), 0, L + 2 + 1
                                H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * (H // 40)
                                W_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                                fig, axs = plt.subplots(1, num_plots,
                                                        figsize=(10 * num_plots, 10),
                                                        gridspec_kw={'width_ratios': [1] * (num_plots - 1) + [0.08]})
                                maps = [tgt_QQ, src_QQ_mean] + [src_QQ[l_i] for l_i in range(L)]
                                titles = ["Tgt", "Src Mean"] + ["Src_L{:02d}".format(l_i + 1) for l_i in range(L)]
                                for map, title in zip(maps, titles):
                                    map -= np.min(map)
                                    map /= np.max(map)
                                    map = np.concatenate((map, QQ_box), axis=0)
                                    df = pd.DataFrame(map, H_labels, W_labels)
                                    this_sn = sn.heatmap(df, cmap="YlGnBu", cbar=plot_axis >= num_plots - 1,
                                                         ax=axs[plot_axis], cbar_ax=axs[num_plots - 1])
                                    this_sn.set_xlabel("")
                                    this_sn.set_ylabel("")
                                    snes.append(this_sn)
                                    axs[plot_axis].set_title(title)
                                    plot_axis += 1

                                for snx in snes:
                                    tl = snx.get_xticklabels()
                                    snx.set_xticklabels(tl, rotation=90)
                                    tly = snx.get_yticklabels()
                                    snx.set_yticklabels(tly, rotation=0)

                                fig.canvas.draw()
                                vis_array = np.array(fig.canvas.renderer._renderer)
                                plt.close(fig)
                                QQ_images.append(vis_array)

                                # map = predictions["K_weights"][-1, n_i].detach().cpu().numpy()
                                # H, W = map.shape
                                # H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                                # W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                                # map -= np.min(map)
                                # map /= np.max(map)
                                # df = pd.DataFrame(map, H_labels, W_labels)
                                # ax = sn.heatmap(df, cbar=False, xticklabels=False, yticklabels=False, square=True)
                                # plt.savefig(os.path.join("../temp", "K_E{:02d}.png".format(epoch)))
                                # plt.close()
                                #
                                # map = predictions["Q_weights"][-1, n_i].detach().cpu().numpy()
                                # H, W = map.shape
                                # H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                                # W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                                # map -= np.min(map)
                                # map /= np.max(map)
                                # df = pd.DataFrame(map, H_labels, W_labels)
                                # ax = sn.heatmap(df, cbar=False, xticklabels=False, yticklabels=False, square=True)
                                # plt.savefig(os.path.join("../temp", "Q_E{:02d}.png".format(epoch)))
                                # plt.close()

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

                    if config.use_wandb:
                        wandb_log_dict = dict({"validation": dict()})

                    for k_i in range(len(KK_images)):
                        this_image = KK_images[k_i]
                        validation_summary_writer.add_image("KK_{:02d}".format(k_i + 1), this_image, epoch,
                                                            dataformats="HWC")
                        if config.use_wandb:
                            wandb_image = wandb.Image(this_image)
                            wandb_log_dict["validation"]["KK_{:02d}".format(k_i + 1)] = wandb_image

                    for q_i in range(len(QQ_images)):
                        this_image = QQ_images[q_i]
                        validation_summary_writer.add_image("QQ_{:02d}".format(q_i + 1), this_image, epoch,
                                                            dataformats="HWC")
                        if config.use_wandb:
                            wandb_image = wandb.Image(this_image)
                            wandb_log_dict["validation"]["QQ_{:02d}".format(q_i + 1)] = wandb_image

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
                        if config.use_wandb:
                            wandb_log_dict["validation"][loss_name] = loss_value / float(validation_batch_index)
                    validation_summary_writer.add_scalar("mAP", validation_mAP, epoch)
                    if config.use_wandb:
                        wandb_log_dict["validation"]["mAP"] = validation_mAP
                        if validation_mAP > 0.0:
                            wandb_log_dict["validation"]["AP@0.50"] = interpolated_mAP[0]
                            wandb_log_dict["validation"]["AP@0.75"] = interpolated_mAP[5]
                            wandb_log_dict["validation"]["AP@0.95"] = interpolated_mAP[-1]

                    if config.use_wandb:
                        wandb.log(data=wandb_log_dict, step=epoch)

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

            validation_summary_writer.flush()

    train_summary_writer.close()
    validation_summary_writer.close()


def test(config):
    model_name = config.model_name

    saved_date = "0226"
    saved_postfix = "v2_base"
    target_epoch = 110

    model, criterion = build(config)
    datasets = Datasets(config)

    train_data, validation_data = datasets.getDataset("train")

    load_ckpt_file = os.path.join(config.root_path, "networks", "weights",
                                  "save", "{}_{}_{}_{}".format(model_name, config.dataset,
                                                               saved_date, saved_postfix),
                                  "weights-{}.pt".format(target_epoch))

    model.load_state_dict(torch.load(load_ckpt_file))
    model = model.cuda()

    if config.dataset == "activitynet":
        with open(os.path.join(datasets.datasets_folder,
                               "ANET2017-CUHK-GTAD", "cuhk_val_simp_share.json"), "r") as fp:
            all_anet2017_cuhk = json.load(fp)

    if config.dataset == "activitynet":
        with open(os.path.join(datasets.meta_folder, "activity_net_1_3_new.json"), "r") as fp:
            new_GT_json = json.load(fp)

    model.eval()
    with torch.no_grad():
        if config.dataset == "thumos14":
            validation_batch_index = 0
            with open(datasets.target_path, "r") as json_fp:
                ground_truth_json = json.loads(json_fp.read())
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

                loop_index = 0
                all_pred_logits = list()
                all_pred_segments = list()
                for start_idx in range(0, len(features), testing_step):
                    this_features = features[start_idx:start_idx + feature_width]

                    if len(this_features) < feature_width:
                        this_features = \
                            np.concatenate([this_features,
                                            np.tile(np.expand_dims(np.zeros_like(this_features[0]), axis=0),
                                                    (config.feature_width - len(this_features), 1))],
                                           axis=0)

                    this_features = torch.from_numpy(this_features).transpose(0, 1).cuda()

                    predictions = model(this_features.unsqueeze(0))
                    pred_logits = predictions["pred_logits"].squeeze(0).sigmoid().detach().cpu()
                    pred_segments = predictions["pred_segments"].squeeze(0).detach().cpu()
                    pred_segments = segment_ops.segment_cw_to_t1t2(pred_segments)
                    all_pred_logits.append(pred_logits)
                    all_pred_segments.append(pred_segments)

                    loop_index += 1

                '''
                Localization
                '''
                frame_length = validation_data.frame_lengths[identity]

                all_class_indices = list()
                all_start_indices = list()
                all_end_indices = list()
                all_scores = list()
                num_loops = len(all_pred_logits)
                for loop_index in range(num_loops):
                    this_pred_logits = all_pred_logits[loop_index]
                    this_pred_segments = all_pred_segments[loop_index]

                    Q, C = this_pred_logits.shape
                    # sort across different instances, pick top 100 at most
                    topk_values, topk_indexes = torch.topk(this_pred_logits.flatten(), min(200, Q * C), dim=0)
                    scores = topk_values.numpy()
                    topk_segments = topk_indexes // C
                    class_indices = (topk_indexes % C).numpy() + 1
                    # bs, nq, 2; bs, num, 2
                    segments = torch.gather(this_pred_segments, 0, topk_segments.unsqueeze(-1).repeat(1, 2))
                    p_s = segments[..., 0].numpy()
                    p_e = segments[..., 1].numpy()

                    # this_pred_logits = this_pred_logits.numpy()
                    # this_pred_segments = this_pred_segments.numpy()

                    # p_s = this_pred_segments[..., 0]
                    # p_e = this_pred_segments[..., 1]
                    # scores = np.max(this_pred_logits, axis=-1)

                    # class_indices = np.argmax(this_pred_logits, axis=-1)
                    # class_indices += 1

                    valid_flags = p_e >= p_s
                    p_s = p_s[valid_flags]
                    p_e = p_e[valid_flags]
                    scores = scores[valid_flags]
                    class_indices = class_indices[valid_flags]

                    start_indices = loop_index * testing_step + p_s * (feature_width - 1)
                    start_indices = np.clip(start_indices, 0, feature_length - 1)
                    start_indices = (start_indices / (feature_length - 1)) * (frame_length - 1) + 1
                    end_indices = loop_index * testing_step + p_e * (feature_width - 1)
                    end_indices = np.clip(end_indices, 0, feature_length - 1)
                    end_indices = (end_indices / (feature_length - 1)) * (frame_length - 1) + 1

                    valid_flags = end_indices - start_indices + 1 >= config.feature_frame_step_size

                    class_indices = class_indices[valid_flags]
                    start_indices = start_indices[valid_flags]
                    end_indices = end_indices[valid_flags]
                    scores = scores[valid_flags]

                    all_class_indices.append(class_indices)
                    all_start_indices.append(start_indices)
                    all_end_indices.append(end_indices)
                    all_scores.append(scores)

                all_class_indices = np.concatenate(all_class_indices, axis=0)
                all_start_indices = np.concatenate(all_start_indices, axis=0)
                all_end_indices = np.concatenate(all_end_indices, axis=0)
                all_scores = np.concatenate(all_scores, axis=0)

                video_prediction_slices = \
                    pd.DataFrame(data={"class_index": all_class_indices,
                                       "start_index": all_start_indices,
                                       "end_index": all_end_indices,
                                       "score": all_scores})

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
                    #                   float(frame_intervals[1]) / config.dataset.video_fps]
                    time_intervals = [float(frame_intervals[0] - 1) / (frame_length - 1) *
                                      datasets.meta_dic["database"][identity]["duration"],
                                      float(frame_intervals[1] - 1) / (frame_length - 1) *
                                      datasets.meta_dic["database"][identity]["duration"]]

                    detection_prediction_json["results"][identity].append(
                        {"label": label, "score": score, "segment": time_intervals})

                    if config.dataset == "thumos14" and label == "CliffDiving":
                        detection_prediction_json["results"][identity].append(
                            {"label": "Diving", "score": score, "segment": time_intervals})

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
        else:
            valid_num = 0.0
            validation_batch_index = 0
            validation_losses = dict()

            loop_rounds = \
                int(math.ceil(float(validation_data.data_count) / float(config.batch_size * config.num_gpus)))

            with open(datasets.target_path, "r") as json_fp:
                first_ground_truth_json = json.loads(json_fp.read())
            ground_truth_json = dict(first_ground_truth_json)
            ground_truth_json["database"] = dict()
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
                pred_segments = predictions["pred_segments"].detach().cpu()
                pred_segments = segment_ops.segment_cw_to_t1t2(pred_segments).numpy()
                if config.act_reg:
                    pred_actionness = predictions["pred_actionness"].detach().cpu().numpy()
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
                    if identity not in new_GT_json["database"]:
                        continue
                    frame_length = frame_lengths[n_i]
                    cuhk_classification_scores = np.array(all_anet2017_cuhk["results"][identity])

                    this_pred_logits = pred_logits[n_i]
                    this_pred_segments = pred_segments[n_i]

                    p_s = this_pred_segments[..., 0]
                    p_e = this_pred_segments[..., 1]
                    scores = np.max(this_pred_logits, axis=-1)

                    if config.act_reg:
                        scores *= pred_actionness[n_i].squeeze(-1)

                    valid_flags = p_e >= p_s
                    p_s = p_s[valid_flags]
                    p_e = p_e[valid_flags]
                    scores = scores[valid_flags]

                    classification = this_pred_logits + np.expand_dims(cuhk_classification_scores, axis=0)
                    class_indices = np.argmax(classification, axis=-1)
                    class_indices += 1

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

                    for annotation in first_ground_truth_json["database"][identity]["annotations"]:
                        if identity in ground_truth_json["database"]:
                            ground_truth_json["database"][identity]["annotations"].append(annotation)
                        else:
                            ground_truth_json["database"][identity] = \
                                dict(first_ground_truth_json["database"][identity])
                            ground_truth_json["database"][identity]["annotations"] = [annotation]

                print_string = \
                    "|{:10s}|Batch {:3d}/{:3d}|Loss: {:.2f}".format(
                        "Validation", validation_batch_index + 1, loop_rounds, loss)
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

    data_folder = os.path.join(datasets.datasets_folder,
                               "{}_{}_{}_{}".format(config.model_name,
                                                    config.dataset,
                                                    saved_date, saved_postfix))
    result_folder = os.path.join(data_folder, "results")
    folders = [data_folder, result_folder]
    for folder in folders:
        try:
            os.mkdir(folder)
        except OSError:
            pass

    result_string = "=" * 90 + "\n"
    result_string += "{}_{}_{} Testing Results\n\n".format(config.model_name, config.dataset, saved_postfix)
    result_string += "=" * 90 + "\n\n"

    result_string += "=" * 90 + "\n"
    result_string += "Localization Results ...\n"
    result_string += "NMS {:.2f} Block Configuration\n\n".format(config.nms_threshold)
    result_string += "=" * 90 + "\n\n"
    result_string += "\nDetection Evaluation\n"

    if config.dataset == "activitynet":
        anet_detection = ANETdetection(ground_truth_json,
                                       detection_prediction_json,
                                       subset="validation", check_status=True,
                                       tiou_thresholds=np.linspace(0.5, 0.95, 10))

        interpolated_mAP = anet_detection.evaluate()
        overall_interpolated_mAP = interpolated_mAP.mean()
    elif config.dataset == "thumos14":
        anet_detection = ANETdetection(ground_truth_json,
                                       detection_prediction_json,
                                       # replaced_labels,
                                       subset="validation", check_status=False,
                                       tiou_thresholds=np.arange(0.3, 0.8, 0.1))

        interpolated_mAP = anet_detection.evaluate()
        overall_interpolated_mAP = interpolated_mAP.mean()
    IoUs = np.arange(0.3, 0.8, 0.1) if config.dataset == "thumos14" else np.arange(0.5, 1.0, 0.05)
    for mAP_index in range(len(interpolated_mAP)):
        result_string += "IoU {:.2f}: {:.7f}\n".format(IoUs[mAP_index], interpolated_mAP[mAP_index])
    result_string += "OVERALL: {:.7f}".format(overall_interpolated_mAP)

    with open(os.path.join(result_folder,
                           "results_NmsTh{}.txt".format(
                               int(config.nms_threshold * 100.0),
                               "_{}".format(config.postfix) if config.postfix is not None else "")), "w") as fp:
        fp.write(result_string)

    with open(os.path.join(result_folder,
                           "detection_NmsTh{}.json".format(
                               int(config.nms_threshold * 100.0),
                               "_{}".format(config.postfix) if config.postfix is not None else ""
                           )), "w") as fp:
        json.dump(detection_prediction_json, fp, indent=4, sort_keys=True)

    print(result_string)

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

            # dataset
            "root_path": os.path.abspath(".."),
            "dataset_root_path": "/mnt/hdd0",
            "number_of_classes": 20 + 1 if args.dataset == "thumos14" else 200 + 1,
            "use_random_crop": True,
            "crop_length": 25 if args.dataset == "thumos14" else 9,
            "feature_frame_step_size": 8,
            "video_fps": 25.0,
            "temporal_width": 64,
            "feature_width": 256 if args.dataset == "thumos14" else 768,
            "dformat": "NDHWC",
            "copypaste_prob": 0.00,

            # train
            "seed": 2023,
            "epochs": 3000 if args.dataset == "thumos14" else 120,
            "lr": 2.0e-4 if args.dataset == "thumos14" else 1.0e-4,
            "validation_term": 10 if args.dataset == "thumos14" else 10,
            "ckpt_save_term": 100 if args.dataset == "thumos14" else 5,
            "display_term": 1,
            "batch_size": 16 // args.num_gpus if args.dataset == "thumos14" else 16 // args.num_gpus,
            "num_workers": 48,
            "prefetch_factor": 2,
            "weight_decay": 1.0e-4,
            "clip_norm": 0.1,
            "lr_decay_steps": (2500, 2800) if args.dataset == "thumos14" else (80, 100),
            "use_wandb": True,

            # test
            "nms_threshold": 0.65,
            "testing_step": 64,
            "use_soft_nms": True,
            "multiclass_nms": True,
            "max_seg_num": 100,
            "min_score": 1.0e-9,
            "nms_sigma": 0.75,
            "voting_thresh": 0.9,  # [0.75, 0.90]
            "iou_threshold": 0.10,

            # model
            "model_name": "SelfDETR",
            "position_embedding": "sine",
            "hidden_dim": 256,
            "num_queries": 100 if args.dataset == "thumos14" else 100,
            "dropout": 0.1,
            "nheads": 8,
            "dim_feedforward": 1024,  # 1024
            "enc_layers": 2 if args.dataset == "thumos14" else 2, # 2
            "dec_layers": 4 if args.dataset == "thumos14" else 4, # 4
            "aux_loss": True,
            "seg_refine": True,
            "use_classification": args.dataset == "thumos14", # args.dataset == "thumos14"
            "act_reg": False,
            "use_KK": True,
            "use_QQ": True,
            "cls_loss_coef": 2,
            "seg_loss_coef": 5,
            "iou_loss_coef": 2,
            "act_loss_coef": 4,
            "KK_loss_coef": 5,
            "QQ_loss_coef": 5,
            "set_cost_class": 2, # 6
            "set_cost_seg": 5,
            "set_cost_iou": 2,
            "focal_alpha": 0.25,
            "pos_temperature": 20, # 10000
        }

    config = EasyDict(config)

    # train(config=config)

    test(config=config)
