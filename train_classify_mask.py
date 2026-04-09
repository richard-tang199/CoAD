# -*-coding:utf-8-*-
import argparse
import os
import numpy as np
import torch
import time
from evaluation.evaluator import evaluate
from toolkit.result_plot import recon_plot, get_segments
from model.patch_soft_mask import PatchFrequencyMask
from toolkit.utils import *
import json
from dataclasses import dataclass, asdict
from toolkit.get_anomaly_score import AnomalyScoreCalculator
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='freq_tf_v2', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default="TSB-AD", help='Name of the dataset')
parser.add_argument("--group_name", type=str, default="225", help="group in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=20, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")
parser.add_argument("--patch_size", type=int, default=8, help="length of the patch")
parser.add_argument("--use_tensorboard", type=bool, default=False, help="use tensorboard for visualization")
parser.add_argument("--device", type=str, default='cuda:0', help="device for training")
parser.add_argument("--plot", type=str, default="True", help="whether to plot the results")
parser.add_argument("--omega", type=float, default=10, help="omega for loss function")
parser.add_argument("--window_multiple", type=int, default=4, help="window multiple for training")
parser.add_argument('--random_seed', type=int, default=46, help='random seed for reproducibility')
parser.add_argument("--timestamp", type=str, default="", help="timestamp for saving the results")

if __name__ == '__main__':
    args = parser.parse_args()
    args.group_name = [args.group_name]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    if args.dataset_name == "UCR":
        use_ucr = True
    else:
        use_ucr = False

    if args.plot.lower() == "true":
        args.plot = True
    elif args.plot.lower() == "false":
        args.plot = False
    else:
        raise ValueError("Invalid value for plot")

    save_dir = f"mask_results/{args.dataset_name}/{args.model_name}_{args.timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figure", exist_ok=True)
    os.makedirs(f"{save_dir}/config", exist_ok=True)
    os.makedirs(f"{save_dir}/model", exist_ok=True)

    if args.use_tensorboard:
        now = datetime.now().strftime("%m-%d-%H-%M")
        log_dir = f"logs/{args.model_name}/{args.dataset_name}_{args.group_name[0].zfill(3)}_{now}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(
            log_dir=f"logs/{args.model_name}/{args.dataset_name}_{args.group_name[0].zfill(3)}_{now}")

    print(f"Start training: {args.dataset_name}_{args.group_name[0]}, model_name: {args.model_name}, device: {device}")
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)
    use_train_data_list = train_data_list[:]
    train_subsequence_list = subsequence_list[:]

    for train_data, subsequence_length in zip(use_train_data_list, train_subsequence_list):
        if len(train_data.shape) == 1:
            train_data = train_data[:, np.newaxis]

        if train_data.shape[0] // subsequence_length > args.window_multiple:
            multiple = args.window_multiple
        else:
            multiple = len(train_data) // subsequence_length


        window_length = subsequence_length * multiple
        window_length = (window_length // args.patch_size) * args.patch_size

        if subsequence_length > window_length:
            subsequence_length = window_length

        if len(train_data) < 1000:
            train_stride = 2
        elif 1000 <= len(train_data) < 10000:
            train_stride = 6
        elif 10000 <= len(train_data) < 60000:
            train_stride = 10
        elif 60000 <= len(train_data) < 100000:
            train_stride = args.patch_size
        else:
            train_stride = subsequence_length

        train_loader, _ = get_dataloader(
            data=train_data,
            batch_size=args.batch_size,
            window_length=window_length,
            mode="train",
            train_stride=train_stride
        )

        if args.model_name == "mix_soft":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=window_length // args.patch_size,
                recon_mode="soft"
            )
        elif args.model_name == "mix_hard":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=window_length // args.patch_size,
                recon_mode="hard"
            )
        elif args.model_name == "freq_hard":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="hard",
                detect_mode="freq_time_v2"
            )
        elif args.model_name == "freq_gating":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="gating",
                detect_mode="freq_time_v2"
            )
        elif args.model_name == "time_soft":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="time"
            )
        elif args.model_name == "freq_tf":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="freq_time"
            )
        elif args.model_name == "freq_tf_v2":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=2,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="freq_time_v2",
                omega=args.omega
                )
        elif args.model_name == "freq_time_mean":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="freq_time_mean"
            )
        elif args.model_name == "freq_time_gate":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="freq_time_gate"
            )
        elif args.model_name == "freq_time_add":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="freq_time_add"
            )
        elif args.model_name == "freq_tf_v3":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="freq_time_v3"
                )
        elif args.model_name == "freq_tf_conv":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="conv"
            )
        elif args.model_name == "freq_time_point":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                detect_mode="freq_time_point",
                recon_mode="soft",
                omega=args.omega
                )
        elif args.model_name == "freq_time_window":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="hard",
                detect_mode="freq_time_window"
            )
        elif args.model_name == "guide_hard":
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="guide_hard",
                detect_mode="freq_time_v2"
            )
        else:
            model = PatchFrequencyMask(
                patch_size=args.patch_size,
                expansion_ratio=3,
                num_layers=3,
                n_fft=4,
                patch_num=None,
                recon_mode="soft",
                detect_mode="else"
            )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        epoch_time_list = []
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(args.num_epochs):
            start_time = time.time()
            model.train()
            epoch_loss = 0
            epoch_classify_loss = 0
            epoch_reconstruction_loss = 0

            for batch_idx, (data,) in enumerate(train_loader):
                data = data.to(device)
                data = data.permute(0, 2, 1).contiguous()
                batch_size, num_channels, seq_len = data.shape
                data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], 1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                    loss, classify_loss, reconstruction_loss = model(data, subsequence_length=subsequence_length)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.detach()
                epoch_classify_loss += classify_loss.detach()
                epoch_reconstruction_loss += reconstruction_loss.detach()

            mean_loss = epoch_loss.item() / len(train_loader)
            mean_classify_loss = epoch_classify_loss.item() / len(train_loader)
            mean_reconstruction_loss = epoch_reconstruction_loss.item() / len(train_loader)

            epoch_duration = time.time() - start_time
            epoch_time_list.append(epoch_duration)
            scheduler.step()
            print(f"Epoch {epoch}: Loss {mean_loss}")

            if args.use_tensorboard:
                writer.add_scalar("Loss/train", mean_loss, epoch)
                writer.add_scalar("Classify_Loss/train", mean_classify_loss, epoch)
                writer.add_scalar("Reconstruction_Loss/train", mean_reconstruction_loss, epoch)

            if args.dataset_name == "UCR":
                if mean_classify_loss < 0.03 and mean_reconstruction_loss < 0.003:
                    break

                if mean_loss < 0.05:
                    break

    for train_data, subsequence_length in zip(use_train_data_list, train_subsequence_list):
        if len(train_data.shape) == 1:
            train_data = train_data[:, np.newaxis]

        window_length = subsequence_length * multiple
        window_length = (window_length // args.patch_size) * args.patch_size
        if subsequence_length > window_length:
            subsequence_length = window_length
        val_loader, valid_window_converter = get_dataloader(
            data=train_data,
            batch_size=16,
            window_length=window_length,
            test_stride=subsequence_length
        )

    with torch.no_grad():
        model.eval()
        val_anomaly_score = None
        val_recon_value = None
        for batch_idx, (data,) in enumerate(val_loader):
            data = data.to(device)
            data = data.permute(0, 2, 1)
            batch_size, num_channels, seq_len = data.shape
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], 1)
            predict_score, recon_value = model.predict(data)
            predict_score = predict_score.reshape(batch_size, num_channels, seq_len)
            recon_value = recon_value.reshape(batch_size, num_channels, seq_len)
            predict_score = predict_score.permute(0, 2, 1)
            recon_value = recon_value.permute(0, 2, 1)

            if val_anomaly_score is None:
                val_anomaly_score = predict_score
            else:
                val_anomaly_score = torch.concat([val_anomaly_score, predict_score], dim=0)

            if val_recon_value is None:
                val_recon_value = recon_value
            else:
                val_recon_value = torch.concat([val_recon_value, recon_value], dim=0)

        val_anomaly_score = val_anomaly_score.detach().cpu().numpy()
        val_anomaly_score = valid_window_converter.windows_to_sequence(val_anomaly_score)
        threshold = np.mean(val_anomaly_score) + 3 * np.std(val_anomaly_score)
        val_recon_value = val_recon_value.detach().cpu().numpy()
        val_recon_value = valid_window_converter.windows_to_sequence(val_recon_value)

        for i, (test_data, test_label, subsequence_length) in enumerate(
                zip(test_data_list, test_label_list, subsequence_list)):
            current_group = args.group_name[i]

            if len(test_data.shape) == 1:
                test_data = test_data[:, np.newaxis]

            window_length = multiple * subsequence_length
            window_length = (window_length // args.patch_size) * args.patch_size

            if subsequence_length > window_length:
                subsequence_length = window_length
            test_loader, test_window_converter = get_dataloader(data=test_data,
                                                                batch_size=args.batch_size,
                                                                window_length=window_length,
                                                                test_stride=subsequence_length)

            test_anomaly_score = None
            test_recon_value = None

            test_start = time.time()
            for batch_idx, (data,) in enumerate(test_loader):
                data = data.to(device)
                data = data.permute(0, 2, 1)
                batch_size, num_channels, seq_len = data.shape
                data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], 1)
                predict_score, recon_value = model.predict(data, threshold=threshold)
                predict_score = predict_score.reshape(batch_size, num_channels, seq_len)
                recon_value = recon_value.reshape(batch_size, num_channels, seq_len)
                predict_score = predict_score.permute(0, 2, 1)
                recon_value = recon_value.permute(0, 2, 1)

                if test_anomaly_score is None:
                    test_anomaly_score = predict_score
                else:
                    test_anomaly_score = torch.concat([test_anomaly_score, predict_score], dim=0)

                if test_recon_value is None:
                    test_recon_value = recon_value
                else:
                    test_recon_value = torch.concat([test_recon_value, recon_value], dim=0)
            test_duration = time.time() - test_start

    test_anomaly_score = test_anomaly_score.detach().cpu().numpy()
    test_anomaly_score = test_window_converter.windows_to_sequence(test_anomaly_score)
    test_recon_value = test_recon_value.detach().cpu().numpy()
    test_recon_value = test_window_converter.windows_to_sequence(test_recon_value)

    args.group_name = args.group_name[0].zfill(3)

    save_path = os.path.join(save_dir, f"figure/group_{args.group_name}.png")
    # test_anomaly_score = np.squeeze(test_anomaly_score, axis=-1)


    average_window = subsequence_length

    anomaly_score_cal_classify = AnomalyScoreCalculator(
        mode="error",
        average_window=average_window,
        add_average=False
    )

    anomaly_score_cal_recon = AnomalyScoreCalculator(
        mode="error",
        average_window=average_window,
        add_average=False
    )

    recon_anomaly_score = anomaly_score_cal_recon.calculate_anomaly_score(
        raw_train_data=train_data,
        raw_test_data=test_data,
        recon_train_data=val_recon_value,
        recon_test_data=test_recon_value
    )

    classify_anomaly_score = anomaly_score_cal_classify.calculate_anomaly_score(
        raw_train_data=np.zeros_like(val_anomaly_score),
        raw_test_data=np.zeros_like(test_anomaly_score),
        recon_train_data=val_anomaly_score,
        recon_test_data=test_anomaly_score
    )

    test_recon_anomaly_score = recon_anomaly_score.test_score_all
    val_recon_anomaly_score = recon_anomaly_score.train_score_all

    test_recon_anomaly_score[:subsequence_length] = 0
    val_recon_anomaly_score[:subsequence_length] = 0

    test_classify_anomaly_score = classify_anomaly_score.test_score_all
    val_classify_anomaly_score = classify_anomaly_score.train_score_all

    test_classify_anomaly_score[:subsequence_length] = 0
    val_classify_anomaly_score[:subsequence_length] = 0

    anomaly_score_path = os.path.join(save_dir, f"anomaly_score")
    os.makedirs(anomaly_score_path, exist_ok=True)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_test_recon_anomaly_score.npy"), test_recon_anomaly_score)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_val_recon_anomaly_score.npy"), val_recon_anomaly_score)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_test_classify_anomaly_score.npy"), test_classify_anomaly_score)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_val_classify_anomaly_score.npy"), val_classify_anomaly_score)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_test_recon_value.npy"), test_recon_value)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_test_data.npy"), test_data)

    if args.dataset_name == "UCR" or args.dataset_name == "ASD":
        test_classify_anomaly_score = (test_classify_anomaly_score - np.min(test_classify_anomaly_score)) / (
                np.max(test_classify_anomaly_score) - np.min(test_classify_anomaly_score))
        val_classify_anomaly_score = (val_classify_anomaly_score - np.min(val_classify_anomaly_score)) / (
                np.max(val_classify_anomaly_score) - np.min(val_classify_anomaly_score))
        test_recon_anomaly_score = (test_recon_anomaly_score - np.min(test_recon_anomaly_score)) / (
                np.max(test_recon_anomaly_score) - np.min(test_recon_anomaly_score))
        val_recon_anomaly_score = (val_recon_anomaly_score - np.min(val_recon_anomaly_score)) / (
                np.max(val_recon_anomaly_score) - np.min(val_recon_anomaly_score))

    test_anomaly_score = (test_classify_anomaly_score + test_recon_anomaly_score) / 2
    val_anomaly_score = (val_classify_anomaly_score + val_recon_anomaly_score) / 2

    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_test_anomaly_score.npy"), test_anomaly_score)
    np.save(os.path.join(anomaly_score_path,
                         f"group_{args.group_name}_val_anomaly_score.npy"), val_anomaly_score)

    eval_result_recon = evaluate(ground_truth=test_label,
                                 anomaly_scores=test_recon_anomaly_score,
                                 subsequence=subsequence_length,
                                 use_ucr=use_ucr,
                                 mode="test")

    eval_result_classify = evaluate(ground_truth=test_label,
                                    anomaly_scores=test_classify_anomaly_score,
                                    subsequence=subsequence_length,
                                    use_ucr=use_ucr,
                                    mode="test")

    eval_result_all = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    test_threshold = np.mean(test_anomaly_score) + 3 * np.std(test_anomaly_score)
    # anomaly_index = test_anomaly_score > test_threshold
    # test_anomaly_segments = get_segments(anomaly_index)
    os.makedirs(f"{save_dir}/all/data", exist_ok=True)
    os.makedirs(f"{save_dir}/recon/data", exist_ok=True)
    os.makedirs(f"{save_dir}/classify/data", exist_ok=True)

    result_path_all = os.path.join(f"{save_dir}/all/data", f"group_{args.group_name}.json")
    result_path_recon = os.path.join(f"{save_dir}/recon/data", f"group_{args.group_name}.json")
    result_path_classify = os.path.join(f"{save_dir}/classify/data", f"group_{args.group_name}.json")

    # save model
    torch.save(model, os.path.join(save_dir, f"model/group_{args.group_name}.pth"))

    eval_result_all.test_duration = test_duration
    eval_result_all.train_duration = np.mean(epoch_time_list)

    with open(result_path_all, "w", encoding="utf-8") as f:
        json.dump(asdict(eval_result_all), f, indent=4)

    with open(result_path_recon, "w", encoding="utf-8") as f:
        json.dump(asdict(eval_result_recon), f, indent=4)

    with open(result_path_classify, "w", encoding="utf-8") as f:
        json.dump(asdict(eval_result_classify), f, indent=4)

    with open(f"{save_dir}/config/config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    if args.dataset_name == "ASD":
        figure_width = 60
    else:
        figure_width = 10

    if args.plot:
        recon_plot(
            save_path=save_path,
            gap=subsequence_list[0],
            figure_width=figure_width,
            figure_length=120,
            train_data=train_data_list[i],
            test_data=test_data,
            test_label=test_label,
            test_anomaly_score=test_anomaly_score,
            train_anomaly_score=val_anomaly_score,
            threshold=test_threshold,
            recon_train_data=val_recon_value,
            recon_test_data=test_recon_value
        )