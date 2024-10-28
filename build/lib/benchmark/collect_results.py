"""
This script collects results in a folder, calculates performance metrics and writes them to csv.
"""

import argparse
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from tqdm import tqdm
def traverse_path(path, output, cross=False, resampled=False, baer=False,operation="split",task1_threshold=0.1,task23_threshold=100):
    """
    Traverses the given path and extracts results for each experiment and version

    :param path: Root path
    :param output: Path to write results csv to
    :param cross: If true, expects cross-domain results.
    :return: None
    """
    path = Path(path)

    if not os.path.exists(output):
        os.makedirs(output)

    results = []

    exp_dirs = [x for x in path.iterdir() if x.is_dir()]
    for exp_dir in tqdm(exp_dirs):
        itr = exp_dir.iterdir()
        if baer:
            itr = [exp_dir]  # Missing version directory in the structure
        for version_dir in itr:
            if not version_dir.is_dir():
                pass

            results.append(
                process_version(
                    version_dir, cross=cross, resampled=resampled, baer=baer,
                    operation=operation, task1_threshold=task1_threshold, task23_threshold=task23_threshold,
                )
            )

    results = pd.DataFrame(results)
    if cross:
        sort_keys = ["data", "model", "target", "lr", "version"]
    else:
        sort_keys = ["data", "model", "lr", "version"]
    results.sort_values(sort_keys, inplace=True)
    results.to_csv(f"{output}/results_250.csv", index=False)


def process_version(version_dir: Path, cross: bool, resampled: bool, baer: bool,
                    operation,task1_threshold,task23_threshold,target_set: str= None):
    """
    Extracts statistics for the given version of the given experiment.

    :param version_dir: Path to the specific version
    :param cross: If true, expects cross-domain results.
    :return: Results dictionary
    """
    stats = parse_exp_name(version_dir, cross=cross, resampled=resampled, baer=baer)

    stats.update(eval_task1(version_dir,operation,task1_threshold,target_set))
    stats.update(eval_task23(version_dir,operation,task23_threshold,target_set))

    return stats


def parse_exp_name(version_dir, cross, resampled, baer):
    if baer:
        exp_name = version_dir.name
        version = "0"
    else:
        exp_name = version_dir.parent.name
        version = version_dir.name.split("_")[-1]

    parts = exp_name.split("_")
    target = None
    sampling_rate = None
    if cross or baer:
        if len(parts) == 4:
            data, model, lr, target = parts
        else:
            data, model, target = parts
            lr = "0.001"
    elif resampled:
        if len(parts) == 5:
            data, model, lr, target, sampling_rate = parts
        else:
            data, model, target, sampling_rate = parts
            lr = "0.001"
    else:
        if len(parts) == 3:
            data, model, lr = parts
        else:
            data, model = parts
            lr = "0.001"

    lr = float(lr)

    stats = {
        "experiment": exp_name,
        "data": data,
        "model": model,
        "lr": lr,
        "version": version,
    }

    if cross or baer:
        stats["target"] = target
    if resampled:
        stats["target"] = target
        stats["sampling_rate"] = sampling_rate

    return stats


def eval_task1(version_dir: Path, operation: str , task1_threshold:float, target_set):
    task1_files = list(version_dir.glob("*_task1.csv"))

    if not task1_files:
        logging.warning(f"No task 1 files found in directory {version_dir}")
        return {}

    if operation == 'merge':
        print('Merging all task 1 files and calculating metrics!')
        # Merge all *_task1.csv files
        merged_df = pd.concat([pd.read_csv(file) for file in task1_files])
        stats = calculate_stats(merged_df,task1_threshold)

    elif operation == 'split':
        stats = {}
        for task_file in task1_files:
            file_key = task_file.stem.split('_')[0]  # e.g., 'dev', 'test', 'train'
            if target_set and file_key not in target_set:
                continue  # Skip files that don't match the target set
            print(f"Processing task file {task_file.name}!")
            df = pd.read_csv(task_file,low_memory=False)
            stats.update(calculate_stats(df, task1_threshold,file_key))

    return stats


def calculate_stats(df, threshold: float, prefix=''):
    df["trace_type_bin"] = df["trace_type"] == "earthquake"

    prec, recall, f1, _ = precision_recall_fscore_support(
        df["trace_type_bin"],
        df["score_detection"] > threshold,
        average="binary",
    )

    auc = roc_auc_score(df["trace_type_bin"], df["score_detection"])

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        df["trace_type_bin"],
        df["score_detection"] > threshold
    ).ravel()

    return {
        f"{prefix}_det_precision": prec,
        f"{prefix}_det_recall": recall,
        f"{prefix}_det_f1": f1,
        f"{prefix}_det_auc": auc,
        f"{prefix}_det_tp": tp,
        f"{prefix}_det_fp": fp,
        f"{prefix}_det_tn": tn,
        f"{prefix}_det_fn": fn,
    }

def eval_task23(version_dir: Path, operation: str, task23_threshold: int, target_set):
    task23_files = list(version_dir.glob("*_task23.csv"))

    if not task23_files:
        logging.warning(f"No task 23 files found in directory {version_dir}")
        return {}

    if operation == 'merge':
        print('Merging all task 23 files and calculating metrics!')
        # Merge all *_task23.csv files
        merged_df = pd.concat([pd.read_csv(file) for file in task23_files])
        stats = calculate_task23_metrics(merged_df, task23_threshold)

    elif operation == 'split':
        stats = {}
        for task_file in task23_files:
            file_key = task_file.stem.split('_')[0]  # e.g., 'dev', 'test', 'train'
            if target_set and file_key not in target_set:
                continue  # Skip files that don't match the target set
            print(f"Processing task file {task_file.name}!")
            df = pd.read_csv(task_file,low_memory=False)
            stats.update(calculate_task23_metrics(df, task23_threshold, file_key))

    return stats

def calculate_task23_metrics(df, threshold: int, prefix=''):
    def add_aux_columns(pred):
        for col in ["p_sample_pred", "s_sample_pred"]:
            if col not in pred.columns:
                pred[col] = np.nan

    add_aux_columns(df)

    def nanmask(pred):
        """
        Returns all entries that are nan in score_p_or_s, p_sample_pred, and s_sample_pred
        """
        mask = np.logical_and(
            np.isnan(pred["p_sample_pred"]), np.isnan(pred["s_sample_pred"])
        )
        return mask

    if nanmask(df).all():
        logging.warning(f"Data contains NaN predictions for tasks 2 and 3")
        return {}

    df = df[~nanmask(df)]

    skip_task2 = False
    if (
        np.logical_or(
            np.isnan(df["score_p_or_s"]), np.isinf(df["score_p_or_s"])
        ).all()
    ):
        skip_task2 = True

    df["score_p_or_s"] = np.clip(df["score_p_or_s"].values, -1e100, 1e100)
    df_restricted = df[~np.isnan(df["score_p_or_s"])]

    stats = {}
    if len(df_restricted) > 0 and not skip_task2:
        mcc_thrs = np.sort(df["score_p_or_s"].values)
        mcc_thrs = mcc_thrs[np.linspace(0, len(mcc_thrs) - 1, 50, dtype=int)]
        mccs = [matthews_corrcoef(df["phase_label"] == "P", df["score_p_or_s"] > thr) for thr in mcc_thrs]
        mcc = np.max(mccs)
        mcc_thr = mcc_thrs[np.argmax(mccs)]

        stats.update({
            f"{prefix}_phase_mcc": mcc,
            f"{prefix}_phase_threshold_mcc": mcc_thr,
        })

    for phase in ["P", "S"]:
        phase_df = df[df["phase_label"] == phase]
        if phase_df.empty:
            continue

        pred_col = f"{phase.lower()}_sample_pred"
        diff = np.abs(phase_df[pred_col] - phase_df["phase_onset"])

        y_true = np.ones(len(phase_df), dtype=bool)
        y_pred = diff <= threshold

        # Compute precision, recall, F1 score, and confusion matrix
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

        stats.update({
            f"{prefix}_{phase}_precision": precision,
            f"{prefix}_{phase}_recall": recall,
            f"{prefix}_{phase}_f1": f1,
            f"{prefix}_{phase}_tp": tp,
            f"{prefix}_{phase}_fp": fp,
            f"{prefix}_{phase}_fn": fn,
            f"{prefix}_{phase}_tn": tn,
            f"{prefix}_{phase}_mean_s": np.mean(diff / phase_df["sampling_rate"]),
            f"{prefix}_{phase}_std_s": np.std(diff / phase_df["sampling_rate"]),
            f"{prefix}_{phase}_mae_s": np.mean(np.abs(diff / phase_df["sampling_rate"])),
        })

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collects results from all experiments in a folder and outputs them in condensed csv format."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Root path of predictions",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path for the output csv",
    )
    parser.add_argument(
        "--cross", action="store_true", help="Expect cross-domain results."
    )
    parser.add_argument(
        "--resampled",
        action="store_true",
        help="Expect cross-domain cross-sampling rate results.",
    )
    parser.add_argument(
        "--baer",
        action="store_true",
        help="Expect results from Baer-Kradolfer picker.",
    )
    parser.add_argument(
        "--operation",
        default= 'split',
        type=str,
        choices=['merge', 'split'],
        help="Operation type for task 1, either 'merge' or 'split'.",
    )
    parser.add_argument(
        "--task1_threshold",
        type=int,
        default=0.1,
        help="Threshold value for task 23 metrics.",
    )
    parser.add_argument(
        "--task23_threshold",
        type=int,
        default=100,
        help="Threshold value for task 23 metrics.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.operation == 'split':
        parser.add_argument(
            "--target_set",
            type=str,
            default='test',
            help="Target set to process for task 1 if splitting.",
        )

    traverse_path(
        args.path,
        args.output,
        cross=args.cross,
        resampled=args.resampled,
        baer=args.baer,
        operation=args.operation,
    )
