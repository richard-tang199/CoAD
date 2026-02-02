import sys

sys.path.append(sys.path[0] + "/evaluation")

from TimeSeAD.evaluator import Evaluator
import torch
import numpy as np
import json
from dataclasses import dataclass, asdict
from sklearn import metrics
from scipy.signal import find_peaks
import pandas as pd
from toolkit.result_plot import get_segments
from evaluation.vus.metrics import get_range_vus_roc
from pate.PATE_metric import PATE


@dataclass
class EvaluationResult:
    point_wise: dict = None
    ucr: dict = None
    seAD: dict = None
    range_based: dict = None
    vus: dict = None
    pate: dict = None
    train_duration: float = None
    test_duration: float = None

@dataclass
class EfficiencyResult:
    duration: float = None
    params: float = None
    flops: float = None


def evaluate(ground_truth: np.ndarray, anomaly_scores: np.ndarray, subsequence: int,
             use_ucr: bool = False, mode: str = "train"):
    """
    Evaluate the performance of anomaly detection.
    :param ground_truth:
    :param anomaly_scores:
    :param use_ucr:
    :param subsequence: The subsequence length for UCR evaluation.
    :return:
    """
    all_results = EvaluationResult()

    # point-wise evaluation
    precision, recall, threshold = metrics.precision_recall_curve(y_true=ground_truth, y_score=anomaly_scores)
    threshold = threshold.astype(np.float64)
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)
    auc_prc = metrics.auc(recall, precision)
    auc_roc = metrics.roc_auc_score(y_true=ground_truth, y_score=anomaly_scores)

    all_results.point_wise = {
        "f1_score": np.max(f1_score),
        "auc_prc": auc_prc,
        "auc_roc": auc_roc,
        "others": {
            "precision": precision[np.argmax(f1_score)],
            "recall": recall[np.argmax(f1_score)],
            "threshold": threshold[np.argmax(f1_score)]
        }
    }

    # UCR evaluation
    if use_ucr:
        assert subsequence is not None, "If use UCR, ucr_distance should be provided."
        mean_value = np.mean(anomaly_scores)
        max_idx = np.argmax(anomaly_scores)

        peak_index, _ = find_peaks(x=anomaly_scores, distance=subsequence, height=mean_value)
        peak_scores = anomaly_scores[peak_index]
        peak_value = pd.DataFrame({"index": peak_index, "value": peak_scores})
        peak_value = peak_value.sort_values(by="value", ascending=False)
        anomaly_segments = get_segments(label_data=ground_truth)
        anomaly_segment = anomaly_segments[0]
        assert len(anomaly_segments) == 1, "UCR evaluation only supports one anomaly segment."
        margin_length = max(100, anomaly_segment[1] - anomaly_segment[0])
        anomaly_segment = np.arange(start=anomaly_segment[0] - margin_length, stop=anomaly_segment[1] + margin_length, step=1)
        ucr_01 = 1 if max_idx in anomaly_segment else 0
        ucr_03 = 1 if np.intersect1d(peak_value["index"][:3], anomaly_segment).size > 0 else 0
        ucr_05 = 1 if np.intersect1d(peak_value["index"][:5], anomaly_segment).size > 0 else 0

        all_results.ucr = {
            "ucr_01": ucr_01,
            "ucr_03": ucr_03,
            "ucr_05": ucr_05
        }

    # PATE evaluation
    if mode == "test":
        # TimeSeAD evaluator
        TimeSeAD_evaluator = Evaluator()

        # SeAD evaluation
        # TimeSeAD
        # SeAD_results = TimeSeAD_evaluator.best_ts_f1_score(labels=torch.tensor(ground_truth),
        #                                                    scores=torch.tensor(anomaly_scores))
        # all_results.seAD = {
        #     "seAD_F1": SeAD_results[0],
        #     "seAD_others": SeAD_results[1]
        # }

        # Range-based evaluation
        # Precision and recall for time series. Advances in neural information processing systems 31 (2018).
        # Range_results = TimeSeAD_evaluator.best_ts_f1_score_classic(labels=torch.tensor(ground_truth),
        #                                                             scores=torch.tensor(anomaly_scores))
        # all_results.range_based = {
        #     "range_F1": Range_results[0],
        #     "range_others": Range_results[1]
        # }

        # VUS score evaluation
        if use_ucr:
            subsequence = None
        vus_metrics = get_range_vus_roc(score=anomaly_scores, labels=ground_truth, subsequence_length=None)
        all_results.vus = vus_metrics

        # PATE evaluation
        # pate_metrics = PATE(y_true=ground_truth, y_score=anomaly_scores,
        #                     n_jobs=8, num_splits_MaxBuffer=10)
        # all_results.pate = pate_metrics
    return all_results


# Example usage:
if __name__ == '__main__':
    y_test = np.zeros(20000)
    # y_test[10:20] = 1
    # y_test[50:60] = 1
    y_test[900:1300] = 1
    anomaly_scores = np.random.randint(0, 500, 20000)/ 1000.0
    anomaly_scores[12:23] = 0.9 + np.random.rand(11)
    anomaly_scores[40:70] = 0.9 + np.random.rand(30)
    anomaly_scores[700:750] = 0.9 + np.random.rand(50)
    anomaly_scores[800:1200] = 0.9 + np.random.rand(400)
    import time
    start_time = time.time()
    eval_result = evaluate(ground_truth=y_test, anomaly_scores=anomaly_scores, use_ucr=True, subsequence=100,
                           mode="test")
    duration = time.time() - start_time
    eval_result.duration = duration
    print(f"Evaluation time: {duration:.2f}s")
    with open('eval_result.json', 'w') as f:
        json.dump(asdict(eval_result), f, indent=4)
