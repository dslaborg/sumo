import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .performance_analysis import PerformanceEvaluation, metric_scores


def get_stats(model, dataloader, overlap_thresholds):
    model.eval()

    n_spindles_detected, n_spindles_gs, n_true_positives = 0, 0, np.zeros_like(overlap_thresholds, dtype=int)
    with torch.no_grad():
        for batch in dataloader:
            data, mask_true = batch

            mask_pred = model(data)

            spindles_pred_batch = F.softmax(mask_pred, dim=1).argmax(dim=1).long().detach().cpu().numpy()
            spindles_gs_batch = mask_true.long().detach().cpu().numpy()

            for spindles_pred, spindles_gs in zip(spindles_pred_batch, spindles_gs_batch):
                res = PerformanceEvaluation(spindles_pred, spindles_gs, overlap_thresholds).evaluate_performance()
                n_spindles_detected += res[0]
                n_spindles_gs += res[1]
                n_true_positives += res[2]

    return n_spindles_detected, n_spindles_gs, n_true_positives


def calculate_metrics(model, dataloader, overlap_thresholds):
    n_spindles_detected, n_spindles_gs, n_true_positives = get_stats(model, dataloader, overlap_thresholds)

    precision, recall, f1 = metric_scores(n_spindles_detected, n_spindles_gs, n_true_positives)
    return precision, recall, f1


def calculate_test_metrics(model, dataloaders, overlap_thresholds):
    n_spindles_detected_all, n_spindles_gs_all, n_true_positives_all = 0, 0, np.zeros_like(overlap_thresholds, dtype=int)
    precisions, recalls, f1s = [], [], []

    for dataloader in dataloaders:
        n_spindles_detected, n_spindles_gs, n_true_positives = get_stats(model, dataloader, overlap_thresholds)
        n_spindles_detected_all += n_spindles_detected
        n_spindles_gs_all += n_spindles_gs
        n_true_positives_all += n_true_positives

        precision, recall, f1 = metric_scores(n_spindles_detected, n_spindles_gs, n_true_positives)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precision, recall, f1 = metric_scores(n_spindles_detected_all, n_spindles_gs_all, n_true_positives_all)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    return precisions, recalls, f1s


def plot_metrics(precision, recall, f1, overlap_thresholds):
    fig, ax = plt.subplots(figsize=(4, 5))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid()

    ax.set_xlabel('Overlap Threshold')
    ax.plot(overlap_thresholds, precision, label='precision')
    ax.plot(overlap_thresholds, recall, label='recall')
    ax.plot(overlap_thresholds, f1, label='f1')
    ax.legend()

    fig.tight_layout()
    fig.show()
