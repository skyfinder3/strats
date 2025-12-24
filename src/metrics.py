import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

def compute_classification_metrics(true, pred):
    """
    Compute AUROC, PR curve, precision/recall, and ROC curve.
    Input:
        true: numpy array of shape (N,)
        pred: numpy array of shape (N,)
    Output:
        metrics: dict with AUROC, AUPRC, minRP, ROC curve arrays, PR curve arrays
    """
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(true, pred)
    roc_auc = roc_auc_score(true, pred)

    # PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(true, pred)
    pr_auc = auc(recall, precision)

    # min(precision, recall) max on PR curve
    minrp = np.minimum(precision, recall).max()

    return {
        "auroc": roc_auc,
        "auprc": pr_auc,
        "minrp": minrp,
        "roc_curve": (fpr, tpr, roc_thresholds),
        "pr_curve": (precision, recall, pr_thresholds)
    }
