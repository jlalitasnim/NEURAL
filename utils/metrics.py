import numpy as np


def accuracy(y_true, y_pred):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        return np.mean(
            np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
        )

    return np.mean(
        (y_pred >= 0.5).astype(int).flatten() ==
        y_true.flatten().astype(int)
    )


def confusion_matrix(y_true, y_pred, n_classes=2):
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_cls = np.argmax(y_true, axis=1)
    else:
        y_pred_cls = (y_pred >= 0.5).astype(int).flatten()
        y_true_cls = y_true.flatten().astype(int)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true_cls, y_pred_cls):
        cm[t][p] += 1

    return cm


def precision_recall_f1(y_true, y_pred):
    n_classes = (
        y_true.shape[1]
        if (y_true.ndim > 1 and y_true.shape[1] > 1)
        else 2
    )

    cm = confusion_matrix(y_true, y_pred, n_classes)

    precisions, recalls, f1s = [], [], []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp + 1e-15)
        r = tp / (tp + fn + 1e-15)
        f1 = 2 * p * r / (p + r + 1e-15)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def mse_metric(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mse_metric(y_true, y_pred)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return float(1 - ss_res / (ss_tot + 1e-15))