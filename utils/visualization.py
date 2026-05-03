import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss_accuracy(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"],
                             name="Train Loss"), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                             name="Val Loss"), row=1, col=1)

    # Accuracy
    if "train_acc" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"],
                                 name="Train Acc"), row=1, col=2)

        fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"],
                                 name="Val Acc"), row=1, col=2)

    fig.update_layout(height=400, template="plotly_white")
    return fig


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)

    if preds.ndim > 1 and preds.shape[1] > 1:
        Z = np.argmax(preds, axis=1)
    else:
        Z = (preds >= 0.5).astype(int).flatten()

    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        showscale=False,
        opacity=0.4
    ))

    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode="markers",
        marker=dict(color=y.flatten(), size=6)
    ))

    fig.update_layout(template="plotly_white")
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig


def plot_regression(y_true, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true.flatten(),
        y=y_pred.flatten(),
        mode="markers",
        name="Predictions"
    ))

    min_val = float(y_true.min())
    max_val = float(y_true.max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect fit"
    ))

    fig.update_layout(template="plotly_white")
    return fig


def plot_compare_losses(histories):
    fig = go.Figure()

    for name, hist in histories.items():
        epochs = list(range(1, len(hist["train_loss"]) + 1))

        fig.add_trace(go.Scatter(
            x=epochs,
            y=hist["train_loss"],
            name=name
        ))

    fig.update_layout(template="plotly_white")
    return fig