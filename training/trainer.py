import numpy as np
from training.optimizer import get_batches
from utils.loss import get_loss
from utils.metrics import accuracy, mse_metric


def train(model, X_train, y_train, X_val, y_val,
          loss_name,
          problem_type,
          learning_rate=0.01,
          epochs=100,
          batch_size=32,
          patience=None,
          callback=None):

    loss_fn, loss_grad_fn = get_loss(loss_name)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            A_out, cache = model.forward(X_batch, training=True)

            grads_W, grads_b = model.backward(
                A_out,
                y_batch,
                cache,
                loss_grad_fn
            )

            model.update(grads_W, grads_b, learning_rate)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_loss = float(loss_fn(y_train, train_pred))
        val_loss = float(loss_fn(y_val, val_pred))

        if problem_type in ["binary", "multiclass"]:
            train_acc = float(accuracy(y_train, train_pred))
            val_acc = float(accuracy(y_val, val_pred))
        else:
            train_acc = float(
                1 - mse_metric(y_train, train_pred) / (np.var(y_train) + 1e-8)
            )

            val_acc = float(
                1 - mse_metric(y_val, val_pred) / (np.var(y_val) + 1e-8)
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights_copy()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    if best_weights is not None:
                        model.set_weights(*best_weights)
                    break

        if callback is not None:
            callback(epoch + 1, epochs, train_loss, val_loss)

    return history