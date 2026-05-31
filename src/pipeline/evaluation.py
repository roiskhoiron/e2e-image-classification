from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf


def plot_training_history(history: tf.keras.callbacks.History, save_dir: str = "reports/figures") -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(str(save_path / "accuracy_loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path / 'accuracy_loss_curves.png'}")


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str] | None = None,
    save_dir: str = "reports/figures",
) -> dict:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Collect all predictions and labels
    all_preds = []
    all_labels = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate([np.argmax(p, axis=1) for p in all_preds])
    y_true = np.concatenate(all_labels)

    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted Label", ylabel="True Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(str(save_path / "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path / 'confusion_matrix.png'}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_path = save_path / "classification_report.txt"
    report_path.write_text(report)
    print(f"Saved classification report to {report_path}")

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    return {"accuracy": float(accuracy), "confusion_matrix": cm.tolist()}
