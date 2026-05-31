import pytest
import numpy as np
import tensorflow as tf
from src.pipeline.model import build_cnn_model, compile_model
from src.pipeline.evaluation import plot_training_history, evaluate_model


def test_plot_training_history(tmp_path):
    class FakeHistory:
        history = {
            "accuracy": [0.5, 0.7, 0.9],
            "val_accuracy": [0.4, 0.6, 0.8],
            "loss": [1.0, 0.6, 0.3],
            "val_loss": [1.2, 0.8, 0.4],
        }
    plot_training_history(FakeHistory(), save_dir=str(tmp_path))
    assert (tmp_path / "accuracy_loss_curves.png").exists()


class TestEvaluateModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = build_cnn_model(input_shape=(32, 32, 3), num_classes=2)
        compile_model(self.model)
        x = tf.random.normal((32, 32, 32, 3))
        y = tf.keras.utils.to_categorical([0] * 16 + [1] * 16, num_classes=2)
        self.model.fit(x, y, epochs=1, verbose=0)
        self.x_test = x
        self.y_int = tf.argmax(y, axis=1)

    def test_evaluate_returns_accuracy(self, tmp_path):
        ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_int)).batch(8)
        result = evaluate_model(self.model, ds, save_dir=str(tmp_path))
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_saved(self, tmp_path):
        ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_int)).batch(8)
        evaluate_model(self.model, ds, save_dir=str(tmp_path))
        assert (tmp_path / "confusion_matrix.png").exists()

    def test_classification_report_saved(self, tmp_path):
        ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_int)).batch(8)
        evaluate_model(
            self.model, ds, class_names=["cat", "dog"], save_dir=str(tmp_path)
        )
        report_path = tmp_path / "classification_report.txt"
        assert report_path.exists()
        content = report_path.read_text()
        assert "cat" in content
        assert "dog" in content
