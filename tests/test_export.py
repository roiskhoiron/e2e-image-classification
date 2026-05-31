from pathlib import Path
import tensorflow as tf
from src.pipeline.model import build_cnn_model, compile_model
from src.pipeline.export_utils import export_saved_model, export_tflite


def _get_test_model():
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=2)
    compile_model(model)
    x = tf.random.normal((8, 32, 32, 3))
    y = tf.keras.utils.to_categorical([0, 1, 0, 1, 0, 1, 0, 1], num_classes=2)
    model.fit(x, y, epochs=1, verbose=0)
    return model


class TestExportSavedModel:
    def test_export_creates_saved_model_pb(self, tmp_path):
        model = _get_test_model()
        export_path = str(tmp_path / "saved_model")
        export_saved_model(model, export_path)
        assert (Path(export_path) / "saved_model.pb").exists()

    def test_saved_model_loadable(self, tmp_path):
        model = _get_test_model()
        export_path = str(tmp_path / "saved_model")
        export_saved_model(model, export_path)
        loaded = tf.saved_model.load(export_path)
        assert loaded is not None

    def test_saved_model_predicts(self, tmp_path):
        model = _get_test_model()
        export_path = str(tmp_path / "saved_model")
        export_saved_model(model, export_path)
        loaded = tf.saved_model.load(export_path)
        infer = loaded.signatures["serving_default"]
        x = tf.random.normal((2, 32, 32, 3))
        out = infer(x)
        assert len(out) > 0


class TestExportTFLite:
    def test_export_creates_file(self, tmp_path):
        model = _get_test_model()
        tflite_path = str(tmp_path / "model.tflite")
        export_tflite(model, tflite_path, quantize=True)
        assert Path(tflite_path).exists()

    def test_tflite_size_under_30mb(self, tmp_path):
        model = _get_test_model()
        tflite_path = str(tmp_path / "model.tflite")
        tflite_bytes = export_tflite(model, tflite_path, quantize=True)
        size_mb = len(tflite_bytes) / (1024 * 1024)
        assert size_mb <= 30

    def test_tflite_interpreter_loads(self, tmp_path):
        model = _get_test_model()
        tflite_path = str(tmp_path / "model.tflite")
        export_tflite(model, tflite_path, quantize=True)
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = tf.random.normal(input_details[0]["shape"], dtype=tf.float32).numpy()
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        assert output.shape == (1, 2)

    def test_quantized_has_expected_size(self, tmp_path):
        model = _get_test_model()
        q_path = str(tmp_path / "quantized.tflite")
        export_tflite(model, q_path, quantize=True)
        size_mb = Path(q_path).stat().st_size / (1024 * 1024)
        assert size_mb <= 30


class TestExportAll:
    def test_export_all_creates_all_artifacts(self, tmp_path):
        model = _get_test_model()
        base = tmp_path / "output"
        sm_path = str(base / "saved_model")
        tflite_path = str(base / "tflite" / "model.tflite")
        from src.pipeline.export_utils import export_all
        class_names = ["cat", "dog"]
        export_all(model, class_names=class_names, saved_model_path=sm_path, tflite_path=tflite_path)
        assert (Path(sm_path) / "saved_model.pb").exists()
        assert Path(tflite_path).exists()
        labels_path = Path(tflite_path).parent / "labels.txt"
        assert labels_path.exists()
        content = labels_path.read_text().strip().split("\n")
        assert content == class_names
