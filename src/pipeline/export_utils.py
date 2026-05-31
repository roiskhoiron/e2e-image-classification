from pathlib import Path
import tensorflow as tf


def export_saved_model(model: tf.keras.Model, export_path: str = "models_output/saved_model") -> None:
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(export_dir))
    print(f"SavedModel exported to {export_dir}")


def export_tflite(
    model: tf.keras.Model,
    export_path: str = "models_output/tflite/model.tflite",
    quantize: bool = True,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    export_file = Path(export_path)
    export_file.parent.mkdir(parents=True, exist_ok=True)
    export_file.write_bytes(tflite_model)
    print(f"TFLite model exported to {export_file} ({len(tflite_model)} bytes)")
    return tflite_model


def export_tfjs(
    saved_model_dir: str = "models_output/saved_model",
    export_dir: str = "models_output/tfjs",
) -> None:
    import subprocess
    import sys
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        str(Path(saved_model_dir).resolve()),
        str(export_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TFJS conversion failed: {result.stderr}")
    print(f"TFJS model exported to {export_path}")


def export_all(
    model: tf.keras.Model,
    class_names: list[str] | None = None,
    saved_model_path: str = "models_output/saved_model",
    tflite_path: str = "models_output/tflite/model.tflite",
    tfjs_path: str = "models_output/tfjs",
) -> dict:
    export_saved_model(model, saved_model_path)
    tflite_bytes = export_tflite(model, tflite_path)
    sizes = {
        "saved_model_dir": str(Path(saved_model_path).resolve()),
        "tflite_bytes": len(tflite_bytes),
    }
    if class_names:
        labels_path = Path(tflite_path).parent / "labels.txt"
        labels_path.write_text("\n".join(class_names))
        sizes["labels_file"] = str(labels_path)
    try:
        export_tfjs(saved_model_path, tfjs_path)
        sizes["tfjs_dir"] = str(Path(tfjs_path).resolve())
    except (ImportError, RuntimeError) as e:
        print(f"TFJS export skipped: {e}")
    return sizes
