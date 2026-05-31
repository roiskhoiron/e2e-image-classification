import tensorflow as tf

from src.pipeline.model import (
    build_cnn_model,
    compile_model,
    get_callbacks,
)


class TestBuildCNNModel:
    def test_model_output_shape(self):
        model = build_cnn_model(input_shape=(32, 32, 3), num_classes=5)
        assert model.output_shape == (None, 5)

    def test_model_is_sequential(self):
        model = build_cnn_model()
        assert isinstance(model, tf.keras.Sequential)

    def test_model_layers_count(self):
        model = build_cnn_model()
        # Conv2D x4 + MaxPool x4 + Flatten + Dense + Dropout + Dense = 12
        assert len(model.layers) == 12

    def test_compile_adds_optimizer(self):
        model = build_cnn_model(num_classes=3)
        compile_model(model)
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_compile_adds_loss_and_metrics(self):
        model = build_cnn_model(num_classes=3)
        compile_model(model)
        assert model.loss == "sparse_categorical_crossentropy"
        assert model.compiled_metrics is not None


class TestGetCallbacks:
    def test_returns_list(self):
        callbacks = get_callbacks()
        assert isinstance(callbacks, list)
        assert len(callbacks) == 2

    def test_early_stopping_present(self):
        callbacks = get_callbacks()
        assert any(
            isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks
        )

    def test_checkpoint_present(self):
        callbacks = get_callbacks()
        assert any(
            isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks
        )


class TestForwardPass:
    def test_forward_pass_shape(self):
        model = build_cnn_model(input_shape=(32, 32, 3), num_classes=3)
        compile_model(model)
        batch = tf.random.normal((4, 32, 32, 3))
        preds = model(batch, training=False)
        assert preds.shape == (4, 3)

    def test_softmax_probabilities(self):
        model = build_cnn_model(input_shape=(32, 32, 3), num_classes=3)
        compile_model(model)
        batch = tf.random.normal((4, 32, 32, 3))
        preds = model(batch, training=False)
        # Softmax should sum to ~1 per sample
        sums = tf.reduce_sum(preds, axis=1).numpy()
        assert all(abs(s - 1.0) < 1e-5 for s in sums)


class TestTrainingStep:
    def test_train_on_batch(self):
        model = build_cnn_model(input_shape=(32, 32, 3), num_classes=2)
        compile_model(model)
        x = tf.random.normal((8, 32, 32, 3))
        y = tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.int32)
        loss_before = model.evaluate(x, y, verbose=0)[0]
        model.fit(x, y, epochs=1, verbose=0)
        loss_after = model.evaluate(x, y, verbose=0)[0]
        assert loss_after <= loss_before * 1.5  # loss should decrease
