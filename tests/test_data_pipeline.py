import pytest
import shutil
import tempfile
from pathlib import Path
import tensorflow as tf


@pytest.fixture(scope="module")
def sample_dataset():
    tmp_dir = Path(tempfile.mkdtemp())
    classes = ["cat", "dog"]
    for cls in classes:
        for i in range(5):
            img_path = tmp_dir / cls / f"img_{i}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            arr = tf.random.uniform((64, 64, 3), 0, 255, dtype=tf.int32, seed=i)
            img_bytes = tf.image.encode_png(tf.cast(arr, tf.uint8)).numpy()
            img_path.write_bytes(img_bytes)
    yield tmp_dir
    shutil.rmtree(tmp_dir)


class TestBuildDataset:
    def test_returns_dataset(self, sample_dataset):
        from src.pipeline.data_pipeline import build_dataset

        ds = build_dataset(str(sample_dataset), image_size=(32, 32), batch_size=4)
        assert isinstance(ds, tf.data.Dataset)

    def test_image_shape(self, sample_dataset):
        from src.pipeline.data_pipeline import build_dataset

        ds = build_dataset(str(sample_dataset), image_size=(32, 32), batch_size=4)
        for images, labels in ds.take(1):
            assert images.shape == (4, 32, 32, 3)

    def test_labels_are_integer(self, sample_dataset):
        from src.pipeline.data_pipeline import build_dataset

        ds = build_dataset(str(sample_dataset), image_size=(32, 32), batch_size=4)
        for images, labels in ds.take(1):
            assert labels.dtype in (tf.int32, tf.int64)

    def test_rescaling(self, sample_dataset):
        from src.pipeline.data_pipeline import build_dataset

        ds = build_dataset(str(sample_dataset), image_size=(32, 32), batch_size=4)
        for images, labels in ds.take(1):
            assert tf.reduce_min(images) >= 0.0
            assert tf.reduce_max(images) <= 1.0

    def test_augmentation_pipeline_runs(self, sample_dataset):
        from src.pipeline.data_pipeline import build_dataset

        ds = build_dataset(
            str(sample_dataset), image_size=(32, 32), batch_size=4, augment=True
        )
        for images, labels in ds.take(1):
            assert images.shape == (4, 32, 32, 3)

    def test_batch_count(self, sample_dataset):
        from src.pipeline.data_pipeline import build_dataset

        ds = build_dataset(str(sample_dataset), image_size=(32, 32), batch_size=2)
        batches = sum(1 for _ in ds)
        assert batches == 5  # 10 images / batch_size 2 = 5


class TestGetTrainValTest:
    def test_returns_three_datasets(self, sample_dataset):
        from src.pipeline.data_pipeline import get_train_val_test

        data_dir = sample_dataset / "processed"
        for split in ("train", "val", "test"):
            for cls in ("cat", "dog"):
                (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
        # Copy 3 train, 1 val, 1 test per class
        for cls in ("cat", "dog"):
            idx = 0
            for split, count in [("train", 3), ("val", 1), ("test", 1)]:
                for _ in range(count):
                    src = sample_dataset / cls / f"img_{idx}.png"
                    if src.exists():
                        shutil.copy2(src, data_dir / split / cls / f"img_{idx}.png")
                    idx += 1
        train_ds, val_ds, test_ds = get_train_val_test(
            data_dir=str(data_dir), image_size=(32, 32), batch_size=2
        )
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)
